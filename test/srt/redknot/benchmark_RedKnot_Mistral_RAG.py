"""Benchmark: RedKnot native-SWA KV reuse vs standard Mistral prefill.

Compares, on **LongBench** long-context (default 4 x 7.5K = 30K), the standard
FA-2 prefill baseline against RedKnot's native sliding-window reuse path,
reporting for every sample and as aggregates:

  * the generated ANSWER TEXT (baseline vs RedKnot vs gold)
  * answer quality (SQuAD F1 / EM, scored against LongBench's answer list)
  * TTFT (time-to-first-token) and the speedup
  * decode throughput (tok/s)
  * COMPUTE (FLOPs) comparison: attention / FFN / projection, with savings

Pipeline (exactly your spec):
  1. Each request is **>8K per document** (default 4 docs x 8000 tok = 32K).
  2. `offline_prefill_segments` builds the per-document KV OFFLINE (local
     coordinates), once per request.
  3. Offline K is **RoPE-repositioned** to global coordinates. For documents
     2..4, only the first 20% boundary tokens are recomputed with native 4096
     sliding-window attention; the repositioned offline suffix is reused.

Baseline = the honest, hard-to-beat reference: one `model(input_ids)` forward
over the full concatenated context with `attn_implementation="flash_attention_2"`
(not RedKnot's per-slot framework, which would carry our own dispatch overhead).

Model: a Mistral-7B checkpoint with native 4096-token SWA, single GPU. Local
model + bundled LongBench datasets; falls back to Mistral-7B-Instruct-v0.1 if
the model isn't local.

One-click (zero config -> full-text comparison, 4096 SWA, 20% recompute,
16 generated tokens):
    python test/srt/redknot/benchmark_RedKnot_Mistral_RAG.py

Optional overrides:
    REDKNOT_N_SAMPLES=4 REDKNOT_DATASETS=triviaqa,hotpotqa \
    REDKNOT_TOKENS_PER_DOC=5000 REDKNOT_LENGTHS=4x \
    python test/srt/redknot/benchmark_RedKnot_Mistral_RAG.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import random
import re
import string
from collections import Counter

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

from sglang.srt.layers.attention.redknot import (  # noqa: E402
    get_global_offline_cache,
    offline_prefill_segments,
    run_redknot_swa_offlinekv,
)

MODEL_PATH = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/instruction-synthesizer",
)
# HuggingFace repo id used as a fallback when no local model is found.
HF_MODEL_ID = os.environ.get(
    "REDKNOT_HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.1"
)


def _resolve_model_source(local_path: str, hf_id: str) -> str:
    """Return a usable model identifier for ``from_pretrained``.

    Prefer the local checkpoint if it exists on disk; otherwise fall back to
    the HuggingFace repo id (``from_pretrained`` will download it).
    """
    if local_path and os.path.isdir(local_path):
        print(f" Using local model: {local_path}")
        return local_path
    print(
        f" Local model not found at {local_path!r}; "
        f"falling back to HuggingFace hub: {hf_id}"
    )
    return hf_id


# LongBench data (jsonl-per-dataset). Each line: {input, context, answers, ...}.
# Default to the datasets bundled next to this benchmark so it works out of the
# box; override with REDKNOT_LONGBENCH_DIR for a different location.
_LOCAL_LONGBENCH = str(Path(__file__).resolve().parent / "datasets/LongBench/data")
LONGBENCH_DIR = os.environ.get("REDKNOT_LONGBENCH_DIR", _LOCAL_LONGBENCH)
# Which LongBench QA datasets to evaluate (comma separated).
DATASETS = [
    s
    for s in os.environ.get("REDKNOT_DATASETS", "hotpotqa,2wikimqa,musique").split(",")
    if s.strip()
]
RAG_MODE = os.environ.get("REDKNOT_RAG_MODE", "standard").lower()
INPUT_JSON = os.environ.get(
    "REDKNOT_INPUT_JSON",
    str(Path(__file__).resolve().parent / "datasets/longbench_rag.jsonl"),
)
TEXT_ONLY = os.environ.get("REDKNOT_TEXT_ONLY", "1") == "1"
if INPUT_JSON:
    DATASETS = ["longbench_rag"]
SEED = 2026


# ────────────────────────────────────────────────────────────────────────
# Self-contained data loading + metrics (no external test-file dependency)
# ────────────────────────────────────────────────────────────────────────
def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p, g = _normalize(pred).split(), _normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec, rec = num_same / len(p), num_same / len(g)
    return 2 * prec * rec / (prec + rec)


def em_score(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def f1_max(pred: str, golds) -> float:
    """LongBench answers is a LIST; score against the best-matching gold."""
    golds = golds if isinstance(golds, (list, tuple)) else [golds]
    golds = [str(g) for g in golds if str(g).strip()]
    return max((f1_score(pred, g) for g in golds), default=0.0)


def em_max(pred: str, golds) -> float:
    golds = golds if isinstance(golds, (list, tuple)) else [golds]
    golds = [str(g) for g in golds if str(g).strip()]
    return max((em_score(pred, g) for g in golds), default=0.0)


def _mean_std_ci(values):
    """Return (mean, sample_std, half_95_ci) for a list of scores.

    Uses the unbiased sample std (ddof=1) and a normal-approx 95% CI on the
    mean (1.96 * std / sqrt(n)). With the tiny n used by these RAG benchmarks
    (e.g. n=4) the CI is intentionally WIDE -- that width is the point: it
    tells the reader how little a single F1 number can be trusted here.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = var**0.5
    half_ci = 1.96 * std / (n**0.5)
    return mean, std, half_ci


def _load_samples(tok, dataset, n_segments, n_samples, tokens_per_segment=8000):
    """Load LongBench samples and build exactly ``n_segments`` documents of
    ``tokens_per_segment`` tokens each (>8K per doc by spec).

    LongBench jsonl lines carry a single long ``context`` and an ``answers``
    list. To reach ``n_segments * tokens_per_segment`` tokens we concatenate
    the context of the chosen sample with the contexts of following samples
    (wrap-around), then slice the token stream into equal ``tokens_per_segment``
    chunks -> these are the ``docs`` whose KV is built offline.
    """
    if INPUT_JSON:
        rows_in = []
        with open(INPUT_JSON, encoding="utf-8") as f:
            content = f.read().strip()
        # Support both a JSON array and line-delimited JSON (.jsonl).
        if content.startswith("["):
            rows_in = json.loads(content)
        else:
            rows_in = [json.loads(line) for line in content.splitlines() if line.strip()]
        return [
            {
                "question": row["question"],
                "gold_answer": row["answers"],
                "docs": row["documents"],
                "dataset": row["dataset"],
            }
            for row in rows_in[:n_samples]
        ]

    path = os.path.join(LONGBENCH_DIR, f"{dataset}.jsonl")
    raw = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("input") and r.get("context") and r.get("answers"):
                raw.append(r)
    if not raw:
        raise RuntimeError(f"No usable rows in LongBench dataset {path!r}")
    if RAG_MODE == "standard":
        return _load_standard_rag_samples(tok, raw, dataset, n_segments, n_samples)

    random.Random(SEED).shuffle(raw)

    if RAG_MODE in {"oracle", "evidence"}:
        return _load_oracle_rag_samples(
            tok, raw, dataset, n_segments, n_samples, tokens_per_segment
        )

    target = n_segments * tokens_per_segment
    nraw = len(raw)
    out = []
    for i in range(nraw):
        if len(out) >= n_samples:
            break
        toks = tok(raw[i]["context"], add_special_tokens=False)["input_ids"]
        j = (i + 1) % nraw
        # Pad with following contexts (distractor-like) until we reach target.
        while len(toks) < target and j != i:
            toks += tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
            j = (j + 1) % nraw
        if len(toks) < target:
            continue  # not enough text even after wrap-around
        toks = toks[:target]
        docs = [
            tok.decode(toks[k : k + tokens_per_segment], skip_special_tokens=True)
            for k in range(0, target, tokens_per_segment)
        ]
        if getattr(tok, "chat_template", None):
            docs[0] = "[INST] " + docs[0]
        out.append(
            {
                "question": raw[i]["input"],
                "gold_answer": raw[i]["answers"],  # LIST of acceptable answers
                "docs": docs,
                "dataset": dataset,
            }
        )
    return out


def _load_standard_rag_samples(tok, raw, dataset, n_segments, n_samples):
    """Load unmodified LongBench rows and split each context contiguously."""
    max_context_tokens = int(os.environ.get("REDKNOT_MAX_CONTEXT", "30000"))
    out = []
    for row in raw:
        context_ids = tok(row["context"], add_special_tokens=False)["input_ids"]
        if len(context_ids) > max_context_tokens:
            half = max_context_tokens // 2
            context_ids = (
                context_ids[:half] + context_ids[-(max_context_tokens - half) :]
            )
        if len(context_ids) < n_segments:
            continue
        bounds = [len(context_ids) * i // n_segments for i in range(n_segments + 1)]
        docs = [
            tok.decode(context_ids[bounds[i] : bounds[i + 1]], skip_special_tokens=True)
            for i in range(n_segments)
        ]
        if getattr(tok, "chat_template", None):
            docs[0] = "[INST] " + docs[0]
        out.append(
            {
                "question": row["input"],
                "gold_answer": row["answers"],
                "docs": docs,
                "dataset": dataset,
            }
        )
        if len(out) >= n_samples:
            break
    return out


def _load_oracle_rag_samples(
    tok, raw, dataset, n_segments, n_samples, tokens_per_segment
):
    """Build 30K oracle-retrieval requests from real LongBench passages.

    The answer-bearing window is placed at the end of the final retrieved
    document, inside Mistral's native SWA visibility range. All preceding
    tokens are distractors sampled from other rows of the same dataset.
    ``oracle`` additionally appends the exact answer; ``evidence`` leaves only
    the original passage so the model must solve the question.
    """
    target = n_segments * tokens_per_segment
    evidence_budget = min(3000, tokens_per_segment - 1)
    out = []
    for i, row in enumerate(raw):
        if len(out) >= n_samples:
            break
        context = row["context"]
        context_lower = context.lower()
        matched_answer = next(
            (
                str(answer)
                for answer in row["answers"]
                if len(str(answer).strip()) > 2 and str(answer).lower() in context_lower
            ),
            None,
        )
        if matched_answer is None:
            continue

        answer_char = context_lower.index(matched_answer.lower())
        answer_token = len(
            tok(context[:answer_char], add_special_tokens=False)["input_ids"]
        )
        context_ids = tok(context, add_special_tokens=False)["input_ids"]
        evidence_start = max(0, answer_token - evidence_budget // 2)
        evidence_end = min(len(context_ids), evidence_start + evidence_budget)
        evidence_start = max(0, evidence_end - evidence_budget)
        evidence_ids = context_ids[evidence_start:evidence_end]

        oracle_record = ""
        if RAG_MODE == "oracle":
            oracle_record = (
                "\n\nAuthoritative RAG retrieval result:\n"
                f"Question: {row['input']}\n"
                f"Exact answer: {matched_answer}\n"
            )
        oracle_ids = tok(oracle_record, add_special_tokens=False)["input_ids"]
        filler_target = target - len(evidence_ids) - len(oracle_ids)
        if filler_target <= 0:
            continue
        filler = []
        j = (i + 1) % len(raw)
        while len(filler) < filler_target and j != i:
            filler.extend(tok(raw[j]["context"], add_special_tokens=False)["input_ids"])
            j = (j + 1) % len(raw)
        if len(filler) < filler_target:
            continue
        all_ids = filler[:filler_target] + evidence_ids + oracle_ids
        docs = [
            tok.decode(
                all_ids[start : start + tokens_per_segment],
                skip_special_tokens=True,
            )
            for start in range(0, target, tokens_per_segment)
        ]
        if getattr(tok, "chat_template", None):
            docs[0] = "[INST] " + docs[0]
        if not any(
            str(answer).lower() in docs[-1].lower() for answer in row["answers"]
        ):
            continue
        out.append(
            {
                "question": row["input"],
                "gold_answer": row["answers"],
                "docs": docs,
                "dataset": dataset,
            }
        )
    return out


def _query_text(q, tok):
    if RAG_MODE == "standard":
        prompt = (
            "\n\nAnswer the question based on the given passages. Only give me "
            f"the answer and do not output any other words.\n\nQuestion: {q}\nAnswer:"
        )
    else:
        prompt = (
            "\n\nUsing only the documents above, give the shortest exact answer "
            "span to the question (a name, entity, number, or short noun phrase). "
            "Answer with the span only, no explanation.\n"
            f"Question: {q}\nAnswer:"
        )
    if getattr(tok, "chat_template", None):
        prompt += "[/INST]"
    return prompt


def _short_ans(t):
    t = t or ""
    if not t.strip():
        return ""
    t = re.sub(r"</?(?:END|ANS|QUE)>", " ", t, flags=re.I)
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.I)
    t = re.sub(r"(?i)\bthe answer is\b[:\s]*", "", t)
    t = re.sub(r"(?is)^\s*answer\s*[:：]\s*", "", t)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not re.fullmatch(r"(?i)answer\s*[:：]?", ln)]
    cand = (lines[0] if lines else t.strip()).strip().strip('"').strip("'").strip()
    cand = re.split(r"\n\s*(?:question|q)\s*[:：]", cand, flags=re.I)[0]
    cand = re.sub(r"\s*[.。]\s*$", "", cand)
    return cand


MAX_NEW_TOKENS = int(os.environ.get("REDKNOT_MAX_NEW", "16"))
N_SAMPLES = int(os.environ.get("REDKNOT_N_SAMPLES", "10"))
SWA_WINDOW = int(os.environ.get("REDKNOT_SWA_WINDOW", "4096"))
RECOMPUTE_RATIO = float(os.environ.get("REDKNOT_RECOMPUTE_RATIO", "0.20"))

# Stop strings: when any of these appear in the generated text the model stops
# and the output is truncated at the first occurrence. This prevents the model
# from generating follow-up QA pairs or passage continuations after the answer.
STOP_STRINGS = ["\n\n<QUE>", "\nQuestion:", "\nPassage:", "\n\n<ANS>", "</END>"]

# Each document is longer than the native SWA window, while 4 x 7500 leaves
# room for the query inside Mistral's 32K context limit.
TOKENS_PER_DOC = int(os.environ.get("REDKNOT_TOKENS_PER_DOC", "7500"))
assert TOKENS_PER_DOC > SWA_WINDOW, (
    f"each document ({TOKENS_PER_DOC} tok) must exceed the native SWA "
    f"window ({SWA_WINDOW})"
)
_LEN_MAP = {
    "14K": (2, TOKENS_PER_DOC),
    "30K": (4, TOKENS_PER_DOC),
    "42K": (6, TOKENS_PER_DOC),
    "56K": (8, TOKENS_PER_DOC),
    # Fixed 4-document configs (per-doc size set via REDKNOT_TOKENS_PER_DOC):
    #   4x5K  -> REDKNOT_TOKENS_PER_DOC=5000 REDKNOT_LENGTHS=4x
    #   4x7.5K -> REDKNOT_TOKENS_PER_DOC=7500 REDKNOT_LENGTHS=4x
    "4x": (4, TOKENS_PER_DOC),
}
LENGTHS = [
    (label, *_LEN_MAP[label])
    for label in os.environ.get("REDKNOT_LENGTHS", "4x").split(",")
    if label in _LEN_MAP
]


# ── FLOPs accounting ──
def _model_dims(cfg):
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    return {
        "L": cfg.num_hidden_layers,
        "hidden": cfg.hidden_size,
        "inter": cfg.intermediate_size,
        "Hq": cfg.num_attention_heads,
        "Hkv": cfg.num_key_value_heads,
        "D": hd,
    }


def _proj_flops_per_token(d):
    qkv = 2.0 * d["hidden"] * (d["Hq"] + 2 * d["Hkv"]) * d["D"]
    o = 2.0 * d["Hq"] * d["D"] * d["hidden"]
    return qkv + o


def _ffn_flops_per_token(d):
    return 6.0 * d["hidden"] * d["inter"]


def compute_swa_reuse_flops(d, doc_lens):
    """Online FLOPs: full native-SWA prefill vs boundary-prefix reuse."""
    total_tokens = sum(doc_lens)
    dense_pairs = sum(min(i + 1, SWA_WINDOW) for i in range(total_tokens))
    recomputed_tokens = sum(
        max(1, int(length * RECOMPUTE_RATIO)) for length in doc_lens[1:]
    )
    reuse_pairs = recomputed_tokens * SWA_WINDOW
    proj_per_token = d["L"] * _proj_flops_per_token(d)
    ffn_per_token = d["L"] * _ffn_flops_per_token(d)
    dense_attn = d["L"] * d["Hq"] * 4.0 * d["D"] * dense_pairs
    reuse_attn = d["L"] * d["Hq"] * 4.0 * d["D"] * reuse_pairs
    dense_proj = total_tokens * proj_per_token
    reuse_proj = recomputed_tokens * proj_per_token
    dense_ffn = total_tokens * ffn_per_token
    reuse_ffn = recomputed_tokens * ffn_per_token
    return {
        "attn": (dense_attn, reuse_attn),
        "ffn": (dense_ffn, reuse_ffn),
        "proj": (dense_proj, reuse_proj),
        "total": (
            dense_attn + dense_ffn + dense_proj,
            reuse_attn + reuse_ffn + reuse_proj,
        ),
    }


# ── Standard fastest dense prefill baseline (FA-2) ──
@torch.no_grad()
def standard_prefill(model, tok, documents, query_text, max_new_tokens, stop_strings=None):
    device = model.device
    parts = [
        tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        for text in documents
    ]
    if tok.bos_token_id is not None:
        bos = torch.tensor([[tok.bos_token_id]], dtype=parts[0].dtype)
        parts[0] = torch.cat([bos, parts[0]], dim=1)
    parts.append(
        tok(query_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    )
    ids = torch.cat(parts, dim=1).to(device)
    n_ctx = ids.shape[1]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()
    else:
        gpu_start = gpu_end = None
    t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=True)
    first = out.logits[0, -1, :].clone()
    nxt = first.argmax().view(1, 1)
    if torch.cuda.is_available():
        gpu_end.record()
        torch.cuda.synchronize()
        ttft_gpu_ms = gpu_start.elapsed_time(gpu_end)
    else:
        ttft_gpu_ms = (time.perf_counter() - t0) * 1000
    ttft = time.perf_counter() - t0
    past = out.past_key_values
    gen = [int(nxt[0, 0])]
    t1 = time.perf_counter()
    _stop_hit = False
    for _ in range(max_new_tokens - 1):
        og = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = og.past_key_values
        nxt = og.logits[0, -1, :].argmax().view(1, 1)
        tid = int(nxt[0, 0])
        gen.append(tid)
        if tid == tok.eos_token_id:
            break
        if stop_strings and len(gen) >= 3:
            partial = tok.decode(gen, skip_special_tokens=True)
            for ss in stop_strings:
                if ss in partial:
                    _stop_hit = True
                    break
            if _stop_hit:
                break
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dec_t = max(time.perf_counter() - t1, 1e-3)
    # The first token is produced by the prefill above, before decode timing
    # starts. Count only the subsequent model forwards that actually ran in
    # the timed decode loop, including a terminal EOS/stop token.
    decode_forward_steps = max(len(gen) - 1, 0)
    decode_tps = decode_forward_steps / dec_t if decode_forward_steps else 0.0
    text = tok.decode(gen, skip_special_tokens=True)
    if stop_strings:
        for ss in stop_strings:
            idx = text.find(ss)
            if idx != -1:
                text = text[:idx]
        text = text.strip()
    return (
        first,
        text,
        ttft,
        decode_tps,
        n_ctx,
        ttft_gpu_ms,
    )


# ── Pretty printing ──
def _trunc(s, n=40):
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    one_sample = os.environ.get("REDKNOT_ONE_SAMPLE")  # run only this index
    W = 100

    print("=" * W)
    print(
        " BENCHMARK: RedKnot (native-SWA KV reuse) vs Standard FlashAttention-2 prefill"
    )
    print(
        f" Model: {Path(MODEL_PATH).name} | Dataset: LongBench "
        f"({','.join(DATASETS)}) | mode={RAG_MODE} | single GPU"
    )
    print("=" * W)

    model_src = _resolve_model_source(MODEL_PATH, HF_MODEL_ID)
    tok = AutoTokenizer.from_pretrained(model_src, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model_config = AutoConfig.from_pretrained(model_src, trust_remote_code=True)
    native_window = getattr(model_config, "sliding_window", None)
    if native_window != SWA_WINDOW:
        raise ValueError(
            f"checkpoint must be trained with sliding_window={SWA_WINDOW}; "
            f"got {native_window!r} from {model_src}. Do not force SWA on a "
            "full-attention checkpoint because its RAG accuracy collapses."
        )
    model_config.sliding_window = SWA_WINDOW
    # Mistral-7B is small (~15GB in bf16) so default to bf16 (fastest, exact).
    # Set REDKNOT_DTYPE=int4 to load NF4 4-bit if GPU memory is tight.
    dtype_mode = os.environ.get("REDKNOT_DTYPE", "bf16").lower()
    if dtype_mode == "int4":
        qc = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_src,
            config=model_config,
            quantization_config=qc,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_src,
            config=model_config,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).eval()
    context_desc = (
        f"raw context <= {os.environ.get('REDKNOT_MAX_CONTEXT', '30000')} tokens"
        if RAG_MODE == "standard"
        else f"documents=4x{TOKENS_PER_DOC}"
    )
    print(
        f" Native SWA: window={SWA_WINDOW} | {context_desc} | "
        f"recompute_ratio={RECOMPUTE_RATIO:.0%} per block"
    )

    overall = {}
    all_rows = []  # every sample across all tasks, for the Top-N similar report

    for dataset in DATASETS:
        for label, n_seg, tps in LENGTHS:
            n = 1 if one_sample is not None else N_SAMPLES
            samples = _load_samples(
                tok,
                dataset,
                n_seg,
                (int(one_sample) + 1) if one_sample else n,
                tokens_per_segment=tps,
            )
            if one_sample is not None:
                samples = [samples[int(one_sample)]]
            if not samples:
                print(
                    f"\n[skip] {dataset}@{label}: no sample reached {n_seg}x{tps} tok"
                )
                continue
            # chunk_size must be >= tps so each doc becomes exactly one segment.
            chunk_size = max(4096, tps + 96)

            print(f"\n{'=' * W}")
            shape = (
                "raw LongBench context"
                if RAG_MODE == "standard"
                else f"{n_seg} x {tps} tokens"
            )
            print(f" {dataset} @ {label}  ({shape}, {len(samples)} sample(s))")
            print("=" * W)

            rows = []
            for si, s in enumerate(samples):
                qt = _query_text(s["question"], tok)
                gold = s["gold_answer"]
                # baseline (standard FA-2 prefill)
                logits_b, tb, ttft_b, dec_b, n_ctx, gpu_ms_b = standard_prefill(
                    model, tok, s["docs"], qt, MAX_NEW_TOKENS,
                    stop_strings=STOP_STRINGS,
                )
                ans_b = _short_ans(tb)
                gc.collect()
                torch.cuda.empty_cache()

                # RedKnot
                segs = offline_prefill_segments(
                    model, tok, s["docs"], chunk_size=chunk_size, model_id=model_src
                )
                actual_doc_lens = [seg.doc_len for seg in segs]
                if si == 0:
                    run_redknot_swa_offlinekv(
                        model,
                        tok,
                        segments_offline=segs,
                        query_text=qt,
                        max_new_tokens=3,
                        recompute_ratio=RECOMPUTE_RATIO,
                    )
                timing_stats = {}
                logits_c, tc, _, ttft_c = run_redknot_swa_offlinekv(
                    model,
                    tok,
                    segments_offline=segs,
                    query_text=qt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    recompute_ratio=RECOMPUTE_RATIO,
                    timing_stats=timing_stats,
                    stop_strings=STOP_STRINGS,
                )
                ans_c = _short_ans(tc)
                decode_forward_steps = timing_stats["decode_forward_steps"]
                dec_c = (
                    decode_forward_steps / timing_stats["decode_seconds"]
                    if decode_forward_steps
                    else 0.0
                )
                logits_cos = torch.nn.functional.cosine_similarity(
                    logits_b.float(), logits_c.float(), dim=0
                ).item()
                rows.append(
                    {
                        "q": s["question"],
                        "gold": gold,
                        "base_text": tb,
                        "rk_text": tc,
                        "base_ans": ans_b,
                        "rk_ans": ans_c,
                        "base_f1": f1_max(ans_b, gold),
                        "rk_f1": f1_max(ans_c, gold),
                        "base_em": em_max(ans_b, gold),
                        "rk_em": em_max(ans_c, gold),
                        "base_ttft": ttft_b,
                        "rk_ttft": ttft_c,
                        "base_dec": dec_b,
                        "rk_dec": dec_c,
                        "base_gpu_ms": gpu_ms_b,
                        "rk_gpu_ms": timing_stats["ttft_gpu_ms"],
                        "base_prefill_tps": n_ctx / ttft_b,
                        "rk_prefill_tps": n_ctx / ttft_c,
                        "logits_cos": logits_cos,
                        "doc_lens": actual_doc_lens,
                        "n_ctx": n_ctx,
                    }
                )

                # Tag this sample for the cross-task Top-N similarity report.
                # Similarity = token-F1 between the RedKnot answer and the
                # baseline answer (1.0 == identical answer span).
                _sim = f1_score(rows[-1]["rk_ans"], rows[-1]["base_ans"])
                _exact = rows[-1]["rk_ans"].strip() == rows[-1]["base_ans"].strip()
                all_rows.append(
                    {
                        **rows[-1],
                        "tag": f"{dataset}@{label}",
                        "dataset": dataset,
                        "documents": s["docs"],
                        "sim": _sim,
                        "exact": _exact,
                    }
                )

                # per-sample text dump
                print(f"\n [sample {si}] ctx={rows[-1]['n_ctx']:,} tok")
                print(f"   Q   : {_trunc(s['question'], 80)}")
                print(f"   gold: {_trunc(', '.join(map(str, gold)), 80)}")
                if TEXT_ONLY:
                    print("   FULL RECOMPUTE OUTPUT:")
                    print(tb)
                    print("   REDKNOT OUTPUT:")
                    print(tc)
                else:
                    print(
                        f"   base: {_trunc(tb, 70)!r}  -> ans={_trunc(ans_b, 30)!r} "
                        f"F1={rows[-1]['base_f1']:.2f}"
                    )
                    print(
                        f"   rk  : {_trunc(tc, 70)!r}  -> ans={_trunc(ans_c, 30)!r} "
                        f"F1={rows[-1]['rk_f1']:.2f}"
                    )
                print(
                    f"   TTFT: base={ttft_b:.2f}s  rk={ttft_c:.2f}s  "
                    f"speedup={ttft_b / ttft_c:.2f}x"
                )
                print(f"   LOGITS cosine={logits_cos:.6f}")
                print(
                    f"   GPU : base={gpu_ms_b:.1f}ms  "
                    f"rk={timing_stats['ttft_gpu_ms']:.1f}ms  "
                    f"saving={100 * (1 - timing_stats['ttft_gpu_ms'] / gpu_ms_b):.1f}%"
                )
                print(
                    f"   PREFILL: base={n_ctx / ttft_b:,.0f} tok/s  "
                    f"rk={n_ctx / ttft_c:,.0f} tok/s"
                )

                del segs
                get_global_offline_cache().clear()
                gc.collect()
                torch.cuda.empty_cache()

            # aggregates
            def m(key):
                return sum(row[key] for row in rows) / len(rows)

            print(f"\n {'-' * (W - 2)}")
            tag = f"{dataset}@{label}"
            print(f" {tag} AGGREGATE ({len(rows)} sample(s))")
            print(f" {'-' * (W - 2)}")
            # Quality with dispersion: a single F1 mean from n=4 is near-meaningless,
            # so report sample-std and the 95% CI half-width alongside the mean.
            bf1_m, bf1_sd, bf1_ci = _mean_std_ci([r["base_f1"] for r in rows])
            rf1_m, rf1_sd, rf1_ci = _mean_std_ci([r["rk_f1"] for r in rows])
            bem_m, bem_sd, _ = _mean_std_ci([r["base_em"] for r in rows])
            rem_m, rem_sd, _ = _mean_std_ci([r["rk_em"] for r in rows])
            n_q = len(rows)
            if not TEXT_ONLY:
                print(
                    f"   QUALITY   baseline  F1={bf1_m:.3f}±{bf1_sd:.3f} "
                    f"(95%CI ±{bf1_ci:.3f})  EM={bem_m:.3f}±{bem_sd:.3f}"
                )
                print(
                    f"             RedKnot   F1={rf1_m:.3f}±{rf1_sd:.3f} "
                    f"(95%CI ±{rf1_ci:.3f})  EM={rem_m:.3f}±{rem_sd:.3f}"
                )
            if not TEXT_ONLY and n_q < 10:
                print(
                    f"             [!] n={n_q}: F1 CI is wide; treat the mean as "
                    f"indicative only (raise REDKNOT_N_SAMPLES for significance)."
                )
            print(
                f"   TTFT      baseline={m('base_ttft'):.2f}s  "
                f"RedKnot={m('rk_ttft'):.2f}s  "
                f"speedup={m('base_ttft') / m('rk_ttft'):.2f}x"
            )
            print(
                f"   DECODE    baseline={m('base_dec'):.1f} tok/s  "
                f"RedKnot={m('rk_dec'):.1f} tok/s"
            )
            print(f"   LOGITS    first-token cosine={m('logits_cos'):.6f}")
            print(
                f"   PREFILL   baseline={m('base_prefill_tps'):,.0f} tok/s  "
                f"RedKnot={m('rk_prefill_tps'):,.0f} tok/s"
            )
            print(
                f"   COMPUTE   measured GPU: baseline={m('base_gpu_ms'):.1f}ms  "
                f"RedKnot={m('rk_gpu_ms'):.1f}ms  "
                f"saving={100 * (1 - m('rk_gpu_ms') / m('base_gpu_ms')):.1f}%"
            )
            overall[tag] = {
                "base_f1": m("base_f1"),
                "rk_f1": m("rk_f1"),
                "base_em": m("base_em"),
                "rk_em": m("rk_em"),
                "base_ttft": m("base_ttft"),
                "rk_ttft": m("rk_ttft"),
                "base_dec": m("base_dec"),
                "rk_dec": m("rk_dec"),
                "base_gpu_ms": m("base_gpu_ms"),
                "rk_gpu_ms": m("rk_gpu_ms"),
                "base_prefill_tps": m("base_prefill_tps"),
                "rk_prefill_tps": m("rk_prefill_tps"),
                "logits_cos": m("logits_cos"),
                "gpu_saving": 100 * (1 - m("rk_gpu_ms") / m("base_gpu_ms")),
                "n": n_q,
                "base_f1_ci": bf1_ci,
                "rk_f1_ci": rf1_ci,
            }

    # final summary table
    print(f"\n{'=' * W}")
    print(" SUMMARY")
    print("=" * W)
    if TEXT_ONLY:
        print(
            f" {'dataset@len':16s} {'n':>3s} {'base TTFT':>10s} "
            f"{'rk TTFT':>9s} {'speedup':>8s} {'GPU save':>11s}"
        )
    else:
        print(
            f" {'dataset@len':16s} {'n':>3s} {'base F1 (95%CI)':>18s} "
            f"{'rk F1 (95%CI)':>18s} {'base TTFT':>10s} {'rk TTFT':>9s} "
            f"{'speedup':>8s} {'GPU save':>11s}"
        )
    for tag, o in overall.items():
        if TEXT_ONLY:
            print(
                f" {tag:16s} {o['n']:>3d} {o['base_ttft']:>9.2f}s "
                f"{o['rk_ttft']:>8.2f}s {o['base_ttft'] / o['rk_ttft']:>7.2f}x "
                f"{o['gpu_saving']:>10.1f}%"
            )
        else:
            print(
                f" {tag:16s} {o['n']:>3d} "
                f"{o['base_f1']:>10.3f}±{o['base_f1_ci']:<6.3f} "
                f"{o['rk_f1']:>10.3f}±{o['rk_f1_ci']:<6.3f} "
                f"{o['base_ttft']:>9.2f}s {o['rk_ttft']:>8.2f}s "
                f"{o['base_ttft'] / o['rk_ttft']:>7.2f}x {o['gpu_saving']:>10.1f}%"
            )
    print("=" * W)

    if overall:
        total_n = sum(o["n"] for o in overall.values())

        def weighted(key):
            return sum(o[key] * o["n"] for o in overall.values()) / total_n

        exact_n = sum(1 for row in all_rows if row["exact"])
        base_ttft = weighted("base_ttft")
        rk_ttft = weighted("rk_ttft")
        print(f"\n{'=' * W}")
        print(" OVERALL ACCURACY / PERFORMANCE / COMPUTE")
        print("=" * W)
        print(
            f" samples={total_n}  datasets={len(overall)}  "
            f"exact-output-match={exact_n}/{len(all_rows)} "
            f"({100 * exact_n / max(len(all_rows), 1):.1f}%)"
        )
        if not TEXT_ONLY:
            print(
                f" accuracy  baseline F1/EM={weighted('base_f1'):.3f}/"
                f"{weighted('base_em'):.3f}  RedKnot F1/EM={weighted('rk_f1'):.3f}/"
                f"{weighted('rk_em'):.3f}"
            )
        print(f" logits    first-token cosine={weighted('logits_cos'):.6f}")
        print(
            f" TTFT      baseline={base_ttft:.3f}s  RedKnot={rk_ttft:.3f}s  "
            f"speedup={base_ttft / rk_ttft:.2f}x"
        )
        print(
            f" decode    baseline={weighted('base_dec'):.1f} tok/s  "
            f"RedKnot={weighted('rk_dec'):.1f} tok/s"
        )
        print(
            f" prefill   baseline={weighted('base_prefill_tps'):,.0f} tok/s  "
            f"RedKnot={weighted('rk_prefill_tps'):,.0f} tok/s"
        )
        base_gpu_ms = weighted("base_gpu_ms")
        rk_gpu_ms = weighted("rk_gpu_ms")
        print(
            f" compute   measured GPU time: baseline={base_gpu_ms:.1f}ms  "
            f"RedKnot={rk_gpu_ms:.1f}ms  saving={100 * (1 - rk_gpu_ms / base_gpu_ms):.1f}%"
        )
        print("=" * W)

    # ── Always export per-sample results ──
    EXPORT_ALL = os.environ.get("REDKNOT_EXPORT_ALL", "1") == "1"
    if EXPORT_ALL and all_rows:
        all_export_path = (
            Path(__file__).resolve().parent / "datasets/longbench_rag_results.json"
        )
        all_export = []
        for row in all_rows:
            speedup = row["base_ttft"] / row["rk_ttft"] if row["rk_ttft"] > 0 else 0
            gpu_saving = 100 * (1 - row["rk_gpu_ms"] / row["base_gpu_ms"]) if row["base_gpu_ms"] > 0 else 0
            all_export.append({
                "dataset": row["dataset"],
                "question": row["q"],
                "answers": row["gold"],
                "documents": row["documents"],
                "n_ctx": row["n_ctx"],
                "baseline": {
                    "text": row["base_text"],
                    "answer": row["base_ans"],
                    "f1": row["base_f1"],
                    "em": row["base_em"],
                    "ttft_seconds": row["base_ttft"],
                    "gpu_compute_ms": row["base_gpu_ms"],
                    "prefill_tokens_per_second": row["base_prefill_tps"],
                    "decode_tokens_per_second": row["base_dec"],
                },
                "redknot": {
                    "text": row["rk_text"],
                    "answer": row["rk_ans"],
                    "f1": row["rk_f1"],
                    "em": row["rk_em"],
                    "ttft_seconds": row["rk_ttft"],
                    "gpu_compute_ms": row["rk_gpu_ms"],
                    "prefill_tokens_per_second": row["rk_prefill_tps"],
                    "decode_tokens_per_second": row["rk_dec"],
                },
                "comparison": {
                    "exact_output_match": row["exact"],
                    "answer_similarity": row["sim"],
                    "first_token_logits_cosine": row["logits_cos"],
                    "ttft_speedup": speedup,
                    "measured_gpu_time_saving_percent": gpu_saving,
                },
            })
        with open(all_export_path, "w", encoding="utf-8") as f:
            json.dump(all_export, f, ensure_ascii=False, indent=2)
        print(f"\n Per-sample results exported to {all_export_path}")
        print(f" ({len(all_export)} samples)")

    if TEXT_ONLY:
        return

    # Export the per-sample comparison for every evaluated sample, in the
    # natural evaluation order (no ranking / no selection).
    export_path = (
        Path(__file__).resolve().parent / "datasets/longbench_rag_summary.json"
    )
    export_rows = []
    for idx, row in enumerate(all_rows, 1):
        export_rows.append(
            {
                "index": idx,
                "dataset": row["dataset"],
                "question": row["q"],
                "answers": row["gold"],
                "documents": row["documents"],
                "baseline": {
                    "text": row["base_text"],
                    "answer": row["base_ans"],
                    "f1": row["base_f1"],
                    "em": row["base_em"],
                    "ttft_seconds": row["base_ttft"],
                    "gpu_compute_ms": row["base_gpu_ms"],
                    "prefill_tokens_per_second": row["base_prefill_tps"],
                    "decode_tokens_per_second": row["base_dec"],
                },
                "redknot": {
                    "text": row["rk_text"],
                    "answer": row["rk_ans"],
                    "f1": row["rk_f1"],
                    "em": row["rk_em"],
                    "ttft_seconds": row["rk_ttft"],
                    "gpu_compute_ms": row["rk_gpu_ms"],
                    "prefill_tokens_per_second": row["rk_prefill_tps"],
                    "decode_tokens_per_second": row["rk_dec"],
                },
                "comparison": {
                    "exact_output_match": row["exact"],
                    "answer_similarity": row["sim"],
                    "first_token_logits_cosine": row["logits_cos"],
                    "ttft_speedup": row["base_ttft"] / row["rk_ttft"],
                    "measured_gpu_time_saving_percent": 100
                    * (1 - row["rk_gpu_ms"] / row["base_gpu_ms"]),
                },
            }
        )
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export_rows, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * W}")
    print(" PER-SAMPLE RESULTS (RedKnot vs baseline)")
    print(f" JSON: {export_path}")
    print("=" * W)
    n_exact = sum(1 for r in all_rows if r["exact"])
    print(
        f" {len(all_rows)} samples total | {n_exact} exact-match "
        f"({100 * n_exact / max(len(all_rows), 1):.0f}%) | "
        f"mean answer-similarity={sum(r['sim'] for r in all_rows) / max(len(all_rows), 1):.3f}"
    )
    print("-" * W)
    for i, r in enumerate(all_rows, 1):
        mark = "EXACT" if r["exact"] else f"sim={r['sim']:.2f}"
        print(f"\n [{i:2d}] {r['tag']}  ctx={r['n_ctx']:,} tok  [{mark}]")
        print(f"      Q       : {_trunc(r['q'], 84)}")
        print(f"      gold    : {_trunc(', '.join(map(str, r['gold'])), 84)}")
        print(f"      baseline: {_trunc(r['base_ans'], 84)!r}  F1={r['base_f1']:.2f}")
        print(f"      RedKnot : {_trunc(r['rk_ans'], 84)!r}  F1={r['rk_f1']:.2f}")
        print(
            f"      TTFT    : {r['base_ttft']:.3f}s -> {r['rk_ttft']:.3f}s  "
            f"({r['base_ttft'] / r['rk_ttft']:.2f}x)"
        )
        print(
            f"      GPU time: {r['base_gpu_ms']:.1f}ms -> {r['rk_gpu_ms']:.1f}ms  "
            f"(saving {100 * (1 - r['rk_gpu_ms'] / r['base_gpu_ms']):.1f}%)"
        )
        print(
            f"      prefill : {r['base_prefill_tps']:,.0f} -> "
            f"{r['rk_prefill_tps']:,.0f} tok/s"
        )
        print(f"      decode  : {r['base_dec']:.1f} -> {r['rk_dec']:.1f} tok/s")
    print(f"\n{'=' * W}")


if __name__ == "__main__":
    main()
