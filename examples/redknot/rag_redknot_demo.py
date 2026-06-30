#!/usr/bin/env python3
"""RAG accuracy / speed / compute demo for the SGLang RedKnot integration.

The demo compares a dense chunked-prefill baseline against RedKnot's
position-independent segment KV reuse on a HotpotQA-style RAG prompt.

Default HuggingFace assets:
  model   : Qwen/Qwen3-32B
  dataset : hotpotqa/hotpot_qa, config=distractor, split=validation

For faster local smoke tests, pass a smaller local or HuggingFace model path and
shorter segments, for example ``--model-path Qwen/Qwen3-0.6B --tokens-per-segment
512``. Large defaults are chosen to match the paper-style RAG setting, not to be
quick on a single GPU.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import string
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from sglang.srt.layers.attention.redknot import (  # noqa: E402
    DEFAULT_SINK_SIZE,
    HeadClassConfig,
    SparseFFNSchedule,
    offline_prefill_segments,
    run_baseline,
    run_redknot,
    sparse_ffn_flops,
)


DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_DATASET = "hotpotqa/hotpot_qa"
DEFAULT_DATASET_CONFIG = "distractor"
DEFAULT_SPLIT = "validation"


@dataclass
class SampleResult:
    sample_id: str
    question: str
    gold: str
    baseline_text: str
    redknot_text: str
    baseline_pred: str
    redknot_pred: str
    baseline_f1: float
    redknot_f1: float
    baseline_em: float
    redknot_em: float
    logits_cosine: float
    top1_match: bool
    top10_overlap: float
    baseline_ttft_s: float
    redknot_online_ttft_s: float
    wall_speedup: float
    offline_prefill_s: float
    baseline_flops: float
    redknot_online_flops: float
    flops_speedup: float
    flops_savings: float
    doc_lens: list[int]
    query_len: int


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def extract_short_answer(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    first_line = re.sub(r"^(short answer|answer)\s*:\s*", "", first_line, flags=re.I)
    return first_line.strip().strip('"')


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).item()
    )


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 10) -> float:
    a_ids = set(torch.topk(a, k).indices.detach().cpu().tolist())
    b_ids = set(torch.topk(b, k).indices.detach().cpu().tolist())
    return len(a_ids & b_ids) / k


def flatten_context(row: dict[str, Any]) -> list[str]:
    context = row.get("context", [])
    paragraphs: list[str] = []

    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])
        for title, sentences in zip(titles, sentences_list):
            paragraphs.append(f"{title}. {' '.join(map(str, sentences))}")
        return paragraphs

    for item in context:
        if isinstance(item, dict):
            title = item.get("title", "")
            sentences = item.get("sentences", item.get("sentence", []))
            paragraphs.append(f"{title}. {' '.join(map(str, sentences))}")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            paragraphs.append(f"{item[0]}. {' '.join(map(str, item[1]))}")
        else:
            paragraphs.append(str(item))
    return paragraphs


def supporting_titles(row: dict[str, Any]) -> set[str]:
    facts = row.get("supporting_facts", [])
    if isinstance(facts, dict):
        return {str(t) for t in facts.get("title", [])}
    titles = set()
    for fact in facts:
        if isinstance(fact, dict) and "title" in fact:
            titles.add(str(fact["title"]))
        elif isinstance(fact, (list, tuple)) and fact:
            titles.add(str(fact[0]))
    return titles


def build_segment(
    text_parts: Sequence[str],
    tokenizer,
    target_tokens: int,
    filler_paragraphs: Sequence[str],
) -> str:
    ids = tokenizer("\n\n".join(text_parts), add_special_tokens=False)["input_ids"]
    filler_ids = [
        tokenizer("\n\n" + p, add_special_tokens=False)["input_ids"]
        for p in filler_paragraphs
        if p.strip()
    ] or [tokenizer(" Continued.", add_special_tokens=False)["input_ids"]]
    idx = 0
    while len(ids) < target_tokens:
        ids.extend(filler_ids[idx % len(filler_ids)])
        idx += 1
    return tokenizer.decode(ids[:target_tokens], skip_special_tokens=True)


def load_rag_samples(args, tokenizer) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Install `datasets` to load the HuggingFace RAG dataset"
        ) from exc

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)

    samples: list[dict[str, Any]] = []
    for idx in indices:
        if len(samples) >= args.n_samples:
            break
        row = dict(dataset[idx])
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        sample_id = str(row.get("id", idx))
        paragraphs = flatten_context(row)
        if not question or not answer or not paragraphs:
            continue

        gold_titles = supporting_titles(row)
        gold = [p for p in paragraphs if p.split(".", 1)[0].strip() in gold_titles]
        distractors = [p for p in paragraphs if p not in gold]
        if not gold:
            gold = paragraphs[:1]
            distractors = paragraphs[1:] or paragraphs

        docs = [
            build_segment(
                list(gold) + distractors[:2],
                tokenizer,
                args.tokens_per_segment,
                distractors or paragraphs,
            )
        ]
        filler = distractors or paragraphs
        for seg_idx in range(1, args.n_segments):
            start = (seg_idx - 1) % len(filler)
            docs.append(
                build_segment(
                    [filler[start]], tokenizer, args.tokens_per_segment, filler
                )
            )

        samples.append(
            {
                "id": sample_id,
                "question": question,
                "answer": answer,
                "docs": docs,
                "doc_token_lens": [
                    len(tokenizer(d, add_special_tokens=False)["input_ids"])
                    for d in docs
                ],
            }
        )
    return samples


def build_query_text(question: str) -> str:
    return (
        "\n\nAnswer the question using only the documents above. "
        "Give the shortest exact answer span. Do not explain.\n"
        f"Question: {question}\nShort answer:"
    )


def make_head_config(model_config, args) -> HeadClassConfig:
    if args.head_config:
        return HeadClassConfig.from_json(args.head_config)

    num_layers = int(model_config.num_hidden_layers)
    num_kv_heads = int(model_config.num_key_value_heads)
    dense_prefix = min(args.dense_prefix_layers, num_layers)
    global_heads = max(1, int(math.ceil(num_kv_heads * args.global_head_ratio)))

    head_class: list[list[str]] = []
    head_max_distance: list[list[int]] = []
    head_sink_size: list[list[int]] = []
    for layer_idx in range(num_layers):
        cls_row: list[str] = []
        win_row: list[int] = []
        sink_row: list[int] = []
        for head_idx in range(num_kv_heads):
            if layer_idx < dense_prefix:
                cls_row.append("dense")
                win_row.append(-1)
            elif head_idx < global_heads:
                cls_row.append("global")
                win_row.append(-1)
            else:
                cls_row.append("local")
                win_row.append(args.local_window)
            sink_row.append(args.sink_size)
        head_class.append(cls_row)
        head_max_distance.append(win_row)
        head_sink_size.append(sink_row)

    return HeadClassConfig(
        head_class=head_class,
        head_max_distance=head_max_distance,
        head_sink_size=head_sink_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        dense_prefix_layers=0,
    )


def model_dims(config) -> dict[str, int]:
    q_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    hidden = int(config.hidden_size)
    head_dim = int(getattr(config, "head_dim", hidden // q_heads))
    layers = int(config.num_hidden_layers)
    intermediate = int(getattr(config, "intermediate_size", hidden * 4))
    return {
        "layers": layers,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "hidden": hidden,
        "intermediate": intermediate,
        "q_per_kv": q_heads // kv_heads,
    }


def ffn_flops(tokens: int, dims: dict[str, int]) -> float:
    # SwiGLU-style MLP: gate/up/down projections. Count multiply-add as 2 FLOPs.
    return 6.0 * tokens * dims["hidden"] * dims["intermediate"]


def dense_attention_flops(q_tokens: int, kv_tokens: int, dims: dict[str, int]) -> float:
    # QK + AV, all query heads.
    return 4.0 * q_tokens * kv_tokens * dims["q_heads"] * dims["head_dim"]


def estimate_baseline_flops(total_tokens: int, config) -> float:
    dims = model_dims(config)
    # Causal prefill averages half the KV length.
    attn = dense_attention_flops(total_tokens, max(1, total_tokens // 2), dims)
    ffn = ffn_flops(total_tokens, dims)
    return dims["layers"] * (attn + ffn)


def estimate_redknot_online_flops(
    doc_lens: Sequence[int], query_len: int, head_cfg: HeadClassConfig, config
) -> float:
    dims = model_dims(config)
    total = 0.0
    q_per_kv = dims["q_per_kv"]

    def layer_attention(tokens: int, prev_tokens: int, layer_idx: int) -> float:
        acc = 0.0
        for kv_head in range(dims["kv_heads"]):
            strat = head_cfg.get_strategy(layer_idx, kv_head)
            q_heads = q_per_kv
            if strat.is_local():
                visible_prev = min(prev_tokens, max(strat.window, 0) + strat.sink_size)
            else:
                visible_prev = prev_tokens
            # Previous KV is fully visible; self region is causal, average half length.
            kv_tokens = visible_prev + max(1, tokens // 2)
            acc += 4.0 * tokens * kv_tokens * q_heads * dims["head_dim"]
        return acc

    prev = doc_lens[0] if doc_lens else 0
    for doc_len in doc_lens[1:]:
        for layer_idx in range(dims["layers"]):
            total += layer_attention(doc_len, prev, layer_idx)
            total += ffn_flops(doc_len, dims)
        prev += doc_len

    # Query always attends over all document KV plus its own causal suffix.
    for layer_idx in range(dims["layers"]):
        total += dense_attention_flops(query_len, prev + max(1, query_len // 2), dims)
        total += ffn_flops(query_len, dims)
    return total


def fmt_flops(value: float) -> str:
    for unit, scale in (("PFLOPs", 1e15), ("TFLOPs", 1e12), ("GFLOPs", 1e9)):
        if value >= scale:
            return f"{value / scale:.2f} {unit}"
    return f"{value:.2f} FLOPs"


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def run_demo(args) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"Loading dataset: {args.dataset} ({args.dataset_config}) split={args.split}")
    samples = load_rag_samples(args, tokenizer)
    if not samples:
        raise RuntimeError("No usable RAG samples were built from the dataset")
    log(f"Built {len(samples)} RAG samples")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    log(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    ).eval()

    head_cfg = make_head_config(model.config, args)
    log(f"Head summary: {head_cfg.summary()}")

    sparse_ffn_schedule = None
    if args.sparse_ffn:
        sparse_ffn_schedule = SparseFFNSchedule(
            dense_until=args.ffn_dense_until,
            mass_thresh=args.ffn_mass_thresh,
            recent_n=args.ffn_recent_n,
        )
        log(
            f"Sparse FFN enabled: dense_until={sparse_ffn_schedule.dense_until} "
            f"mass_thresh={sparse_ffn_schedule.mass_thresh} "
            f"recent_n={sparse_ffn_schedule.recent_n}"
        )

    results: list[SampleResult] = []
    for sample_idx, sample in enumerate(samples, 1):
        log("=" * 80)
        log(f"Sample {sample_idx}/{len(samples)} id={sample['id']}")
        log(f"Q: {sample['question']}")
        log(f"Gold: {sample['answer']}")
        query_text = build_query_text(sample["question"])
        prompt = "\n\n".join(sample["docs"]) + query_text

        log("Running dense baseline...")
        bl_logits, bl_text, bl_prompt_len, bl_ttft = run_baseline(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            chunk_size=args.baseline_chunk_size,
            attn_impl=args.attn_implementation,
        )
        bl_pred = extract_short_answer(bl_text)

        log("Running RedKnot offline prefill per document segment...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        offline_t0 = time.perf_counter()
        segments = offline_prefill_segments(
            model,
            tokenizer,
            sample["docs"],
            chunk_size=args.offline_chunk_size,
            model_id=args.model_path,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        offline_s = time.perf_counter() - offline_t0

        log("Running RedKnot online reuse path...")
        ffn_stats: list[dict] = []
        rc_logits, rc_text, query_len, rc_ttft = run_redknot(
            model,
            tokenizer,
            segments_offline=segments,
            query_text=query_text,
            head_cfg=head_cfg,
            max_new_tokens=args.max_new_tokens,
            q_chunk_size=args.redknot_q_chunk_size,
            kernel=args.kernel,
            sparse_ffn_schedule=sparse_ffn_schedule,
            sparse_ffn_stats=ffn_stats if sparse_ffn_schedule else None,
        )
        rc_pred = extract_short_answer(rc_text)

        # ── Sparse FFN savings (deep-layer selected fraction) ──
        ffn_report = None
        if sparse_ffn_schedule and ffn_stats:
            deep = [s for s in ffn_stats if s.get("mode") == "sparse"]
            sel_frac = (
                sum(s["selected_frac"] for s in deep) / len(deep) if deep else 1.0
            )
            cfg = model.config
            hidden = int(cfg.hidden_size)
            inter = int(getattr(cfg, "intermediate_size", hidden * 4))
            ffn_report = sparse_ffn_flops(
                num_layers=int(cfg.num_hidden_layers),
                tokens_per_layer=sum(doc_lens[1:]) if len(doc_lens) > 1 else 1,
                hidden=hidden,
                intermediate=inter,
                schedule=sparse_ffn_schedule,
                selected_frac_deep=sel_frac,
            )
            log(
                f"  Sparse FFN: deep_selected_frac={sel_frac:.3f} "
                f"ffn_savings={ffn_report['ffn_flops_savings'] * 100:.1f}% "
                f"ffn_speedup={ffn_report['ffn_speedup']:.2f}x"
            )

        doc_lens = [seg.doc_len for seg in segments]
        bl_flops = estimate_baseline_flops(sum(doc_lens) + query_len, model.config)
        rc_flops = estimate_redknot_online_flops(
            doc_lens, query_len, head_cfg, model.config
        )
        wall_speedup = bl_ttft / rc_ttft if rc_ttft > 0 else float("inf")
        flops_speedup = bl_flops / rc_flops if rc_flops > 0 else float("inf")
        flops_savings = 1.0 - rc_flops / bl_flops if bl_flops > 0 else 0.0

        result = SampleResult(
            sample_id=sample["id"],
            question=sample["question"],
            gold=sample["answer"],
            baseline_text=bl_text,
            redknot_text=rc_text,
            baseline_pred=bl_pred,
            redknot_pred=rc_pred,
            baseline_f1=f1_score(bl_pred, sample["answer"]),
            redknot_f1=f1_score(rc_pred, sample["answer"]),
            baseline_em=exact_match_score(bl_pred, sample["answer"]),
            redknot_em=exact_match_score(rc_pred, sample["answer"]),
            logits_cosine=cosine_sim(bl_logits, rc_logits),
            top1_match=bool(bl_logits.argmax().item() == rc_logits.argmax().item()),
            top10_overlap=topk_overlap(bl_logits, rc_logits),
            baseline_ttft_s=bl_ttft,
            redknot_online_ttft_s=rc_ttft,
            wall_speedup=wall_speedup,
            offline_prefill_s=offline_s,
            baseline_flops=bl_flops,
            redknot_online_flops=rc_flops,
            flops_speedup=flops_speedup,
            flops_savings=flops_savings,
            doc_lens=doc_lens,
            query_len=query_len,
        )
        results.append(result)

        print("\nSample comparison")
        print(f"  baseline_pred={result.baseline_pred!r}")
        print(f"  redknot_pred={result.redknot_pred!r}")
        print(f"  gold={result.gold!r}")
        print(
            f"  F1 baseline={result.baseline_f1:.3f} redknot={result.redknot_f1:.3f} | "
            f"EM baseline={result.baseline_em:.0f} redknot={result.redknot_em:.0f}"
        )
        print(
            f"  Top1={result.top1_match} Top10={result.top10_overlap:.0%} "
            f"Cosine={result.logits_cosine:.4f}"
        )
        print(
            f"  TTFT baseline={result.baseline_ttft_s:.2f}s "
            f"redknot_online={result.redknot_online_ttft_s:.2f}s "
            f"speedup={result.wall_speedup:.2f}x"
        )
        print(
            f"  FLOPs baseline={fmt_flops(result.baseline_flops)} "
            f"redknot_online={fmt_flops(result.redknot_online_flops)} "
            f"speedup={result.flops_speedup:.2f}x savings={result.flops_savings * 100:.1f}%"
        )

        del segments
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "n": len(results),
        "baseline_f1": mean(r.baseline_f1 for r in results),
        "redknot_f1": mean(r.redknot_f1 for r in results),
        "baseline_em": mean(r.baseline_em for r in results),
        "redknot_em": mean(r.redknot_em for r in results),
        "logits_cosine": mean(r.logits_cosine for r in results),
        "top1_match_rate": mean(float(r.top1_match) for r in results),
        "top10_overlap": mean(r.top10_overlap for r in results),
        "baseline_ttft_s": mean(r.baseline_ttft_s for r in results),
        "redknot_online_ttft_s": mean(r.redknot_online_ttft_s for r in results),
        "wall_speedup": mean(r.wall_speedup for r in results),
        "offline_prefill_s": mean(r.offline_prefill_s for r in results),
        "baseline_flops": mean(r.baseline_flops for r in results),
        "redknot_online_flops": mean(r.redknot_online_flops for r in results),
        "flops_speedup": mean(r.flops_speedup for r in results),
        "flops_savings": mean(r.flops_savings for r in results),
    }

    print("\n" + "=" * 80)
    print("Aggregate RAG demo summary")
    print(
        f"  Accuracy F1: baseline={summary['baseline_f1']:.3f} redknot={summary['redknot_f1']:.3f}"
    )
    print(
        f"  Accuracy EM: baseline={summary['baseline_em']:.3f} redknot={summary['redknot_em']:.3f}"
    )
    print(
        f"  Logits: cosine={summary['logits_cosine']:.4f} "
        f"top1={summary['top1_match_rate']:.0%} top10={summary['top10_overlap']:.0%}"
    )
    print(
        f"  TTFT: baseline={summary['baseline_ttft_s']:.2f}s "
        f"redknot_online={summary['redknot_online_ttft_s']:.2f}s "
        f"speedup={summary['wall_speedup']:.2f}x"
    )
    print(
        f"  FLOPs: baseline={fmt_flops(summary['baseline_flops'])} "
        f"redknot_online={fmt_flops(summary['redknot_online_flops'])} "
        f"speedup={summary['flops_speedup']:.2f}x "
        f"savings={summary['flops_savings'] * 100:.1f}%"
    )
    print("=" * 80)

    return {
        "model": args.model_path,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "head_config": args.head_config or "generated",
        "head_summary": head_cfg.summary(),
        "n_segments": args.n_segments,
        "tokens_per_segment": args.tokens_per_segment,
        "summary": summary,
        "results": [asdict(r) for r in results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--head-config", default=None, help="Optional RedKnot head config JSON"
    )
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--n-segments", type=int, default=6)
    parser.add_argument("--tokens-per-segment", type=int, default=5000)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        choices=["sdpa", "flash_attention_2", "eager"],
        default="sdpa",
    )
    parser.add_argument("--kernel", choices=["fa2", "fa3"], default="fa2")
    parser.add_argument("--baseline-chunk-size", type=int, default=8192)
    parser.add_argument("--offline-chunk-size", type=int, default=4096)
    parser.add_argument("--redknot-q-chunk-size", type=int, default=2048)
    parser.add_argument("--global-head-ratio", type=float, default=0.15)
    parser.add_argument("--local-window", type=int, default=256)
    parser.add_argument("--sink-size", type=int, default=DEFAULT_SINK_SIZE)
    parser.add_argument("--dense-prefix-layers", type=int, default=0)
    # RedKnot Sparse FFN (paper §4.2, Table 4 footnote defaults).
    parser.add_argument(
        "--sparse-ffn",
        action="store_true",
        help="Enable RedKnot token-selective Sparse FFN recovery.",
    )
    parser.add_argument("--ffn-dense-until", type=int, default=20)
    parser.add_argument("--ffn-mass-thresh", type=float, default=0.5)
    parser.add_argument("--ffn-recent-n", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/redknot_rag_demo.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_demo(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
