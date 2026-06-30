#!/usr/bin/env python3
"""benchmark_RedKnot_QPS.py — concurrency/QPS comparison: baseline vs RedKnot.

Methodology (per the agreed design):
  * Each context length maps to a request count ("concurrency"):
        16K -> 4 requests, 32K -> 8 requests, 64K -> 16 requests.
  * We run ALL requests for a length back-to-back on one model (HF device_map is
    a single CUDA stream, so concurrent submission would just serialize on the
    GPU anyway), measure total wall-clock, and report:
        - total time for the batch of requests
        - QPS = requests / total_time
        - avg per-request latency
  * Baseline = standard full prefill + decode.
    RedKnot   = RedKnot sparse prefill + decode (same per-request path the
    accuracy benchmark uses).

Run (Qwen3.5):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
    REDKNOT_MAX_NEW=32 CUDA_VISIBLE_DEVICES=0,1 \
    PYTHONPATH=python:.venv_tf5/lib/python3.11/site-packages:/root/miniconda3/lib/python3.11/site-packages \
    .venv_tf5/bin/python test/srt/redknot/benchmark_RedKnot_QPS.py --model qwen35

Run (Qwen3):
  CUDA_VISIBLE_DEVICES=0 python test/srt/redknot/benchmark_RedKnot_QPS.py --model qwen3

Run (Llama3.3):
  CUDA_VISIBLE_DEVICES=0 python test/srt/redknot/benchmark_RedKnot_QPS.py --model llama33
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))
_HERE = Path(__file__).resolve().parent

# length(tokens) -> (n_chunk, chunk_tokens, concurrency/request-count)
LENGTH_PLAN = {
    16000: (2, 8000, 4),
    32000: (4, 8000, 8),
    64000: (8, 8000, 16),
}
DATASET = os.environ.get("REDKNOT_QPS_DATASET", "triviaqa")
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "24"))


def _summ(name, n_req, total_s):
    qps = n_req / max(total_s, 1e-6)
    lat = total_s / max(n_req, 1)
    return f"{name:8s} reqs={n_req:2d}  total={total_s:7.2f}s  QPS={qps:6.3f}  avg_latency={lat:6.2f}s"


@torch.no_grad()
def run_qwen35(lengths):
    import benchmark_RedKnot_QWen35_RAG as b
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        _install_full_patches,
        build_full_attention_head_config,
        collect_attention_mass,
        install_linear_segmented,
        install_moe_token_sparse,
    )
    from transformers import DynamicCache

    tok = AutoTokenizer.from_pretrained(b.MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        b.MODEL, dtype=torch.bfloat16, device_map=b.DEVICE_MAP, trust_remote_code=True
    ).eval()

    head_cfg = build_full_attention_head_config(
        model.config, frac_global=b.FRAC_GLOBAL, local_window=b.FULL_WINDOW
    )

    def load(name, n, n_chunk, chunk, target, seed=0):
        import json
        import random

        raw = []
        with open(os.path.join(b.LB, f"{name}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r.get("input") and r.get("context") and r.get("answers"):
                    raw.append(r)
        random.Random(seed).shuffle(raw)
        out, nraw = [], len(raw)
        for i in range(nraw):
            if len(out) >= n:
                break
            toks = tok(raw[i]["context"], add_special_tokens=False)["input_ids"]
            j = (i + 1) % nraw
            while len(toks) < target and j != i:
                toks += tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
                j = (j + 1) % nraw
            toks = toks[:target]
            if len(toks) < target:
                continue
            chunks = [
                tok.decode(toks[k : k + chunk], skip_special_tokens=True)
                for k in range(0, target, chunk)
            ]
            out.append({"q": raw[i]["input"], "chunks": chunks})
        # pad by cycling if dataset too small to reach n
        if out:
            while len(out) < n:
                out.append(out[len(out) % max(1, len(out) - 1)])
        return out[:n]

    @torch.no_grad()
    def baseline_request(sample):
        qt = b.QP.format(q=sample["q"])
        full_text = "\n\n".join(sample["chunks"]) + qt
        b.standard(model, tok, full_text)

    @torch.no_grad()
    def redknot_request(sample, n_chunk, chunk):
        device = model.device
        chunks, qt = sample["chunks"], b.QP.format(q=sample["q"])
        text = "\n\n".join(chunks) + qt
        ids0 = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            device
        )
        mass = collect_attention_mass(model, ids0, deep_full_frac=0.5)
        win, _, _ = b.build_win(model, tok, text, n_chunk, chunk)
        rf = _install_full_patches(
            model, head_cfg, dense_prefix_full_layers=b.DENSE_FULL_LAYERS
        )
        rl = install_linear_segmented(model, win, seg=b.LINEAR_SEG)
        rm = install_moe_token_sparse(
            model,
            mass,
            deep_moe_start_layer=b.DEEP_MOE_START,
            mass_thresh=b.MOE_MASS_THRESH,
        )
        try:
            cache = DynamicCache(config=model.config)
            pos = 0
            last = None
            for piece in list(chunks) + [qt]:
                ids = tok(piece, return_tensors="pt", add_special_tokens=False)[
                    "input_ids"
                ].to(device)
                pids = torch.arange(pos, pos + ids.shape[1], device=device).unsqueeze(0)
                out = b._forward_keep_last(
                    model,
                    input_ids=ids,
                    position_ids=pids,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = out.past_key_values
                last = out.logits[0, -1, :]
                pos += ids.shape[1]
            nxt = last.argmax().view(1, 1)
            for _ in range(b.MAX_NEW - 1):
                pids = torch.tensor([[pos]], device=device)
                og = b._forward_keep_last(
                    model,
                    input_ids=nxt,
                    position_ids=pids,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = og.past_key_values
                nxt = og.logits[0, -1, :].argmax().view(1, 1)
                pos += 1
                if int(nxt[0, 0]) == tok.eos_token_id:
                    break
        finally:
            rm()
            rl()
            rf()

    rows = []
    for length in lengths:
        n_chunk, chunk, conc = LENGTH_PLAN[length]
        samples = load(DATASET, conc, n_chunk, chunk, length)
        if not samples:
            print(f"[skip] no samples for {length}")
            continue
        print("\n" + "=" * 84)
        print(f"Length {length // 1000}K | concurrency={conc} | dataset={DATASET}")
        print("=" * 84)

        # warmup once to remove first-call overhead from QPS
        baseline_request(samples[0])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s in samples:
            baseline_request(s)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        base_t = time.perf_counter() - t0

        redknot_request(samples[0], n_chunk, chunk)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s in samples:
            redknot_request(s, n_chunk, chunk)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        rk_t = time.perf_counter() - t0

        print(_summ("baseline", conc, base_t))
        print(_summ("RedKnot", conc, rk_t))
        print(f"QPS speedup = {(conc / rk_t) / (conc / base_t):.2f}x")
        rows.append((length, conc, base_t, rk_t))
    return rows


@torch.no_grad()
def run_qwen3(lengths):
    """QPS benchmark for Qwen3-32B (INT4 NF4, single GPU).

    Uses the same offline_prefill_segments + run_redknot_offlinekv path as
    benchmark_RedKnot_Qwen3_RAG.py, with HotpotQA data padded to target length.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from sglang.srt.layers.attention.redknot import (
        HeadClassConfig,
        SparseFFNSchedule,
        offline_prefill_segments,
        run_redknot_offlinekv,
    )
    import random

    MODEL_PATH = os.environ.get(
        "REDKNOT_MODEL_PATH",
        "/mnt/tidal-alsh01/dataset/redone/096/models/Qwen3-32B",
    )
    HEAD_CFG_JSON = str(_HERE / "head_class/qwen3-32B_optimal_g15_lf_ret.json")
    FFN_CFG_JSON = str(_HERE / "sparse_ffn_params/qwen3-32B.json")
    HOTPOT = os.environ.get(
        "REDKNOT_HOTPOT_PARQUET",
        str(_HERE / "datasets/HotpotQA/distractor/validation-00000-of-00001.parquet"),
    )
    WINDOW = 256 + 4
    SEED = 2026

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hc = HeadClassConfig.from_json(HEAD_CFG_JSON)
    hc.merge_retrieval_to_global()
    with open(FFN_CFG_JSON) as f:
        ffn_cfg = json.load(f)
    sched = SparseFFNSchedule(**ffn_cfg)

    qc = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=qc,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).eval()

    def _query_text(q):
        return (
            "\n\nUsing only the documents above, give the shortest exact answer "
            "span to the question (a name, entity, number, or short noun phrase). "
            "Answer with the span only, no explanation.\n"
            f"Question: {q}\nAnswer:"
        )

    def load_hotpot(n_samples, n_segments, tokens_per_seg):
        import pandas as pd

        df = pd.read_parquet(HOTPOT)
        idxs = list(range(len(df)))
        random.Random(SEED).shuffle(idxs)
        out = []
        for i in idxs:
            if len(out) >= n_samples:
                break
            row = dict(df.iloc[i])
            q = str(row.get("question", ""))
            a = str(row.get("answer", ""))
            ctx = row.get("context", {})
            titles, sents = ctx.get("title", []), ctx.get("sentences", [])
            if hasattr(titles, "tolist"):
                titles = titles.tolist()
            if hasattr(sents, "tolist"):
                sents = sents.tolist()
            paras = []
            for t, s in zip(titles, sents):
                if hasattr(s, "tolist"):
                    s = s.tolist()
                paras.append(f"{t}. {' '.join(map(str, s))}")
            if not q or not paras or not a:
                continue
            # build n_segments docs, each tokens_per_seg long
            all_ids = tok("\n\n".join(paras), add_special_tokens=False)["input_ids"]
            # pad by cycling
            while len(all_ids) < n_segments * tokens_per_seg:
                all_ids = all_ids + all_ids
            all_ids = all_ids[: n_segments * tokens_per_seg]
            docs = [
                tok.decode(all_ids[k : k + tokens_per_seg], skip_special_tokens=True)
                for k in range(0, n_segments * tokens_per_seg, tokens_per_seg)
            ]
            out.append({"question": q, "docs": docs})
        return out

    def baseline_request(sample, n_segments, tokens_per_seg):
        qt = _query_text(sample["question"])
        full_text = "\n\n".join(sample["docs"])
        ids = tok(full_text + qt, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(model.device)
        out = model(input_ids=ids, use_cache=True)
        nxt = out.logits[0, -1, :].argmax().view(1, 1)
        past = out.past_key_values
        for _ in range(MAX_NEW - 1):
            og = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = og.past_key_values
            nxt = og.logits[0, -1, :].argmax().view(1, 1)
            if int(nxt[0, 0]) == tok.eos_token_id:
                break
        del past
        gc.collect()
        torch.cuda.empty_cache()

    def redknot_request(sample, n_segments, tokens_per_seg):
        qt = _query_text(sample["question"])
        chunk_size = max(4096, tokens_per_seg + 96)
        segs = offline_prefill_segments(
            model, tok, sample["docs"], chunk_size=chunk_size, model_id=MODEL_PATH
        )
        run_redknot_offlinekv(
            model,
            tok,
            segments_offline=segs,
            query_text=qt,
            head_cfg=hc,
            max_new_tokens=MAX_NEW,
            kernel="fa3_parallel",
            sparse_ffn_schedule=sched,
            use_compile=True,
        )
        del segs
        gc.collect()
        torch.cuda.empty_cache()

    rows = []
    for length in lengths:
        n_chunk, chunk, conc = LENGTH_PLAN[length]
        n_segments = n_chunk
        tokens_per_seg = chunk
        samples = load_hotpot(conc, n_segments, tokens_per_seg)
        if not samples:
            print(f"[skip] no samples for {length}")
            continue
        print("\n" + "=" * 84)
        print(f"Qwen3 | Length {length // 1000}K | concurrency={conc}")
        print("=" * 84)

        # warmup
        baseline_request(samples[0], n_segments, tokens_per_seg)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s in samples:
            baseline_request(s, n_segments, tokens_per_seg)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        base_t = time.perf_counter() - t0

        # RedKnot warmup (first call triggers compile)
        redknot_request(samples[0], n_segments, tokens_per_seg)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s in samples:
            redknot_request(s, n_segments, tokens_per_seg)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        rk_t = time.perf_counter() - t0

        print(_summ("baseline", conc, base_t))
        print(_summ("RedKnot", conc, rk_t))
        print(f"QPS speedup = {(conc / rk_t) / (conc / base_t):.2f}x")
        rows.append((length, conc, base_t, rk_t))
    return rows


@torch.no_grad()
def run_llama33(lengths):
    """QPS benchmark for Llama-3.3-70B-Instruct (INT4 NF4 or bf16).

    Uses the same offline_prefill_segments + run_redknot_offlinekv path as
    benchmark_RedKnot_Llama3.3_RAG.py, with LongBench data padded to target.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from sglang.srt.layers.attention.redknot import (
        HeadClassConfig,
        SparseFFNSchedule,
        offline_prefill_segments,
        run_redknot_offlinekv,
    )

    MODEL_PATH = os.environ.get(
        "REDKNOT_MODEL_PATH",
        "/mnt/tidal-alsh01/dataset/redone/096/models/Llama-3.3-70B-Instruct",
    )
    HEAD_CFG_JSON = str(_HERE / "head_class/llama-70B_sweetspot_g10_w4096.json")
    FFN_CFG_JSON = str(_HERE / "sparse_ffn_params/llama-70B.json")
    LONGBENCH_DIR = os.environ.get(
        "REDKNOT_LONGBENCH_DIR",
        "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
    )
    CHUNK_TOKENS = 4000

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hc = HeadClassConfig.from_json(HEAD_CFG_JSON)
    hc.merge_retrieval_to_global()
    # Fixed local-window from FFN config
    with open(FFN_CFG_JSON) as f:
        ffn_cfg = json.load(f)
    _wfix = ffn_cfg.get("local_window", 0)
    if _wfix > 0:
        hc.set_local_window(_wfix)
    sched = SparseFFNSchedule(
        dense_until=ffn_cfg["dense_until"],
        mass_thresh=ffn_cfg["mass_thresh"],
        deep_layer_start=ffn_cfg["deep_layer_start"],
        mass_thresh_deep=ffn_cfg["mass_thresh_deep"],
        recent_n=ffn_cfg["recent_n"],
    )

    dtype_mode = os.environ.get("REDKNOT_DTYPE", "int4").lower()
    device_map = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
    if dtype_mode == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).eval()
    else:
        qc = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=qc,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).eval()

    def _query_text(q):
        return (
            "\n\nAnswer the question based only on the documents above. "
            "Give the shortest exact answer span (a name, entity, number, or short "
            "phrase), with no explanation.\nQuestion: " + q + "\nAnswer:"
        )

    def load_longbench_padded(ds_name, n_samples, target_tokens):
        path = os.path.join(LONGBENCH_DIR, f"{ds_name}.jsonl")
        raw = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("input") and r.get("context") and r.get("answers"):
                    raw.append(r)
        out = []
        n = len(raw)
        for i in range(n):
            if len(out) >= n_samples:
                break
            base = raw[i]
            q = base["input"]
            ctx_tokens = tok(base["context"], add_special_tokens=False)["input_ids"]
            j = (i + 1) % n
            while len(ctx_tokens) < target_tokens and j != i:
                extra = tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
                ctx_tokens = ctx_tokens + extra
                j = (j + 1) % n
            ctx_tokens = ctx_tokens[:target_tokens]
            docs = []
            for k in range(0, len(ctx_tokens), CHUNK_TOKENS):
                piece = ctx_tokens[k : k + CHUNK_TOKENS]
                if len(piece) < 64:
                    break
                docs.append(tok.decode(piece, skip_special_tokens=True))
            if len(docs) < 2:
                continue
            out.append({"question": q, "docs": docs})
        return out

    def baseline_request(sample):
        qt = _query_text(sample["question"])
        full_text = "\n\n".join(sample["docs"])
        ids = tok(full_text + qt, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(model.device)
        out = model(input_ids=ids, use_cache=True)
        nxt = out.logits[0, -1, :].argmax().view(1, 1)
        past = out.past_key_values
        for _ in range(MAX_NEW - 1):
            og = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = og.past_key_values
            nxt = og.logits[0, -1, :].argmax().view(1, 1)
            if int(nxt[0, 0]) == tok.eos_token_id:
                break
        del past
        gc.collect()
        torch.cuda.empty_cache()

    def redknot_request(sample):
        qt = _query_text(sample["question"])
        chunk_size = max(4096, CHUNK_TOKENS + 96)
        segs = offline_prefill_segments(
            model, tok, sample["docs"], chunk_size=chunk_size, model_id=MODEL_PATH
        )
        run_redknot_offlinekv(
            model,
            tok,
            segments_offline=segs,
            query_text=qt,
            head_cfg=hc,
            max_new_tokens=MAX_NEW,
            kernel="fa3_parallel",
            sparse_ffn_schedule=sched,
            use_compile=True,
        )
        del segs
        gc.collect()
        torch.cuda.empty_cache()

    rows = []
    for length in lengths:
        n_chunk, chunk, conc = LENGTH_PLAN[length]
        samples = load_longbench_padded(DATASET, conc, length)
        if not samples:
            print(f"[skip] no samples for {length}")
            continue
        print("\n" + "=" * 84)
        print(
            f"Llama3.3 | Length {length // 1000}K | concurrency={conc} | dataset={DATASET}"
        )
        print("=" * 84)

        # warmup
        baseline_request(samples[0])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s in samples:
            baseline_request(s)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        base_t = time.perf_counter() - t0

        # RedKnot warmup
        redknot_request(samples[0])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s in samples:
            redknot_request(s)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        rk_t = time.perf_counter() - t0

        print(_summ("baseline", conc, base_t))
        print(_summ("RedKnot", conc, rk_t))
        print(f"QPS speedup = {(conc / rk_t) / (conc / base_t):.2f}x")
        rows.append((length, conc, base_t, rk_t))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen35", "qwen3", "llama33"])
    ap.add_argument(
        "--lengths",
        default="16000,32000,64000",
        help="comma-separated context lengths (16000,32000,64000)",
    )
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]

    if args.model == "qwen35":
        rows = run_qwen35(lengths)
    elif args.model == "qwen3":
        rows = run_qwen3(lengths)
    elif args.model == "llama33":
        rows = run_llama33(lengths)
    else:
        raise SystemExit(f"Unknown model: {args.model}")

    print("\n" + "=" * 84)
    print(f"QPS Summary — {args.model} (baseline vs RedKnot)")
    print("=" * 84)
    print(
        f" {'length':>8} {'conc':>5} {'base QPS':>10} {'rk QPS':>10} {'QPS speedup':>12}"
    )
    for length, conc, base_t, rk_t in rows:
        bq, rq = conc / base_t, conc / rk_t
        print(
            f" {length // 1000:>7}K {conc:>5} {bq:>10.3f} {rq:>10.3f} {rq / bq:>11.2f}x"
        )
    print("=" * 84)


if __name__ == "__main__":
    main()
