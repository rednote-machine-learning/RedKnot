#!/usr/bin/env python3
"""Cross-dataset accuracy validation of per-head KV trimming (Qwen3-32B).

Validates that trim<32 (accuracy-safe config) preserves quality across multiple
real long-context datasets and tasks, not just one synthetic prompt:

  * hotpotqa   (LongBench)  -- multi-hop QA      (retrieval-style attention)
  * gov_report (LongBench)  -- summarization     (long-range dependency)
  * lcc        (LongBench)  -- code completion   (structured text)
  * wikitext-103            -- language modeling (perplexity)

Metrics (baseline full-KV vs trimmed, per sample, then averaged):
  * cos(d1)         logits cosine at first decode step
  * avg cos         mean logits cosine over generated steps
  * greedy match %  fraction of generated tokens identical to baseline
  * dPPL            perplexity delta on the context (trim - base); ~0 is good
                    (PPL computed via teacher-forcing over the context with the
                     trimmed vs full cache is not directly comparable per-token,
                     so we report PPL of the *baseline* and the *trimmed* model
                     on a held-out continuation; see below.)

For PPL we use the standard sliding teacher-forced NLL over a continuation
segment, comparing the full-cache model and the trimmed-cache model predicting
the SAME continuation tokens.

Usage
-----
  CUDA_VISIBLE_DEVICES=0,1 python test/srt/redknot/test_swa_pd_datasets.py
  SWA_DS_SAMPLES=8 SWA_DS_LIST=hotpotqa,gov_report,lcc,wikitext \
    python test/srt/redknot/test_swa_pd_datasets.py
"""

from __future__ import annotations

import gc
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import test_swa_pd_2gpu as B

MODEL_PATH = B.MODEL_PATH
PROFILE_PATH = B.PROFILE_PATH
LB_DIR = "/mnt/tidal-alsh01/dataset/redone/data/sft/longbench/raw_data"
WIKITEXT = "/mnt/tidal-alsh01/dataset/redone/data/sft/wikitext/wikitext-103-raw-v1/test-00000-of-00001.parquet"

DS_LIST = os.environ.get("SWA_DS_LIST", "hotpotqa,gov_report,lcc,wikitext").split(",")
N_SAMPLES = int(os.environ.get("SWA_DS_SAMPLES", "8"))
MAX_NEW_TOKENS = int(os.environ.get("SWA_DS_NEW_TOKENS", "48"))
WINDOW_SIZE = int(os.environ.get("SWA_WINDOW_SIZE", "4096"))
SINK_SIZE = int(os.environ.get("SWA_SINK_SIZE", "128"))
TRIM_LAYER_MAX = int(os.environ.get("SWA_TRIM_LAYER_MAX", "32"))
MAX_CTX = int(os.environ.get("SWA_DS_MAX_CTX", "32000"))  # cap to fit & native ctx
MIN_CTX = int(os.environ.get("SWA_DS_MIN_CTX", "6000"))  # need > window to trim
PPL_SEG = int(os.environ.get("SWA_DS_PPL_SEG", "256"))  # continuation length for PPL
PREFILL_DEV = B.PREFILL_DEV


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_longbench(name, n, tokenizer):
    path = os.path.join(LB_DIR, f"{name}.jsonl")
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            ctx = d.get("context", "")
            q = d.get("input", "")
            if not ctx:
                continue
            # tokenized length filter
            ids = tokenizer(ctx, add_special_tokens=False)["input_ids"]
            if len(ids) < MIN_CTX:
                continue
            out.append({"context": ctx, "query": q, "answers": d.get("answers")})
            if len(out) >= n:
                break
    return out


def load_wikitext(n, tokenizer):
    import pandas as pd

    df = pd.read_parquet(WIKITEXT)
    # concatenate text into long blocks
    texts = [t for t in df["text"].tolist() if t and len(t) > 200]
    out, buf = [], ""
    for t in texts:
        buf += t
        ids = tokenizer(buf, add_special_tokens=False)["input_ids"]
        if len(ids) >= MIN_CTX:
            out.append({"context": buf, "query": "", "answers": None})
            buf = ""
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def cos(a, b):
    return F.cosine_similarity(
        a.double().flatten().unsqueeze(0), b.double().flatten().unsqueeze(0)
    ).item()


@torch.no_grad()
def ppl_on_continuation(
    model, full_ids, ctx_len, seg_len, local_mask, trim_layer_max, device, trimmed
):
    """Teacher-forced PPL of [ctx_len : ctx_len+seg_len] tokens.

    Prefill ctx; optionally trim KV; then feed the continuation tokens with the
    cache and measure NLL of each next-token prediction.  Lower is better.
    """
    ctx = full_ids[:, :ctx_len]
    cont = full_ids[:, ctx_len : ctx_len + seg_len]
    if cont.shape[1] < 2:
        return float("nan")

    out = model(input_ids=ctx, use_cache=True)
    cache = out.past_key_values
    total_len = ctx_len
    if trimmed:
        B.TRIM_LAYER_MAX = trim_layer_max
        B.trim_and_transfer(
            cache, total_len, local_mask, WINDOW_SIZE, SINK_SIZE, PREFILL_DEV
        )
    # feed continuation, gather NLL
    logits_prev = out.logits[:, -1:, :]  # predicts cont[0]
    nlls = []
    cur = cache
    inp = cont[:, :1]
    # first token prediction from prefill last logit
    lp = F.log_softmax(logits_prev[0, -1].float(), dim=-1)
    nlls.append(-lp[int(cont[0, 0])].item())
    for i in range(1, cont.shape[1]):
        o = model(input_ids=cont[:, i - 1 : i], past_key_values=cur, use_cache=True)
        cur = o.past_key_values
        lp = F.log_softmax(o.logits[0, -1].float(), dim=-1)
        nlls.append(-lp[int(cont[0, i])].item())
    del out, cache, cur
    gc.collect()
    torch.cuda.empty_cache()
    return math.exp(sum(nlls) / len(nlls))


@torch.no_grad()
def eval_sample(model, tokenizer, ctx, query, local_mask, device):
    """Return dict of metrics for one sample (baseline vs trim<32)."""
    ctx_ids = tokenizer(ctx, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ctx_ids = ctx_ids[:, :MAX_CTX].to(device)
    if query:
        q_ids = tokenizer(
            "\n\n" + query + "\nAnswer:", add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(device)
        input_ids = torch.cat([ctx_ids, q_ids], dim=1)
    else:
        input_ids = ctx_ids
    total_len = input_ids.shape[1]
    if total_len <= WINDOW_SIZE + SINK_SIZE + 4:
        return None

    # prefill
    out = model(input_ids=input_ids, use_cache=True)
    first_tid = int(out.logits[0, -1, :].argmax())

    # baseline decode (clone)
    base_cache = B._clone_cache(out.past_key_values)
    base_tokens, base_logits = B.greedy_decode(
        model, base_cache, first_tid, tokenizer, MAX_NEW_TOKENS, device
    )
    del base_cache
    gc.collect()
    torch.cuda.empty_cache()

    # trim + decode
    B.TRIM_LAYER_MAX = TRIM_LAYER_MAX
    trim_cache = out.past_key_values
    B.trim_and_transfer(
        trim_cache, total_len, local_mask, WINDOW_SIZE, SINK_SIZE, B.DECODE_DEV
    )
    trim_tokens, trim_logits = B.greedy_decode(
        model, trim_cache, first_tid, tokenizer, MAX_NEW_TOKENS, device
    )
    del out, trim_cache
    gc.collect()
    torch.cuda.empty_cache()

    n = min(len(base_logits), len(trim_logits))
    step_cos = [cos(base_logits[i], trim_logits[i]) for i in range(n)]
    cos_d1 = step_cos[0] if step_cos else 1.0
    avg_cos = sum(step_cos) / len(step_cos) if step_cos else 1.0
    m = min(len(base_tokens), len(trim_tokens))
    match = (
        sum(1 for i in range(m) if base_tokens[i] == trim_tokens[i]) / m if m else 1.0
    )

    # PPL on a continuation segment (only if enough room)
    ppl_base = ppl_trim = float("nan")
    if input_ids.shape[1] > WINDOW_SIZE + SINK_SIZE + PPL_SEG + 100:
        ctx_len = input_ids.shape[1] - PPL_SEG
        ppl_base = ppl_on_continuation(
            model,
            input_ids,
            ctx_len,
            PPL_SEG,
            local_mask,
            TRIM_LAYER_MAX,
            device,
            trimmed=False,
        )
        ppl_trim = ppl_on_continuation(
            model,
            input_ids,
            ctx_len,
            PPL_SEG,
            local_mask,
            TRIM_LAYER_MAX,
            device,
            trimmed=True,
        )

    return {
        "ctx_len": total_len,
        "cos_d1": cos_d1,
        "avg_cos": avg_cos,
        "greedy_match": match,
        "ppl_base": ppl_base,
        "ppl_trim": ppl_trim,
        "dppl": (ppl_trim - ppl_base) if not math.isnan(ppl_base) else float("nan"),
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 100)
    print(" Cross-dataset accuracy validation (Qwen3-32B, trim<32)")
    print(f" datasets={DS_LIST} samples={N_SAMPLES} new_tokens={MAX_NEW_TOKENS}")
    print(
        f" window={WINDOW_SIZE} sink={SINK_SIZE} trim_layer_max={TRIM_LAYER_MAX} "
        f"max_ctx={MAX_CTX} ppl_seg={PPL_SEG}"
    )
    print("=" * 100)

    local_mask, mstats = B.load_local_head_mask(PROFILE_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(" Loading model bf16 over 2 GPUs ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "40GiB", 1: "40GiB"},
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).eval()
    print(" Model loaded.\n")

    all_rows = []
    summary = []
    for ds in DS_LIST:
        print(f"\n{'-' * 100}\n DATASET: {ds}\n{'-' * 100}")
        if ds == "wikitext":
            samples = load_wikitext(N_SAMPLES, tokenizer)
        else:
            samples = load_longbench(ds, N_SAMPLES, tokenizer)
        print(f"  loaded {len(samples)} samples")

        ds_rows = []
        for i, s in enumerate(samples):
            try:
                r = eval_sample(
                    model, tokenizer, s["context"], s["query"], local_mask, PREFILL_DEV
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"   [{i}] OOM, skip")
                continue
            if r is None:
                continue
            r["dataset"] = ds
            ds_rows.append(r)
            all_rows.append(r)
            print(
                f"   [{i}] ctx={r['ctx_len']:>6} cos_d1={r['cos_d1']:.4f} "
                f"match={r['greedy_match']:.2%} "
                f"ppl {r['ppl_base']:.3f}->{r['ppl_trim']:.3f} (d{r['dppl']:+.4f})"
            )

        if ds_rows:
            import statistics as st

            avg = (
                lambda k: st.mean([x[k] for x in ds_rows if not math.isnan(x[k])])
                if any(not math.isnan(x[k]) for x in ds_rows)
                else float("nan")
            )
            summary.append(
                {
                    "dataset": ds,
                    "n": len(ds_rows),
                    "avg_ctx": int(avg("ctx_len")),
                    "cos_d1": avg("cos_d1"),
                    "avg_cos": avg("avg_cos"),
                    "greedy_match": avg("greedy_match"),
                    "ppl_base": avg("ppl_base"),
                    "ppl_trim": avg("ppl_trim"),
                    "dppl": avg("dppl"),
                }
            )

    # summary table
    print(
        f"\n{'=' * 100}\n CROSS-DATASET SUMMARY (trim<32 vs full-KV baseline)\n{'=' * 100}"
    )
    print(
        f"  {'dataset':>11} {'n':>3} {'avg_ctx':>8} {'cos_d1':>8} {'avg_cos':>8} "
        f"{'greedy%':>8} {'ppl_base':>9} {'ppl_trim':>9} {'dPPL':>8}"
    )
    for r in summary:
        print(
            f"  {r['dataset']:>11} {r['n']:>3} {r['avg_ctx']:>8} {r['cos_d1']:>8.4f} "
            f"{r['avg_cos']:>8.4f} {r['greedy_match']:>7.1%} "
            f"{r['ppl_base']:>9.3f} {r['ppl_trim']:>9.3f} {r['dppl']:>+8.4f}"
        )

    out = Path(__file__).with_suffix(".results.json")
    with open(out, "w") as f:
        json.dump(
            {
                "config": {
                    "window": WINDOW_SIZE,
                    "sink": SINK_SIZE,
                    "trim_layer_max": TRIM_LAYER_MAX,
                    "max_ctx": MAX_CTX,
                    "samples": N_SAMPLES,
                    "new_tokens": MAX_NEW_TOKENS,
                },
                "summary": summary,
                "rows": all_rows,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\n Results -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
