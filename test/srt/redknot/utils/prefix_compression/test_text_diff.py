#!/usr/bin/env python3
"""Real-output text-difference study for the near-lossless KV-trim config.

Single fixed config (the only one we use):
    trim<32  (trim first 32 layers' local heads only)
    sink = 128, window = 4096, dense model (NO sliding-window mask)

For 6 LongBench/WikiText datasets spanning QA / summarization / code / few-shot
/ language modeling, we greedy-decode from the full-KV baseline and from the
trimmed cache, then compare the ACTUAL generated text:

  * exact_match   : trimmed text == baseline text (char-identical)
  * token_match   : fraction of generated tokens identical (prefix-aligned)
  * norm_edit     : normalized Levenshtein distance on generated text (0=identical)

We bucket by real context length and also save side-by-side text samples.

Usage
-----
  CUDA_VISIBLE_DEVICES=0,1 python test_text_diff.py
  SWA_TD_SAMPLES=8 SWA_TD_NEW=64 python test_text_diff.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import test_swa_pd_2gpu as B

MODEL_PATH = B.MODEL_PATH
PROFILE = os.environ.get(
    "SWA_PROFILE_PATH",
    "/mnt/tidal-alsh01/dataset/redone/RedKnotV0.2/test/srt/redknot/head_class/qwen3-32B_optimal_g15_lf_ret.json",
)
LB_DIR = "/mnt/tidal-alsh01/dataset/redone/data/sft/longbench/raw_data"
WIKITEXT = "/mnt/tidal-alsh01/dataset/redone/data/sft/wikitext/wikitext-103-raw-v1/test-00000-of-00001.parquet"

DS_LIST = os.environ.get(
    "SWA_TD_LIST", "hotpotqa,multifieldqa_en,triviaqa,gov_report,lcc,wikitext"
).split(",")
N_SAMPLES = int(os.environ.get("SWA_TD_SAMPLES", "8"))
NEW = int(os.environ.get("SWA_TD_NEW", "64"))
WINDOW = 4096
SINK = 128
TRIM_LAYER_MAX = 32
MAX_CTX = int(os.environ.get("SWA_TD_MAX_CTX", "32000"))
MIN_CTX = int(os.environ.get("SWA_TD_MIN_CTX", "5000"))
DEV = "cuda:0"

DS_TASK = {
    "hotpotqa": "multi-hop QA",
    "multifieldqa_en": "multi-field QA",
    "triviaqa": "few-shot QA",
    "gov_report": "summarization",
    "lcc": "code completion",
    "wikitext": "language modeling",
}


def levenshtein(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def load_longbench(name, n, tok):
    out = []
    with open(os.path.join(LB_DIR, f"{name}.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            ctx = d.get("context", "")
            if not ctx:
                continue
            if len(tok(ctx, add_special_tokens=False)["input_ids"]) < MIN_CTX:
                continue
            out.append({"context": ctx, "query": d.get("input", "")})
            if len(out) >= n:
                break
    return out


def load_wikitext(n, tok):
    import pandas as pd

    df = pd.read_parquet(WIKITEXT)
    texts = [t for t in df["text"].tolist() if t and len(t) > 200]
    out, buf = [], ""
    for t in texts:
        buf += t
        if len(tok(buf, add_special_tokens=False)["input_ids"]) >= MIN_CTX:
            out.append({"context": buf, "query": ""})
            buf = ""
        if len(out) >= n:
            break
    return out


def bucket(ctx_len):
    for hi in (6000, 12000, 18000, 24000, 30000, 10**9):
        if ctx_len <= hi:
            return {
                6000: "<=6K",
                12000: "6-12K",
                18000: "12-18K",
                24000: "18-24K",
                30000: "24-30K",
                10**9: ">30K",
            }[hi]


@torch.no_grad()
def eval_one(model, tok, ctx, query, local_mask):
    ctx_ids = tok(ctx, add_special_tokens=False, return_tensors="pt")["input_ids"][
        :, :MAX_CTX
    ].to(DEV)
    if query:
        q = tok(
            "\n\n" + query + "\nAnswer:", add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(DEV)
        inp = torch.cat([ctx_ids, q], 1)
    else:
        inp = ctx_ids
    total = inp.shape[1]
    if total <= WINDOW + SINK + 4:
        return None

    out = model(input_ids=inp, use_cache=True)
    first = int(out.logits[0, -1, :].argmax())

    base_cache = B._clone_cache(out.past_key_values)
    base_tokens, _ = B.greedy_decode(model, base_cache, first, tok, NEW, DEV)
    base_text = tok.decode(base_tokens, skip_special_tokens=True)
    del base_cache
    gc.collect()
    torch.cuda.empty_cache()

    B.TRIM_LAYER_MAX = TRIM_LAYER_MAX
    trim_cache = out.past_key_values
    B.trim_and_transfer(trim_cache, total, local_mask, WINDOW, SINK, B.DECODE_DEV)
    trim_tokens, _ = B.greedy_decode(model, trim_cache, first, tok, NEW, DEV)
    trim_text = tok.decode(trim_tokens, skip_special_tokens=True)
    del out, trim_cache
    gc.collect()
    torch.cuda.empty_cache()

    m = min(len(base_tokens), len(trim_tokens))
    token_match = (
        sum(1 for i in range(m) if base_tokens[i] == trim_tokens[i]) / m if m else 1.0
    )
    ed = levenshtein(base_text, trim_text)
    norm_edit = ed / max(len(base_text), len(trim_text), 1)
    return {
        "ctx_len": total,
        "bucket": bucket(total),
        "exact_match": base_text == trim_text,
        "token_match": token_match,
        "norm_edit": norm_edit,
        "base_text": base_text,
        "trim_text": trim_text,
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 100)
    print(" Real-output text-difference study (config: trim<32, sink=128, dense)")
    print(f" datasets={DS_LIST} samples={N_SAMPLES} new_tokens={NEW}")
    print("=" * 100)

    local_mask, _ = B.load_local_head_mask(PROFILE)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(" loading model bf16 2GPU ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "40GiB", 1: "40GiB"},
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).eval()
    print(" model loaded.\n")

    all_rows, samples = [], []
    summary = []
    for ds in DS_LIST:
        print(f"\n{'-' * 100}\n DATASET {ds} ({DS_TASK.get(ds, '?')})\n{'-' * 100}")
        data = (
            load_wikitext(N_SAMPLES, tok)
            if ds == "wikitext"
            else load_longbench(ds, N_SAMPLES, tok)
        )
        print(f"  loaded {len(data)}")
        rows = []
        for i, s in enumerate(data):
            try:
                r = eval_one(model, tok, s["context"], s["query"], local_mask)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"   [{i}] OOM")
                continue
            if not r:
                continue
            r["dataset"] = ds
            rows.append(r)
            all_rows.append(r)
            print(
                f"   [{i}] ctx={r['ctx_len']:>6} ({r['bucket']:>6}) "
                f"exact={'Y' if r['exact_match'] else 'N'} "
                f"tok_match={r['token_match']:.1%} edit={r['norm_edit']:.3f}"
            )
            if len(samples) < 12:  # keep a few full samples for the paper
                samples.append(
                    {
                        "dataset": ds,
                        "ctx_len": r["ctx_len"],
                        "base": r["base_text"],
                        "trim": r["trim_text"],
                        "exact": r["exact_match"],
                    }
                )
        if rows:
            import statistics as st

            summary.append(
                {
                    "dataset": ds,
                    "task": DS_TASK.get(ds, "?"),
                    "n": len(rows),
                    "avg_ctx": int(st.mean(r["ctx_len"] for r in rows)),
                    "exact_match_rate": sum(r["exact_match"] for r in rows) / len(rows),
                    "avg_token_match": st.mean(r["token_match"] for r in rows),
                    "avg_norm_edit": st.mean(r["norm_edit"] for r in rows),
                }
            )

    # ---- summary table ----
    print(
        f"\n{'=' * 100}\n TEXT-DIFFERENCE SUMMARY (trim<32, sink=128 vs full-KV baseline)\n{'=' * 100}"
    )
    print(
        f"  {'dataset':>15} {'task':>16} {'n':>3} {'avg_ctx':>8} "
        f"{'exact%':>7} {'tok_match':>9} {'norm_edit':>9}"
    )
    for r in summary:
        print(
            f"  {r['dataset']:>15} {r['task']:>16} {r['n']:>3} {r['avg_ctx']:>8} "
            f"{r['exact_match_rate']:>6.0%} {r['avg_token_match']:>8.1%} "
            f"{r['avg_norm_edit']:>9.3f}"
        )

    # ---- bucket-by-length table ----
    print(f"\n{'=' * 100}\n BY CONTEXT LENGTH\n{'=' * 100}")
    from collections import defaultdict

    bk = defaultdict(list)
    order = ["<=6K", "6-12K", "12-18K", "18-24K", "24-30K", ">30K"]
    for r in all_rows:
        bk[r["bucket"]].append(r)
    print(f"  {'bucket':>8} {'n':>3} {'exact%':>7} {'tok_match':>9} {'norm_edit':>9}")
    for b in order:
        rs = bk.get(b, [])
        if not rs:
            continue
        import statistics as st

        print(
            f"  {b:>8} {len(rs):>3} {sum(x['exact_match'] for x in rs) / len(rs):>6.0%} "
            f"{st.mean(x['token_match'] for x in rs):>8.1%} "
            f"{st.mean(x['norm_edit'] for x in rs):>9.3f}"
        )

    out = Path(__file__).with_suffix(".results.json")
    json.dump(
        {
            "config": {
                "trim_layer_max": TRIM_LAYER_MAX,
                "sink": SINK,
                "window": WINDOW,
                "new_tokens": NEW,
            },
            "summary": summary,
            "rows": [
                {k: v for k, v in r.items() if k not in ("base_text", "trim_text")}
                for r in all_rows
            ],
            "samples": samples,
        },
        open(out, "w"),
        indent=2,
        default=str,
    )
    print(f"\n results -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
