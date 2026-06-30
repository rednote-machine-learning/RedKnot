#!/usr/bin/env python3
"""Native sliding-window attention (SWA) KV trimming for Qwen3-32B.

Qwen3 *supports* sliding-window attention natively: when
``use_sliding_window=True`` and a layer's ``layer_types[i] == "sliding_attention"``
(which the config sets for ``i >= max_window_layers``), that layer applies a
sliding-window causal mask of width ``sliding_window``. A query then attends
ONLY to KV within the window, so KV entries outside the window are *masked out
of the attention computation*. Dropping them is therefore **lossless** — unlike
the earlier experiments that zeroed KV under full causal attention (an
approximation that cost +5-13% PPL).

This script runs two comparisons:

  (a) SWA-trim vs SWA-notrim
      Same SWA model. We zero the out-of-window KV of sliding layers and check
      the decode is *bit-for-bit identical* (cos = 1.0, PPL unchanged). This
      proves trimming is lossless once SWA is enabled.

  (b) SWA vs full-attention baseline
      How much does enabling SWA itself change outputs vs the original
      full-attention model? (A model-behavior change, independent of trimming.)

Metrics: logits cos(d1)/avg, greedy token match, continuation PPL.

Usage
-----
  CUDA_VISIBLE_DEVICES=0,1 python test_swa_native.py
  SWA_MWL=48 SWA_WINDOW=4096 SWA_PREFIX_LENGTHS=8192,16384,32768 \
    python test_swa_native.py
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_PATH = os.environ.get(
    "SWA_MODEL_PATH", "/mnt/tidal-alsh01/dataset/redone/096/models/Qwen3-32B"
)
PREFIX_LENGTHS = [
    int(x) for x in os.environ.get("SWA_PREFIX_LENGTHS", "8192,16384,32768").split(",")
]
WINDOW = int(os.environ.get("SWA_WINDOW", "4096"))
MWL = int(os.environ.get("SWA_MWL", "48"))  # max_window_layers
MAX_NEW = int(os.environ.get("SWA_MAX_NEW_TOKENS", "48"))
PPL_SEG = int(os.environ.get("SWA_PPL_SEG", "256"))
DEV = "cuda:0"


def cos(a, b):
    return F.cosine_similarity(
        a.double().flatten().unsqueeze(0), b.double().flatten().unsqueeze(0)
    ).item()


def make_prefix(tok, n):
    seed = (
        "The history of artificial intelligence began in antiquity with "
        "myths and stories of artificial beings endowed with intelligence. "
        "Philosophers attempted to describe human thinking as the mechanical "
        "manipulation of symbols, culminating in the programmable digital "
        "computer, a machine based on mathematical reasoning. "
    )
    ids = tok(seed, add_special_tokens=False)["input_ids"]
    rep = ids * (n // len(ids) + 2)
    return torch.tensor([rep[:n]], dtype=torch.long)


def sliding_layer_mask(layer_types):
    return [t == "sliding_attention" for t in layer_types]


def get_kv(cache, li):
    if hasattr(cache, "layers"):
        return cache.layers[li].keys, cache.layers[li].values
    if hasattr(cache, "key_cache"):
        return cache.key_cache[li], cache.value_cache[li]
    return cache[li][0], cache[li][1]


def trim_sliding(cache, seq_len, is_sliding, window):
    """Zero out-of-window KV for sliding layers. Lossless under SWA mask.

    Returns (bytes_full, bytes_kept)."""
    w_start = max(0, seq_len - window)
    nl = len(is_sliding)
    bf = bk = 0
    for li in range(nl):
        k, v = get_kv(cache, li)
        H, L, D = k.shape[1], k.shape[2], k.shape[3]
        elem = k.element_size()
        per = 2 * H * D * elem
        bf += L * per
        if is_sliding[li] and w_start > 0:
            k[:, :, :w_start, :] = 0
            v[:, :, :w_start, :] = 0
            bk += (L - w_start) * per
        else:
            bk += L * per
    return bf, bk


@torch.no_grad()
def greedy(model, cache, first, tok, n, dev):
    nxt = torch.tensor([[first]], device=dev)
    toks = [first]
    logs = []
    for _ in range(n - 1):
        o = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        lg = o.logits[0, -1, :]
        logs.append(lg.detach().cpu().clone())
        nxt = lg.argmax().view(1, 1)
        t = int(nxt[0, 0])
        toks.append(t)
        if t == tok.eos_token_id:
            break
    return toks, logs


def clone_cache(cache):
    from copy import deepcopy

    return deepcopy(cache)


@torch.no_grad()
def ppl_cont(model, ids, ctx_len, seg, dev, is_sliding=None, window=None, trim=False):
    ctx = ids[:, :ctx_len]
    cont = ids[:, ctx_len : ctx_len + seg]
    if cont.shape[1] < 2:
        return float("nan")
    o = model(input_ids=ctx, use_cache=True)
    cache = o.past_key_values
    if trim and is_sliding is not None:
        trim_sliding(cache, ctx_len, is_sliding, window)
    nlls = []
    lp = F.log_softmax(o.logits[0, -1].float(), -1)
    nlls.append(-lp[int(cont[0, 0])].item())
    cur = cache
    for i in range(1, cont.shape[1]):
        oo = model(input_ids=cont[:, i - 1 : i], past_key_values=cur, use_cache=True)
        cur = oo.past_key_values
        lp = F.log_softmax(oo.logits[0, -1].float(), -1)
        nlls.append(-lp[int(cont[0, i])].item())
    del o, cache, cur
    gc.collect()
    torch.cuda.empty_cache()
    return math.exp(sum(nlls) / len(nlls))


@torch.no_grad()
def run_prefix(swa_model, full_model, tok, plen, is_sliding):
    ids = make_prefix(tok, plen).to(DEV)
    q = tok(
        "\n\nSummarize the key points:\n", add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(DEV)
    inp = torch.cat([ids, q], dim=1)
    total = inp.shape[1]

    # ---- SWA model prefill ----
    o = swa_model(input_ids=inp, use_cache=True)
    first = int(o.logits[0, -1, :].argmax())

    # (a) SWA notrim decode
    c_notrim = clone_cache(o.past_key_values)
    tok_notrim, log_notrim = greedy(swa_model, c_notrim, first, tok, MAX_NEW, DEV)
    del c_notrim
    gc.collect()
    torch.cuda.empty_cache()

    # (a) SWA trim decode
    c_trim = o.past_key_values
    bf, bk = trim_sliding(c_trim, total, is_sliding, WINDOW)
    tok_trim, log_trim = greedy(swa_model, c_trim, first, tok, MAX_NEW, DEV)
    del o, c_trim
    gc.collect()
    torch.cuda.empty_cache()

    n = min(len(log_notrim), len(log_trim))
    a_cos = [cos(log_notrim[i], log_trim[i]) for i in range(n)]
    a_cos_d1 = a_cos[0] if a_cos else 1.0
    a_cos_avg = sum(a_cos) / len(a_cos) if a_cos else 1.0
    m = min(len(tok_notrim), len(tok_trim))
    a_match = sum(1 for i in range(m) if tok_notrim[i] == tok_trim[i]) / m if m else 1.0
    saved = 100 * (bf - bk) / bf if bf else 0

    # (a) PPL: SWA notrim vs SWA trim
    ppl_a_base = ppl_a_trim = float("nan")
    if total > WINDOW + PPL_SEG + 100:
        cl = total - PPL_SEG
        ppl_a_base = ppl_cont(swa_model, inp, cl, PPL_SEG, DEV)
        ppl_a_trim = ppl_cont(
            swa_model, inp, cl, PPL_SEG, DEV, is_sliding, WINDOW, trim=True
        )

    # (b) SWA vs full-attention baseline (greedy + cos on first step)
    b_cos_d1 = b_match = float("nan")
    ppl_b_full = float("nan")
    if full_model is not None:
        of = full_model(input_ids=inp, use_cache=True)
        first_f = int(of.logits[0, -1, :].argmax())
        cf = of.past_key_values
        tok_full, log_full = greedy(full_model, cf, first_f, tok, MAX_NEW, DEV)
        del of, cf
        gc.collect()
        torch.cuda.empty_cache()
        nb = min(len(log_full), len(log_notrim))
        b_cos_d1 = cos(log_full[0], log_notrim[0]) if nb else float("nan")
        mb = min(len(tok_full), len(tok_notrim))
        b_match = (
            sum(1 for i in range(mb) if tok_full[i] == tok_notrim[i]) / mb if mb else 0
        )
        if total > WINDOW + PPL_SEG + 100:
            ppl_b_full = ppl_cont(full_model, inp, total - PPL_SEG, PPL_SEG, DEV)

    return {
        "prefix_len": plen,
        "total_len": total,
        "saved_pct": saved,
        # (a) lossless check
        "a_cos_d1": a_cos_d1,
        "a_cos_avg": a_cos_avg,
        "a_match": a_match,
        "a_ppl_notrim": ppl_a_base,
        "a_ppl_trim": ppl_a_trim,
        # (b) swa vs full
        "b_cos_d1_swa_vs_full": b_cos_d1,
        "b_match_swa_vs_full": b_match,
        "b_ppl_swa_notrim": ppl_a_base,
        "b_ppl_full": ppl_b_full,
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

    print("=" * 96)
    print(" Native SWA KV-trim test (Qwen3-32B)")
    print(f" window={WINDOW} max_window_layers={MWL} prefixes={PREFIX_LENGTHS}")
    print("=" * 96)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # SWA-enabled config
    cfg = Qwen3Config.from_pretrained(MODEL_PATH)
    cfg.use_sliding_window = True
    cfg.sliding_window = WINDOW
    cfg.max_window_layers = MWL
    cfg.layer_types = [
        "sliding_attention" if (i >= MWL) else "full_attention"
        for i in range(cfg.num_hidden_layers)
    ]
    is_sliding = sliding_layer_mask(cfg.layer_types)
    n_slide = sum(is_sliding)
    print(
        f" sliding layers: {n_slide}/{cfg.num_hidden_layers} "
        f"(layers {MWL}..{cfg.num_hidden_layers - 1})"
    )

    print(" loading SWA model (bf16, 2 GPU) ...")
    swa = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "40GiB", 1: "40GiB"},
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).eval()

    rows = []
    for plen in PREFIX_LENGTHS:
        print(f"\n[prefix={plen}]")
        try:
            r = run_prefix(swa, None, tok, plen, is_sliding)  # full baseline below
        except torch.cuda.OutOfMemoryError as e:
            print("  OOM")
            torch.cuda.empty_cache()
            continue
        rows.append(r)
        print(
            f"  (a) SWA-trim vs SWA-notrim: cos_d1={r['a_cos_d1']:.6f} "
            f"avg={r['a_cos_avg']:.6f} match={r['a_match']:.1%} "
            f"PPL {r['a_ppl_notrim']:.4f}->{r['a_ppl_trim']:.4f}  saved={r['saved_pct']:.1f}%"
        )

    # (b) reload full-attention model for the SWA-vs-full PPL comparison.
    # 32B bf16 needs 2 GPUs, so we hold only one model at a time: SWA PPL was
    # captured above (a_ppl_notrim == SWA no-trim PPL); now load FULL for its PPL.
    del swa
    gc.collect()
    torch.cuda.empty_cache()
    print("\n loading FULL-attention baseline model for (b) PPL ...")
    full = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "40GiB", 1: "40GiB"},
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).eval()

    for r in rows:
        plen = r["prefix_len"]
        try:
            ids = make_prefix(tok, plen).to(DEV)
            q = tok(
                "\n\nSummarize the key points:\n",
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].to(DEV)
            inp = torch.cat([ids, q], 1)
            total = inp.shape[1]
            if total > WINDOW + PPL_SEG + 100:
                r["b_ppl_full"] = ppl_cont(full, inp, total - PPL_SEG, PPL_SEG, DEV)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    del full
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'=' * 96}\n SUMMARY\n{'=' * 96}")
    print(" (a) lossless check: SWA-trim vs SWA-notrim (expect cos=1.0, dPPL=0)")
    print(
        f"  {'prefix':>7} {'cos_d1':>9} {'cos_avg':>9} {'match':>7} "
        f"{'ppl_notrim':>11} {'ppl_trim':>10} {'dPPL':>8} {'saved%':>7}"
    )
    for r in rows:
        dppl = (
            r["a_ppl_trim"] - r["a_ppl_notrim"]
            if not math.isnan(r["a_ppl_notrim"])
            else float("nan")
        )
        print(
            f"  {r['prefix_len']:>7} {r['a_cos_d1']:>9.6f} {r['a_cos_avg']:>9.6f} "
            f"{r['a_match']:>6.1%} {r['a_ppl_notrim']:>11.4f} {r['a_ppl_trim']:>10.4f} "
            f"{dppl:>+8.5f} {r['saved_pct']:>6.1f}%"
        )
    print("\n (b) SWA vs full-attention baseline (model-behavior change)")
    print(f"  {'prefix':>7} {'ppl_full':>10} {'ppl_swa':>10} {'dPPL':>8}")
    for r in rows:
        dppl = (
            r["b_ppl_swa_notrim"] - r["b_ppl_full"]
            if not math.isnan(r.get("b_ppl_full", float("nan")))
            else float("nan")
        )
        print(
            f"  {r['prefix_len']:>7} {r.get('b_ppl_full', float('nan')):>10.4f} "
            f"{r['b_ppl_swa_notrim']:>10.4f} {dppl:>+8.4f}"
        )

    out = Path(__file__).with_suffix(".results.json")
    json.dump(
        {
            "config": {
                "window": WINDOW,
                "max_window_layers": MWL,
                "n_sliding": n_slide,
            },
            "rows": rows,
        },
        open(out, "w"),
        indent=2,
        default=str,
    )
    print(f"\n results -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
