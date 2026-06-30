#!/usr/bin/env python3
"""Throughput benchmark for Qwen3.5-397B-A17B (FP8): RedKnot vs full-recompute.

Measures end-to-end token throughput INCLUDING input (prefill) and output
(decode) tokens:  throughput = (input_tokens + output_tokens) / wall_time.

Also breaks it down into prefill throughput (input_tokens / prefill_time) and
decode throughput (output_tokens / decode_time) so the prefill-vs-decode split
is visible.

Two systems are compared on identical inputs:
  * BASELINE   : full-recompute, dense attention (single prefill, native).
  * RedKnot    : carry-prefix linear status + head-class full attn + deep MoE
                 token-sparsity (the Qwen3.5-397B sweet-spot config).

NOTE: This runs via transformers device_map (inter-GPU pipeline serial), so
wall-clock throughput reflects that single-request setup, not a continuous-
batching server. It is an apples-to-apples RedKnot-vs-baseline comparison on
the SAME runtime, which is what we want for the compute-saving -> latency study.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

MODEL = os.environ["REDKNOT_MODEL_PATH"]
LB = os.environ["REDKNOT_LONGBENCH_DIR"]
DATASET = os.environ.get("REDKNOT_DATASETS", "triviaqa").split(",")[0]
N = int(os.environ.get("REDKNOT_N_SAMPLES", "10"))
SEED = int(os.environ.get("REDKNOT_SEED", "0"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "64"))
N_CHUNK = int(os.environ.get("REDKNOT_N_CHUNK", "4"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK", "8000"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
QP = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"

# RedKnot sweet-spot knobs (Qwen3.5-397B)
FRAC_GLOBAL = float(os.environ.get("REDKNOT_FRAC_GLOBAL", "0.4"))
FULL_WINDOW = int(os.environ.get("REDKNOT_FULL_WINDOW", "2048"))
DENSE_FULL_LAYERS = int(os.environ.get("REDKNOT_DENSE_FULL_LAYERS", "9"))
DENSE_PREFIX = int(os.environ.get("REDKNOT_DENSE_PREFIX_LAYERS", "5"))
DECAY_Q = float(os.environ.get("REDKNOT_DECAY_QUANTILE", "0.95"))
SAFETY = float(os.environ.get("REDKNOT_SAFETY", "2.0"))
MIN_WINDOW = int(os.environ.get("REDKNOT_MIN_WINDOW", "256"))
LINEAR_SEG = int(os.environ.get("REDKNOT_LINEAR_SEG", "2048"))
MOE_MASS_THRESH = float(os.environ.get("REDKNOT_MOE_MASS_THRESH", "0.7"))
DEEP_MOE_START = int(os.environ.get("REDKNOT_DEEP_MOE_START", "24"))


def _fwd_last(model, **kw):
    try:
        return model(**kw, logits_to_keep=1)
    except TypeError as e:
        if "logits_to_keep" not in str(e):
            raise
        return model(**kw)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def run_baseline(model, tok, text):
    """Full-recompute: single dense prefill + greedy decode. Returns metrics."""
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    n_in = ids.shape[1]
    _sync()
    t0 = time.perf_counter()
    out = _fwd_last(model, input_ids=ids, use_cache=True)
    nxt = out.logits[0, -1, :].argmax().view(1, 1)
    _sync()
    t_prefill = time.perf_counter() - t0
    past = out.past_key_values
    g = [int(nxt[0, 0])]
    t1 = time.perf_counter()
    for _ in range(MAX_NEW - 1):
        og = _fwd_last(model, input_ids=nxt, past_key_values=past, use_cache=True)
        past = og.past_key_values
        nxt = og.logits[0, -1, :].argmax().view(1, 1)
        tid = int(nxt[0, 0])
        g.append(tid)
        if tid == tok.eos_token_id:
            break
    _sync()
    t_dec = max(time.perf_counter() - t1, 1e-6)
    n_out = len(g)
    total_t = t_prefill + t_dec
    return {
        "n_in": n_in,
        "n_out": n_out,
        "t_prefill": t_prefill,
        "t_dec": t_dec,
        "t_total": total_t,
        "prefill_tps": n_in / t_prefill,
        "decode_tps": n_out / t_dec,
        "total_tps": (n_in + n_out) / total_t,
    }


@torch.no_grad()
def run_redknot(model, tok, drv, chunks, qt):
    """RedKnot: chunked prefill with carry-prefix linear status + head-class +
    deep MoE token-sparsity. Returns metrics (same schema as baseline)."""
    from transformers import DynamicCache

    device = model.device
    text = "\n\n".join(chunks) + qt
    ids0 = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        device
    )
    n_in = ids0.shape[1]

    # pass-1 setup (mass for MoE, per-head window for linear) — part of prefill cost
    _sync()
    t0 = time.perf_counter()
    mass = drv["collect_attention_mass"](model, ids0, deep_full_frac=0.5)
    win = build_win(model, tok, drv, text)
    head_cfg = drv["build_full_attention_head_config"](
        model.config, frac_global=FRAC_GLOBAL, local_window=FULL_WINDOW
    )
    rf = drv["_install_full_patches"](
        model, head_cfg, dense_prefix_full_layers=DENSE_FULL_LAYERS
    )
    rl = drv["install_linear_segmented"](model, win, seg=LINEAR_SEG)
    rm = drv["install_moe_token_sparse"](
        model, mass, deep_moe_start_layer=DEEP_MOE_START, mass_thresh=MOE_MASS_THRESH
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
            out = _fwd_last(
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
        _sync()
        t_prefill = time.perf_counter() - t0
        g = [int(nxt[0, 0])]
        t1 = time.perf_counter()
        for _ in range(MAX_NEW - 1):
            pids = torch.tensor([[pos]], device=device)
            og = _fwd_last(
                model,
                input_ids=nxt,
                position_ids=pids,
                past_key_values=cache,
                use_cache=True,
            )
            cache = og.past_key_values
            nxt = og.logits[0, -1, :].argmax().view(1, 1)
            tid = int(nxt[0, 0])
            g.append(tid)
            pos += 1
            if tid == tok.eos_token_id:
                break
        _sync()
        t_dec = max(time.perf_counter() - t1, 1e-6)
    finally:
        rm()
        rl()
        rf()
    n_out = len(g)
    total_t = t_prefill + t_dec
    return {
        "n_in": n_in,
        "n_out": n_out,
        "t_prefill": t_prefill,
        "t_dec": t_dec,
        "t_total": total_t,
        "prefill_tps": n_in / t_prefill,
        "decode_tps": n_out / t_dec,
        "total_tps": (n_in + n_out) / total_t,
    }


@torch.no_grad()
def build_win(model, tok, drv, sample_text):
    bm = model.model if hasattr(model, "model") else model
    ids = tok(sample_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(model.device)
    hs_in = {}
    handles = []
    for li in drv["linear_attention_layer_indices"](model.config):

        def mk(_li):
            def hook(m, args, kwargs):
                hs = (
                    args[0]
                    if args and torch.is_tensor(args[0])
                    else kwargs.get("hidden_states")
                )
                if hs is not None:
                    hs_in[_li] = hs.detach()

            return hook

        handles.append(
            bm.layers[li].linear_attn.register_forward_pre_hook(
                mk(li), with_kwargs=True
            )
        )
    _fwd_last(model, input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()
    decay = drv["measure_linear_head_decay"](model, hs_in, decay_quantile=DECAY_Q)
    win = {}
    ctx = N_CHUNK * CHUNK
    for li, d in decay.items():
        if li < DENSE_PREFIX:
            win[li] = None
            continue
        memlen = 1.0 / (1.0 - d.clamp(max=0.99999))
        wt = torch.ceil(SAFETY * memlen).long().clamp(min=MIN_WINDOW)
        wt = torch.where(wt >= ctx, torch.zeros_like(wt), wt)
        win[li] = wt
    return win


def load_samples(tok):
    target = N_CHUNK * CHUNK
    raw = []
    with open(os.path.join(LB, f"{DATASET}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("input") and r.get("context") and r.get("answers"):
                raw.append(r)
    random.Random(SEED).shuffle(raw)
    out, nraw = [], len(raw)
    for i in range(nraw):
        if len(out) >= N:
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
            tok.decode(toks[k : k + CHUNK], skip_special_tokens=True)
            for k in range(0, target, CHUNK)
        ]
        out.append({"q": raw[i]["input"], "chunks": chunks})
    return out


def _avg(rows, key):
    return sum(r[key] for r in rows) / max(len(rows), 1)


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sglang.srt.layers.attention.redknot import driver_qwen35 as D

    drv = {
        k: getattr(D, k)
        for k in [
            "_install_full_patches",
            "build_full_attention_head_config",
            "collect_attention_mass",
            "full_attention_layer_indices",
            "linear_attention_layer_indices",
            "install_linear_segmented",
            "install_moe_token_sparse",
            "measure_linear_head_decay",
        ]
    }

    print("=" * 96)
    print(f"Qwen3.5-397B THROUGHPUT (input+output tok/s) | RedKnot vs full-recompute")
    print(f"dataset={DATASET} N={N} ctx={N_CHUNK * CHUNK} MAX_NEW={MAX_NEW}")
    print("=" * 96, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("loading model (once)...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    samples = load_samples(tok)
    print(f"loaded {len(samples)} samples\n", flush=True)

    # warmup (exclude first-call CUDA/JIT cost from timing)
    print("warmup ...", flush=True)
    _ = run_baseline(
        model, tok, "\n\n".join(samples[0]["chunks"]) + QP.format(q=samples[0]["q"])
    )

    base_rows, rk_rows = [], []
    print(
        f"\n{'#':>3} | {'BASE total':>11} {'RK total':>10} {'spd':>5} | {'BASE pf':>9} {'RK pf':>9} | {'BASE dec':>9} {'RK dec':>9}",
        flush=True,
    )
    print("-" * 96, flush=True)
    for i, s in enumerate(samples):
        text = "\n\n".join(s["chunks"]) + QP.format(q=s["q"])
        b = run_baseline(model, tok, text)
        base_rows.append(b)
        r = run_redknot(model, tok, drv, s["chunks"], QP.format(q=s["q"]))
        rk_rows.append(r)
        spd = r["total_tps"] / b["total_tps"]
        print(
            f"{i:>3} | {b['total_tps']:>11.1f} {r['total_tps']:>10.1f} {spd:>5.2f}x | "
            f"{b['prefill_tps']:>9.1f} {r['prefill_tps']:>9.1f} | {b['decode_tps']:>9.2f} {r['decode_tps']:>9.2f}",
            flush=True,
        )

    print("=" * 96)
    print("AVERAGES (tokens/sec, input+output):")
    bt, rt = _avg(base_rows, "total_tps"), _avg(rk_rows, "total_tps")
    bp, rp = _avg(base_rows, "prefill_tps"), _avg(rk_rows, "prefill_tps")
    bd, rd = _avg(base_rows, "decode_tps"), _avg(rk_rows, "decode_tps")
    bin_, bout = _avg(base_rows, "n_in"), _avg(base_rows, "n_out")
    print(f"  avg input tokens={bin_:.0f}  avg output tokens={bout:.0f}")
    print(f"  {'metric':<22}{'BASELINE':>14}{'RedKnot':>14}{'speedup':>10}")
    print(f"  {'total (in+out) tok/s':<22}{bt:>14.1f}{rt:>14.1f}{rt / bt:>9.2f}x")
    print(f"  {'prefill tok/s':<22}{bp:>14.1f}{rp:>14.1f}{rp / bp:>9.2f}x")
    print(f"  {'decode tok/s':<22}{bd:>14.2f}{rd:>14.2f}{rd / bd:>9.2f}x")
    print(
        f"  {'avg total latency (s)':<22}{_avg(base_rows, 't_total'):>14.2f}{_avg(rk_rows, 't_total'):>14.2f}"
        f"{_avg(base_rows, 't_total') / _avg(rk_rows, 't_total'):>9.2f}x"
    )
    print("=" * 96, flush=True)


if __name__ == "__main__":
    main()
