#!/usr/bin/env python3
"""Multi-context REAL latency measurement for Qwen3.5-397B-A17B (FP8).

For each context length in {16K, 32K, 64K, 128K}:
  * BASELINE (full-recompute, dense): REAL prefill latency + REAL decode tok/s.
  * RedKnot  (carry-prefix linear status + head-class full attn + deep MoE
              token-sparsity): REAL run; if a kernel crashes at long ctx it is
              reported as FAILED (the harness records baseline regardless).

Outputs per-context real measurements (prefill ms, decode tok/s, total tok/s).
These real numbers then feed the engine QPS/throughput extrapolation.

Single-request transformers device_map run: numbers are a faithful
RedKnot-vs-baseline comparison on the SAME runtime. Engine (continuous-batching)
QPS is derived separately from these measured prefill/decode times.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

MODEL = os.environ["REDKNOT_MODEL_PATH"]
LB = os.environ["REDKNOT_LONGBENCH_DIR"]
DATASET = os.environ.get("REDKNOT_DATASETS", "triviaqa").split(",")[0]
N = int(os.environ.get("REDKNOT_N_SAMPLES", "3"))
SEED = int(os.environ.get("REDKNOT_SEED", "0"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "64"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
CHUNK = int(os.environ.get("REDKNOT_CHUNK", "8000"))
OUT_JSON = os.environ.get(
    "REDKNOT_LATENCY_OUT",
    str(Path(__file__).resolve().parent / "figures" / "latency_breakdown_397b.json"),
)
CTXS = [
    int(x)
    for x in os.environ.get("REDKNOT_CTXS", "16000,32000,64000,128000").split(",")
]
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
def run_baseline(model, tok, ids):
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
        g.append(int(nxt[0, 0]))
    _sync()
    t_dec = max(time.perf_counter() - t1, 1e-6)
    n_out = len(g)
    return dict(
        n_in=n_in,
        n_out=n_out,
        t_prefill=t_prefill,
        t_dec=t_dec,
        prefill_tps=n_in / t_prefill,
        decode_tps=n_out / t_dec,
        total_tps=(n_in + n_out) / (t_prefill + t_dec),
    )


@torch.no_grad()
def run_redknot(model, tok, drv, chunks, qt, ctx):
    from transformers import DynamicCache

    device = model.device
    text = "\n\n".join(chunks) + qt
    ids0 = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        device
    )
    n_in = ids0.shape[1]
    _sync()
    t0 = time.perf_counter()
    mass = drv["collect_attention_mass"](model, ids0, deep_full_frac=0.5)
    win = build_win(model, tok, drv, text, ctx)
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
            g.append(int(nxt[0, 0]))
            pos += 1
        _sync()
        t_dec = max(time.perf_counter() - t1, 1e-6)
    finally:
        rm()
        rl()
        rf()
    n_out = len(g)
    return dict(
        n_in=n_in,
        n_out=n_out,
        t_prefill=t_prefill,
        t_dec=t_dec,
        prefill_tps=n_in / t_prefill,
        decode_tps=n_out / t_dec,
        total_tps=(n_in + n_out) / (t_prefill + t_dec),
    )


@torch.no_grad()
def build_win(model, tok, drv, sample_text, ctx):
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
    for li, d in decay.items():
        if li < DENSE_PREFIX:
            win[li] = None
            continue
        memlen = 1.0 / (1.0 - d.clamp(max=0.99999))
        wt = torch.ceil(SAFETY * memlen).long().clamp(min=MIN_WINDOW)
        wt = torch.where(wt >= ctx, torch.zeros_like(wt), wt)
        win[li] = wt
    return win


def make_context_text(tok, raw, target):
    """Concatenate contexts to reach exactly `target` tokens, return chunks."""
    toks = []
    j = 0
    while len(toks) < target:
        toks += tok(raw[j % len(raw)]["context"], add_special_tokens=False)["input_ids"]
        j += 1
    toks = toks[:target]
    chunks = [
        tok.decode(toks[k : k + CHUNK], skip_special_tokens=True)
        for k in range(0, target, CHUNK)
    ]
    return chunks


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
            "linear_attention_layer_indices",
            "install_linear_segmented",
            "install_moe_token_sparse",
            "measure_linear_head_decay",
        ]
    }

    print("=" * 92)
    print(
        f"Qwen3.5-397B MULTI-CONTEXT real latency | ctxs={CTXS} N={N} MAX_NEW={MAX_NEW}"
    )
    print("=" * 92, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("loading model (once)...", flush=True)
    t0 = time.time()
    _from_kwargs = dict(
        dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    )
    _mm = os.environ.get("REDKNOT_MAX_MEM_PER_GPU")  # e.g. "72GiB" — leaves headroom
    if _mm and DEVICE_MAP == "auto":
        _nd = torch.cuda.device_count()
        _from_kwargs["max_memory"] = {i: _mm for i in range(_nd)}
        _from_kwargs["max_memory"]["cpu"] = os.environ.get(
            "REDKNOT_MAX_CPU_MEM", "1500GiB"
        )
    model = AutoModelForCausalLM.from_pretrained(MODEL, **_from_kwargs).eval()
    print(f"model loaded in {time.time() - t0:.0f}s\n", flush=True)

    raw = []
    with open(os.path.join(LB, f"{DATASET}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("input") and r.get("context"):
                raw.append(r)
    random.Random(SEED).shuffle(raw)
    qlist = [r["input"] for r in raw[:N]]

    results = {}
    if os.environ.get("REDKNOT_APPEND") and os.path.exists(
        "/tmp/multi_ctx_results.pkl"
    ):
        try:
            import pickle as _pk

            with open("/tmp/multi_ctx_results.pkl", "rb") as f:
                results = _pk.load(f)
        except Exception:
            results = {}
    for ctx in CTXS:
        print(f"\n{'=' * 92}\n### CONTEXT = {ctx} tokens\n{'=' * 92}", flush=True)
        chunks = make_context_text(tok, raw, ctx)
        ids_base = tok(
            "\n\n".join(chunks) + QP.format(q=qlist[0]),
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(model.device)
        # --- BASELINE (must be real) ---
        b_rows = []
        try:
            # warmup once per ctx (exclude alloc/JIT)
            _ = run_baseline(model, tok, ids_base)
            for qi in range(N):
                idsq = tok(
                    "\n\n".join(chunks) + QP.format(q=qlist[qi]),
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"].to(model.device)
                b = run_baseline(model, tok, idsq)
                b_rows.append(b)
                print(
                    f"  [base q{qi}] prefill {b['t_prefill'] * 1000:8.1f}ms  pf_tps {b['prefill_tps']:8.0f}  dec_tps {b['decode_tps']:6.2f}",
                    flush=True,
                )
            base = {k: sum(r[k] for r in b_rows) / len(b_rows) for k in b_rows[0]}
            base["ok"] = True
        except Exception as e:
            print(f"  [base] FAILED: {e}")
            traceback.print_exc()
            base = {"ok": False, "err": str(e)}
        # --- RedKnot (real; may crash at long ctx) ---
        r_rows = []
        try:
            for qi in range(N):
                r = run_redknot(model, tok, drv, chunks, QP.format(q=qlist[qi]), ctx)
                r_rows.append(r)
                print(
                    f"  [rk   q{qi}] prefill {r['t_prefill'] * 1000:8.1f}ms  pf_tps {r['prefill_tps']:8.0f}  dec_tps {r['decode_tps']:6.2f}",
                    flush=True,
                )
            rk = {k: sum(r[k] for r in r_rows) / len(r_rows) for k in r_rows[0]}
            rk["ok"] = True
        except Exception as e:
            print(
                f"  [rk] FAILED (will extrapolate): {type(e).__name__}: {str(e)[:120]}"
            )
            rk = {"ok": False, "err": str(e)}
        results[ctx] = {"base": base, "rk": rk}
        # free per-ctx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- summary ----
    print(
        f"\n{'=' * 92}\nREAL MEASURED SUMMARY (single-request, transformers device_map)\n{'=' * 92}"
    )
    print(
        f"{'ctx':>8} | {'BASE pf ms':>11} {'BASE pf tps':>12} {'BASE dec tps':>13} | {'RK pf ms':>10} {'RK pf tps':>11} {'RK dec tps':>11} | {'pf speedup':>10}"
    )
    print("-" * 92)
    for ctx in CTXS:
        b = results[ctx]["base"]
        r = results[ctx]["rk"]
        if not b.get("ok"):
            print(f"{ctx:>8} | BASELINE FAILED: {b.get('err', '')[:60]}")
            continue
        bl = f"{b['t_prefill'] * 1000:>11.1f} {b['prefill_tps']:>12.0f} {b['decode_tps']:>13.2f}"
        if r.get("ok"):
            spd = b["t_prefill"] / r["t_prefill"]
            rk = f"{r['t_prefill'] * 1000:>10.1f} {r['prefill_tps']:>11.0f} {r['decode_tps']:>11.2f}"
            print(f"{ctx:>8} | {bl} | {rk} | {spd:>9.2f}x")
        else:
            print(f"{ctx:>8} | {bl} | {'CRASHED -> extrapolate':>34} | {'~':>10}")
    print("=" * 92, flush=True)
    # dump machine-readable for the QPS extrapolation step
    import pickle

    with open("/tmp/multi_ctx_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("saved /tmp/multi_ctx_results.pkl", flush=True)
    out_json = Path(OUT_JSON)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out_json}", flush=True)


if __name__ == "__main__":
    main()
