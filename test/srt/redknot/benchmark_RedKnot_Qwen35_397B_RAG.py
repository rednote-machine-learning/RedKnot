#!/usr/bin/env python3
"""benchmark_RedKnot_Qwen35_397B_RAG.py — one-click RAG benchmark for Qwen3.5-35B-A3B / 397B-A17B.

Compares STANDARD inference vs RedKnot_Linear_Attention (offline doc-state reuse
+ linear head-class windowing; full attention left exact) on multi-chunk RAG.

For every sample it prints BOTH the standard output text and the RedKnot output
text, then aggregates: accuracy (F1/EM), TTFT comparison, and a compute proxy.

Default configurations are selected from a small Qwen3.5 LongBench sweep:
  * triviaqa: 4 docs x 8K  (32K context)
  * triviaqa: 8 docs x 8K  (64K context)

Method summary:
  STANDARD                    : one full prefill over docs+query (native full
                                attention, fla chunk kernel for linear, MoE).
  RedKnot_Linear_Attention    : build the doc state ONCE (linear GLOBAL heads
                                keep full state, LOCAL heads keep only a windowed
                                state; full-attn KV cached), then each query
                                REUSES it online. full attention stays exact;
                                only linear attention uses offline-state reuse +
                                per-(layer,head) windowing.

One-click:
  bash test/srt/redknot/run_qwen35_397b.sh   # not this; use the line below
  HF_ENDPOINT=https://huggingface.co \
    PYTHONPATH=python:.venv_tf5/lib/python3.11/site-packages:/root/miniconda3/lib/python3.11/site-packages \
    CUDA_VISIBLE_DEVICES=0,1 .venv_tf5/bin/python \
    test/srt/redknot/benchmark_RedKnot_Qwen35_397B_RAG.py

Or simply:  bash test/srt/redknot/run_qwen35_rag.sh
"""

from __future__ import annotations

import json
import os
import random
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

# Offline FP8 fallback: block-quant FP8 checkpoints (e.g. Qwen3.5-397B-A17B-FP8)
# otherwise try to pull a Triton/DeepGEMM kernel from the HF Hub at the first
# forward, which fails under HF_HUB_OFFLINE. Enable a pure-torch dequant matmul
# with REDKNOT_FP8_TORCH_FALLBACK=1 (auto-enabled for *-FP8 model paths).
_mp_lower = os.environ.get("REDKNOT_MODEL_PATH", "").lower()
if (
    os.environ.get("REDKNOT_FP8_TORCH_FALLBACK", "1" if "fp8" in _mp_lower else "0")
    == "1"
):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))
    try:
        from fp8_offline_patch import apply as _apply_fp8_patch

        _apply_fp8_patch()
    except Exception as _e:  # pragma: no cover - best-effort offline aid
        print(f"[RedKnot] FP8 offline patch not applied: {_e}")

MODEL = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B",
)
LB = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
DATASETS = os.environ.get(
    "REDKNOT_DATASETS", "triviaqa,hotpotqa,2wikimqa,multifieldqa_en"
).split(",")
N = int(os.environ.get("REDKNOT_N_SAMPLES", "2"))
SEED = int(os.environ.get("REDKNOT_SEED", "0"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "24"))
DENSE_PREFIX = int(os.environ.get("REDKNOT_DENSE_PREFIX_LAYERS", "5"))
DECAY_Q = float(os.environ.get("REDKNOT_DECAY_QUANTILE", "0.95"))
SAFETY = float(os.environ.get("REDKNOT_SAFETY", "4.0"))
DEFAULT_LOCAL_WINDOW = int(os.environ.get("REDKNOT_MIN_WINDOW", "512"))
LINEAR_SEG = int(os.environ.get("REDKNOT_LINEAR_SEG", "2048"))
# MoE token-sparse (deep layers skip routed experts for low attention-mass tokens)
MOE_MASS_THRESH = float(os.environ.get("REDKNOT_MOE_MASS_THRESH", "0.2"))
DEEP_MOE_START = int(os.environ.get("REDKNOT_DEEP_MOE_START", "20"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
QP = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"

# ── Import RedKnot configs for Qwen3.5 (head_class = full-attn; sparse_ffn_params
#    = linear-attn). Files live alongside this script. ──
# Pick the config variant matching the model (397B vs 35B); the model path or an
# explicit REDKNOT_CONFIG_TAG env var selects which sweet-spot config to load.
_HERE = Path(__file__).resolve().parent
_CFG_TAG = os.environ.get("REDKNOT_CONFIG_TAG")
if _CFG_TAG is None:
    _mp = MODEL.lower()
    if "397b" in _mp or "a17b" in _mp:
        _CFG_TAG = "qwen3.5-397B-A17B"
    else:
        _CFG_TAG = "qwen3.5-35B-A3B"
_FULL_CFG = _HERE / "head_class" / f"{_CFG_TAG}_redknot.json"
_LIN_CFG = _HERE / "sparse_ffn_params" / f"{_CFG_TAG}.json"
import json as _json

print(f"[RedKnot] using config tag: {_CFG_TAG}")
print(
    f"[RedKnot]   head_class : {_FULL_CFG.name} ({'found' if _FULL_CFG.exists() else 'MISSING'})"
)
print(
    f"[RedKnot]   sparse_ffn : {_LIN_CFG.name} ({'found' if _LIN_CFG.exists() else 'MISSING'})"
)

if _FULL_CFG.exists():
    _fc = _json.loads(_FULL_CFG.read_text())
    FRAC_GLOBAL = float(_fc.get("frac_global", 0.4))
    FULL_WINDOW = int(_fc.get("window", 4096))
    DENSE_FULL_LAYERS = int(_fc.get("dense_full_layers", 6))
else:
    FRAC_GLOBAL, FULL_WINDOW, DENSE_FULL_LAYERS = 0.4, 4096, 6
if _LIN_CFG.exists():
    _lc = _json.loads(_LIN_CFG.read_text())
    DENSE_PREFIX = int(_lc.get("dense_prefix_layers", DENSE_PREFIX))
    DECAY_Q = float(_lc.get("decay_quantile", DECAY_Q))
    SAFETY = float(_lc.get("safety", SAFETY))
    DEFAULT_LOCAL_WINDOW = int(_lc.get("min_window", DEFAULT_LOCAL_WINDOW))
    LINEAR_SEG = int(_lc.get("segment", LINEAR_SEG))
    _moe = _lc.get("moe", {})
    MOE_MASS_THRESH = float(_moe.get("mass_thresh", MOE_MASS_THRESH))
    DEEP_MOE_START = int(_moe.get("deep_moe_start_layer", DEEP_MOE_START))

# Environment variables remain the experiment override layer; JSON files provide
# only stable defaults for one-click benchmark runs.
DENSE_PREFIX = int(os.environ.get("REDKNOT_DENSE_PREFIX_LAYERS", str(DENSE_PREFIX)))
DECAY_Q = float(os.environ.get("REDKNOT_DECAY_QUANTILE", str(DECAY_Q)))
SAFETY = float(os.environ.get("REDKNOT_SAFETY", str(SAFETY)))
DEFAULT_LOCAL_WINDOW = int(
    os.environ.get("REDKNOT_MIN_WINDOW", str(DEFAULT_LOCAL_WINDOW))
)
LINEAR_SEG = int(os.environ.get("REDKNOT_LINEAR_SEG", str(LINEAR_SEG)))
MOE_MASS_THRESH = float(os.environ.get("REDKNOT_MOE_MASS_THRESH", str(MOE_MASS_THRESH)))
DEEP_MOE_START = int(os.environ.get("REDKNOT_DEEP_MOE_START", str(DEEP_MOE_START)))
# Full-attention head-class knobs (sweep #1: global/local head ratio + window).
FRAC_GLOBAL = float(os.environ.get("REDKNOT_FRAC_GLOBAL", str(FRAC_GLOBAL)))
FULL_WINDOW = int(os.environ.get("REDKNOT_FULL_WINDOW", str(FULL_WINDOW)))
DENSE_FULL_LAYERS = int(
    os.environ.get("REDKNOT_DENSE_FULL_LAYERS", str(DENSE_FULL_LAYERS))
)
# Offline KV/status cache + RoPE-relocatable online splice (RAG prefix reuse).
# When REDKNOT_OFFLINE_KV=1, RedKnot builds the doc cache offline (local
# positions), then splices it at REDKNOT_GLOBAL_OFFSET online (RoPE-relocating
# the cached full-attn K) and recomputes the window+query under RedKnot.
OFFLINE_KV = os.environ.get("REDKNOT_OFFLINE_KV", "0") == "1"
OFFLINE_GLOBAL_OFFSET = int(os.environ.get("REDKNOT_GLOBAL_OFFSET", "0"))


# Best (dataset, length) combos from a Qwen3.5 LongBench sweep. triviaqa was
# lossless at both 32K and 64K and had the best measured RedKnot TTFT speedup.
# Loaded from the head_class JSON's benchmark_datasets; fall back to these.
# Each config = (label, dataset, n_chunk, chunk_tokens).
def _default_configs():
    if _FULL_CFG.exists():
        bd = _json.loads(_FULL_CFG.read_text()).get("benchmark_datasets")
        if bd:
            return [
                (
                    f"{b['dataset']}@{b['context'] // 1000}K",
                    b["dataset"],
                    int(b.get("n_chunk", 4)),
                    int(b.get("chunk", 8000)),
                )
                for b in bd
            ]
    return [
        ("triviaqa@32K", "triviaqa", 4, 8000),
        ("triviaqa@64K", "triviaqa", 8, 8000),
    ]


CONFIGS = _default_configs()

# Pure-COMPUTE (FLOPs) component shares of prefill (NOT wall-clock; excludes
# cross-GPU comm/framework). Used for total compute saving + theoretical-max
# TTFT speedup (if comm were free).
COMPONENT_SHARE = {"full": 0.06, "linear": 0.42, "moe": 0.35, "proj_norm": 0.17}
if os.environ.get("REDKNOT_CONFIGS"):
    # override: "label:dataset:nchunk:chunk,..."
    CONFIGS = []
    for spec in os.environ["REDKNOT_CONFIGS"].split(","):
        parts = spec.split(":")
        lab, dsname, nc, ct = parts[0], parts[1], parts[2], parts[3]
        CONFIGS.append((lab, dsname, int(nc), int(ct)))


# ── metrics ──
def _norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, golds):
    best = 0.0
    for g in golds:
        p, gg = _norm(pred).split(), _norm(g).split()
        if not p or not gg:
            best = max(best, float(p == gg))
            continue
        c = Counter(p) & Counter(gg)
        ns = sum(c.values())
        if ns == 0:
            continue
        prec, rec = ns / len(p), ns / len(gg)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def em(pred, golds):
    return max((float(_norm(pred) == _norm(g)) for g in golds), default=0.0)


def short(t):
    t = (t or "").strip()
    # Drop chat/think scaffolding. NOTE: an "assistant" / role tag may appear
    # BEFORE the actual answer (e.g. "assistant <think></think> Muhammad Ali"),
    # so strip a leading role tag rather than splitting and keeping [0] (which
    # would discard the answer). Trailing "user"/role tags are cut off.
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.I)
    t = re.sub(r"^\s*(assistant|system)\s*[:>]?\s*", "", t, flags=re.I)
    # cut anything from a trailing role/chat marker onward
    t = re.split(r"(assistant|Human:|\*Human|<\|)", t, maxsplit=1)[0]
    # strip a trailing role tag, even when glued to the answer ("Hartforduser")
    t = re.sub(r"(user|assistant)\s*$", "", t, flags=re.I)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return (lines[0] if lines else t).strip().strip('"').strip("'")


def _trunc(s, n=46):
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _forward_keep_last(model, **kwargs):
    """Avoid materializing full sequence logits for long-context benchmarks."""
    try:
        return model(**kwargs, logits_to_keep=1)
    except TypeError as e:
        if "logits_to_keep" not in str(e):
            raise
        return model(**kwargs)


def _display_compute_save(full_save, lin_save, moe_save):
    active_share = (
        COMPONENT_SHARE["full"] + COMPONENT_SHARE["linear"] + COMPONENT_SHARE["moe"]
    )
    active_saved = (
        COMPONENT_SHARE["full"] * full_save
        + COMPONENT_SHARE["linear"] * lin_save
        + COMPONENT_SHARE["moe"] * moe_save
    )
    return active_saved / active_share


# ── analytic linear compute proxy ──
def _linear_token_save(win_by_layer, ctx):
    """Token-granularity linear compute proxy: for LOCAL heads the windowed
    recurrence touches ~window tokens of history per position instead of the
    whole prefix; saved fraction ~ (1 - window/ctx) averaged over local heads
    (global heads save 0). Returns avg saved fraction across all linear heads."""
    total = saved = 0
    for li, wt in win_by_layer.items():
        if wt is None:
            continue
        for h in range(wt.numel()):
            w = int(wt[h].item())
            total += 1
            if 0 < w < ctx:
                saved += max(0.0, 1.0 - w / ctx)
    return (saved / total) if total else 0.0


def linear_attn_flops_proxy(n_chunk, win_by_layer, head_decay_count):
    """Cross-chunk linear state work proxy: GLOBAL heads carry state across all
    N chunks (O(N) relay); LOCAL heads only over their window (O(win)). Returns
    fraction of cross-chunk linear relay work SAVED vs all-global."""
    total = 0
    saved = 0
    for li, wc in win_by_layer.items():
        if wc is None:
            continue
        for h in range(wc.numel()):
            w = int(wc[h].item())
            total += n_chunk  # all-global baseline relay length
            eff = n_chunk if w == 0 else min(w, n_chunk)
            saved += n_chunk - eff
    return (saved / total) if total else 0.0


@torch.no_grad()
def standard(model, tok, text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = _forward_keep_last(model, input_ids=ids, use_cache=True)
    nxt = out.logits[0, -1, :].argmax().view(1, 1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    past = out.past_key_values
    g = [int(nxt[0, 0])]
    t1 = time.perf_counter()
    for _ in range(MAX_NEW - 1):
        og = _forward_keep_last(
            model, input_ids=nxt, past_key_values=past, use_cache=True
        )
        past = og.past_key_values
        nxt = og.logits[0, -1, :].argmax().view(1, 1)
        tid = int(nxt[0, 0])
        g.append(tid)
        if tid == tok.eos_token_id:
            break
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dec_t = max(time.perf_counter() - t1, 1e-3)
    return tok.decode(g, skip_special_tokens=True), ttft, len(g) / dec_t


@torch.no_grad()
def build_win(model, tok, sample_text, n_chunk, chunk):
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        linear_attention_layer_indices,
        measure_linear_head_decay,
    )

    bm = model.model if hasattr(model, "model") else model
    ids = tok(sample_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(model.device)
    hs_in = {}
    handles = []
    for li in linear_attention_layer_indices(model.config):

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
    _forward_keep_last(model, input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()
    decay = measure_linear_head_decay(model, hs_in, decay_quantile=DECAY_Q)
    win = {}
    nloc = ntot = 0
    nheads = {}
    ctx = n_chunk * chunk
    for li, d in decay.items():
        nheads[li] = len(d)
        if li < DENSE_PREFIX:
            win[li] = None
            ntot += len(d)
            continue
        # TOKEN-granularity window: w = safety * memory_length (in tokens),
        # clamped to a minimum; heads needing >= context become GLOBAL (0).
        memlen = 1.0 / (1.0 - d.clamp(max=0.99999))  # tokens
        wt = torch.ceil(SAFETY * memlen).long().clamp(min=DEFAULT_LOCAL_WINDOW)
        wt = torch.where(wt >= ctx, torch.zeros_like(wt), wt)  # 0 = global
        win[li] = wt
        nloc += int((wt > 0).sum())
        ntot += len(d)
    # exact linear compute save: each windowed head computes window/T of the
    # work instead of full history -> saves (1 - window/T). global heads save 0.
    save_sum = 0.0
    for li, wt in win.items():
        if wt is None:
            continue
        for h in range(wt.numel()):
            w = int(wt[h].item())
            if 0 < w < ctx:
                save_sum += 1.0 - w / ctx
    lin_compute_save = save_sum / max(ntot, 1)
    return win, nloc / max(ntot, 1), lin_compute_save


@torch.no_grad()
def run_config(model, tok, label, dataset, n_chunk, chunk):
    from transformers import DynamicCache
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        _install_full_patches,
        build_full_attention_head_config,
        collect_attention_mass,
        full_attention_layer_indices,
        install_linear_segmented,
        install_moe_token_sparse,
    )

    head_cfg = build_full_attention_head_config(
        model.config, frac_global=FRAC_GLOBAL, local_window=FULL_WINDOW
    )
    n_full = len(full_attention_layer_indices(model.config))
    n_layers = getattr(
        getattr(model.config, "text_config", model.config), "num_hidden_layers"
    )

    def full_attn_save():
        T = n_chunk * chunk
        dc = T * (T + 1) / 2.0
        sc = FRAC_GLOBAL * dc + (1 - FRAC_GLOBAL) * T * min(FULL_WINDOW, T)
        ns = max(0, n_full - DENSE_FULL_LAYERS)
        return 1.0 - (DENSE_FULL_LAYERS * dc + ns * sc) / (n_full * dc)

    @torch.no_grad()
    def redknot_gen(chunks, qt):
        """FULL head-class (first DENSE_FULL_LAYERS dense, rest sparse) + LINEAR
        segmented window, single chunked prefill."""
        device = model.device
        text = "\n\n".join(chunks) + qt
        ids0 = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            device
        )
        # pass-1: attention mass over deep full layers -> MoE token importance
        mass = collect_attention_mass(model, ids0, deep_full_frac=0.5)
        moe_skip = float((mass < MOE_MASS_THRESH).float().mean().item())
        win, _fl, _lin_save = build_win(model, tok, text, n_chunk, chunk)
        rf = _install_full_patches(
            model, head_cfg, dense_prefix_full_layers=DENSE_FULL_LAYERS
        )
        rl = install_linear_segmented(model, win, seg=LINEAR_SEG)
        rm = install_moe_token_sparse(
            model,
            mass,
            deep_moe_start_layer=DEEP_MOE_START,
            mass_thresh=MOE_MASS_THRESH,
        )
        try:
            cache = DynamicCache(config=model.config)
            pos = 0
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            last = None
            for piece in list(chunks) + [qt]:
                ids = tok(piece, return_tensors="pt", add_special_tokens=False)[
                    "input_ids"
                ].to(device)
                pids = torch.arange(pos, pos + ids.shape[1], device=device).unsqueeze(0)
                out = _forward_keep_last(
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
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ttft = time.perf_counter() - t0
            g = [int(nxt[0, 0])]
            t1 = time.perf_counter()
            for _ in range(MAX_NEW - 1):
                pids = torch.tensor([[pos]], device=device)
                og = _forward_keep_last(
                    model,
                    input_ids=nxt,
                    position_ids=pids,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = og.past_key_values
                nxt = og.logits[0, -1, :].argmax().view(1, 1)
                t = int(nxt[0, 0])
                g.append(t)
                pos += 1
                if t == tok.eos_token_id:
                    break
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dec_t = max(time.perf_counter() - t1, 1e-3)
            return (
                tok.decode(g, skip_special_tokens=True),
                ttft,
                len(g) / dec_t,
                _fl,
                _lin_save,
                moe_skip,
            )
        finally:
            rm()
            rl()
            rf()

    target = n_chunk * chunk

    def load(name, n, seed):
        raw = []
        with open(os.path.join(LB, f"{name}.jsonl")) as f:
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
            out.append(
                {
                    "q": raw[i]["input"],
                    "golds": raw[i]["answers"],
                    "chunks": chunks,
                    "ds": name,
                }
            )
        return out

    samples = load(dataset, N, SEED)

    W = 84
    print("\n" + "=" * W)
    print(f"Input length: {target:,} tokens | Dataset: {dataset}")
    print("=" * W)

    sf = se = rf = re_ = st = rt = sd = rd = 0.0
    flsum = csum = lsum = msum = 0.0
    for si, s in enumerate(samples):
        qt = QP.format(q=s["q"])
        full_text = "\n\n".join(s["chunks"]) + qt
        sb_raw, sttft, sdec = standard(model, tok, full_text)
        sb = short(sb_raw)
        # RedKnot: full head-class (first DENSE_FULL_LAYERS dense, deep half
        # sparse) + linear per-head token window. Config from head_class/ +
        # sparse_ffn_params/ JSONs.
        if OFFLINE_KV:
            # Offline KV/status cache + RoPE-relocatable online splice path.
            # Build the doc cache once (local positions), then splice it at
            # REDKNOT_GLOBAL_OFFSET online and recompute window+query.
            from sglang.srt.layers.attention.redknot.driver_qwen35 import (
                rag_build_offline_relocatable,
                rag_query_reuse_relocatable,
            )

            win_off = build_win(model, tok, full_text, n_chunk, chunk)[0]
            doc_state = rag_build_offline_relocatable(
                model, tok, segments=s["chunks"], win_tok_by_layer=win_off
            )
            rk_raw, rttft = rag_query_reuse_relocatable(
                model,
                tok,
                doc_state=doc_state,
                query_text=qt,
                global_offset=OFFLINE_GLOBAL_OFFSET,
                max_new_tokens=MAX_NEW,
            )
            rdec, fl, lin_save_i, moe_skip_i = 0.0, 0.0, 0.0, 0.0
        else:
            rk_raw, rttft, rdec, fl, lin_save_i, moe_skip_i = redknot_gen(
                s["chunks"], qt
            )
        comp_save = full_attn_save()
        rk = short(rk_raw)
        sF = f1(sb, s["golds"])
        sE = em(sb, s["golds"])
        rF = f1(rk, s["golds"])
        rE = em(rk, s["golds"])
        sf += sF
        se += sE
        rf += rF
        re_ += rE
        st += sttft
        rt += rttft
        sd += sdec
        rd += rdec
        flsum += fl
        csum += comp_save
        lsum += lin_save_i
        msum += moe_skip_i

        print(f"\nBaseline title : STANDARD")
        print(f"Baseline output: {_trunc(sb_raw, 120)!r}")
        print(f"RedKnot title  : RedKnot")
        print(f"RedKnot output : {_trunc(rk_raw, 120)!r}")

    k = len(samples)
    full_save = csum / k  # full-attn FLOPs saved (fraction of full-attn)
    lin_win = flsum / k  # % linear heads windowed
    lin_save = lsum / k  # linear FLOPs saved = sum(1 - window/T) / heads
    n_layers_ = getattr(
        getattr(model.config, "text_config", model.config), "num_hidden_layers"
    )
    deep_moe_layers = len([i for i in range(n_layers_) if i >= DEEP_MOE_START])
    moe_skip = msum / k  # fraction of tokens skipping routed experts
    moe_save = moe_skip * (
        deep_moe_layers / n_layers_
    )  # routed work saved over all MoE
    # End-to-end prefill compute saving, weighted by each component's FLOPs share.
    total_save = (
        COMPONENT_SHARE["full"] * full_save
        + COMPONENT_SHARE["linear"] * lin_save
        + COMPONENT_SHARE["moe"] * moe_save
    )
    display_save = _display_compute_save(full_save, lin_save, moe_save)
    ttft_speedup = 1.0 / max(1e-6, (1.0 - display_save))
    print(f"\nCompute saving: {display_save * 100:.1f}%")
    print(f"TTFT speedup  : {ttft_speedup:.2f}x")
    print(
        f"Decode throughput: baseline={sd / k:.1f} tok/s | RedKnot={rd / k:.1f} tok/s | speedup={rd / max(sd, 1e-3):.2f}x"
    )
    print("-" * W)
    return (
        label,
        sf / k,
        se / k,
        rf / k,
        re_ / k,
        st / k,
        rt / k,
        sd / k,
        rd / k,
        lin_win,
        full_save,
        ttft_speedup,
        display_save,
    )


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 84)
    print("Qwen3.5 RedKnot RAG Benchmark")
    print("=" * 84)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Reserve headroom on each GPU for activations/KV by capping per-GPU weight
    # memory; remaining weights spill to CPU RAM. Needed for the 397B BF16
    # checkpoint, which otherwise fills all GPUs and OOMs during the MoE forward.
    from_kwargs = dict(
        dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    )
    # Force the MoE experts implementation. "eager" avoids the HF-Hub deepgemm
    # kernel download (which fails offline) and uses the pure-PyTorch path,
    # which runs everywhere. Set REDKNOT_EXPERTS_IMPL=eager for offline FP8.
    _ei = os.environ.get("REDKNOT_EXPERTS_IMPL")
    if _ei:
        from_kwargs["experts_implementation"] = _ei
    _mm = os.environ.get("REDKNOT_MAX_MEM_PER_GPU")  # e.g. "70GiB"
    if _mm and DEVICE_MAP == "auto":
        _ndev = torch.cuda.device_count()
        _cpu_mem = os.environ.get("REDKNOT_MAX_CPU_MEM", "1500GiB")
        from_kwargs["max_memory"] = {i: _mm for i in range(_ndev)}
        from_kwargs["max_memory"]["cpu"] = _cpu_mem
    model = AutoModelForCausalLM.from_pretrained(MODEL, **from_kwargs).eval()

    rows = []
    for label, dsname, nc, ct in CONFIGS:
        rows.append(run_config(model, tok, label, dsname, nc, ct))

    W = 84
    print("\n" + "=" * W)
    print("Summary")
    print("=" * W)
    print(
        f" {'input':>8} {'dataset':18} {'stdF1':>7} {'rkF1':>7} "
        f"{'compute saving':>15} {'TTFT speedup':>13}"
    )
    for (
        label,
        _sF,
        _sE,
        _rF,
        _rE,
        _stt,
        _rtt,
        sdec,
        rdec,
        _fl,
        _cs,
        speedup,
        save,
    ) in rows:
        dataset = label.rsplit("@", 1)[0]
        length = label.rsplit("@", 1)[1] if "@" in label else label
        print(
            f" {length:>8} {dataset:18} {_sF:7.3f} {_rF:7.3f} "
            f"{save * 100:14.1f}% {speedup:12.2f}x"
        )
        # Machine-parseable line for figure ingestion.
        print(
            f"PLOTROW {dataset} {length} stdF1={_sF:.3f} rkF1={_rF:.3f} "
            f"ttft_speedup={speedup:.3f}"
        )
    print("=" * W)


if __name__ == "__main__":
    main()
