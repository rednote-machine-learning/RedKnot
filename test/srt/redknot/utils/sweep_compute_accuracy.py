#!/usr/bin/env python3
"""Sweep RedKnot sparsity strength (full + linear + MoE) -> compute/accuracy curve.

Each config jointly sets the three knobs and reports TOTAL compute saving (FLOPs,
weighted by component share) vs ΔF1. Lets you pick the operating point on the
compute-vs-accuracy curve. Qwen3.5-35B-A3B, multi-dataset.

Configs (conservative -> aggressive):
  C0 baseline: no sparsity
  C1 mild    : full dense6/frac0.4/w4096, linear safety6/minw1024, MoE thr0.1 deep28
  C2 medium  : full dense5/frac0.4/w4096, linear safety4/minw512,  MoE thr0.15 deep24
  C3 aggr.   : full dense4/frac0.4/w2048, linear safety3/minw512,  MoE thr0.25 deep20

Run:
  HF_HUB_OFFLINE=1 PYTHONPATH=python:<venv-sp>:<sys-sp> CUDA_VISIBLE_DEVICES=0,1 \
    .venv_tf5/bin/python test/srt/redknot/sweep_compute_accuracy.py
"""

from __future__ import annotations
import json, os, random, re, string, sys
from collections import Counter
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

MODEL = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B",
)
LB = "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data"
DATASETS = os.environ.get(
    "REDKNOT_DATASETS", "triviaqa,hotpotqa,2wikimqa,multifieldqa_en"
).split(",")
N = int(os.environ.get("REDKNOT_N_SAMPLES", "2"))
N_CHUNK = int(os.environ.get("REDKNOT_N_CHUNK", "4"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "8000"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "24"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
QP = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"
# FLOPs component shares (pure compute, no comm)
SHARE = {"full": 0.06, "linear": 0.42, "moe": 0.35, "other": 0.17}

# (name, dense_full, frac, win, lin_safety, lin_minw, moe_thr, deep_moe)
CONFIGS = [
    ("C1_mild", 6, 0.4, 4096, 6.0, 1024, 0.10, 28),
    ("C2_medium", 5, 0.4, 4096, 4.0, 512, 0.15, 24),
    ("C3_aggr", 4, 0.4, 2048, 3.0, 512, 0.25, 20),
]


def _n(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def f1(p, gs):
    b = 0.0
    for g in gs:
        a, c = _n(p).split(), _n(g).split()
        if not a or not c:
            b = max(b, float(a == c))
            continue
        cm = Counter(a) & Counter(c)
        ns = sum(cm.values())
        if ns == 0:
            continue
        pr, rc = ns / len(a), ns / len(c)
        b = max(b, 2 * pr * rc / (pr + rc))
    return b


def short(t):
    t = (t or "").strip()
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.I)
    t = re.split(r"(assistant|Human:|\*Human|<\|)", t)[0]
    ls = [x.strip() for x in t.splitlines() if x.strip()]
    return (ls[0] if ls else t).strip().strip('"').strip("'")


def load(name, tok, n):
    raw = [
        json.loads(l)
        for l in open(os.path.join(LB, f"{name}.jsonl"))
        if json.loads(l).get("input")
        and json.loads(l).get("context")
        and json.loads(l).get("answers")
    ]
    random.Random(0).shuffle(raw)
    tgt = N_CHUNK * CHUNK
    out = []
    nr = len(raw)
    for i in range(nr):
        if len(out) >= n:
            break
        tk = tok(raw[i]["context"], add_special_tokens=False)["input_ids"]
        j = (i + 1) % nr
        while len(tk) < tgt and j != i:
            tk += tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
            j = (j + 1) % nr
        tk = tk[:tgt]
        if len(tk) < tgt:
            continue
        ch = [
            tok.decode(tk[k : k + CHUNK], skip_special_tokens=True)
            for k in range(0, tgt, CHUNK)
        ]
        out.append(
            {"q": raw[i]["input"], "golds": raw[i]["answers"], "chunks": ch, "ds": name}
        )
    return out


@torch.no_grad()
def std_gen(model, tok, text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    o = model(input_ids=ids, use_cache=True)
    nx = o.logits[0, -1, :].argmax().view(1, 1)
    p = o.past_key_values
    g = [int(nx[0, 0])]
    for _ in range(MAX_NEW - 1):
        og = model(input_ids=nx, past_key_values=p, use_cache=True)
        p = og.past_key_values
        nx = og.logits[0, -1, :].argmax().view(1, 1)
        t = int(nx[0, 0])
        g.append(t)
        if t == tok.eos_token_id:
            break
    return tok.decode(g, skip_special_tokens=True)


@torch.no_grad()
def build_lin_win(model, tok, text, safety, minw):
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        linear_attention_layer_indices,
        measure_linear_head_decay,
    )

    bm = model.model
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    hs = {}
    hh = []
    for li in linear_attention_layer_indices(model.config):

        def mk(_li):
            def hook(m, a, k):
                h = a[0] if a and torch.is_tensor(a[0]) else k.get("hidden_states")
                if h is not None:
                    hs[_li] = h.detach()

            return hook

        hh.append(
            bm.layers[li].linear_attn.register_forward_pre_hook(
                mk(li), with_kwargs=True
            )
        )
    model(input_ids=ids, use_cache=False)
    for h in hh:
        h.remove()
    decay = measure_linear_head_decay(model, hs, decay_quantile=0.95)
    ctx = N_CHUNK * CHUNK
    win = {}
    nloc = ntot = 0
    for li, d in decay.items():
        if li < 5:
            win[li] = None
            ntot += len(d)
            continue
        ml = 1.0 / (1.0 - d.clamp(max=0.99999))
        wt = torch.ceil(safety * ml).long().clamp(min=minw)
        wt = torch.where(wt >= ctx, torch.zeros_like(wt), wt)
        win[li] = wt
        nloc += int((wt > 0).sum())
        ntot += len(d)
    return win, nloc / max(ntot, 1)


@torch.no_grad()
def redknot_gen(model, tok, chunks, qt, cfg, head_cfg):
    from transformers import DynamicCache
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        _install_full_patches,
        install_linear_segmented,
        install_moe_token_sparse,
        collect_attention_mass,
        full_attention_layer_indices,
    )

    name, dense_full, frac, win_, safety, minw, moe_thr, deep_moe = cfg
    device = model.device
    text = "\n\n".join(chunks) + qt
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        device
    )
    # pass-1: attention mass for MoE token sparsity
    mass = collect_attention_mass(model, ids, deep_full_frac=0.5)
    skip_frac = (mass < moe_thr).float().mean().item()
    lin_win, lin_w = build_lin_win(model, tok, text, safety, minw)
    rf = _install_full_patches(model, head_cfg, dense_prefix_full_layers=dense_full)
    rl = install_linear_segmented(model, lin_win, seg=4096)
    rm = install_moe_token_sparse(
        model, mass, deep_moe_start_layer=deep_moe, mass_thresh=moe_thr
    )
    try:
        cache = DynamicCache(config=model.config)
        pos = 0
        last = None
        for piece in list(chunks) + [qt]:
            pid = tok(piece, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ].to(device)
            pids = torch.arange(pos, pos + pid.shape[1], device=device).unsqueeze(0)
            out = model(
                input_ids=pid, position_ids=pids, past_key_values=cache, use_cache=True
            )
            cache = out.past_key_values
            last = out.logits[0, -1, :]
            pos += pid.shape[1]
        nxt = last.argmax().view(1, 1)
        g = [int(nxt[0, 0])]
        for _ in range(MAX_NEW - 1):
            pids = torch.tensor([[pos]], device=device)
            og = model(
                input_ids=nxt, position_ids=pids, past_key_values=cache, use_cache=True
            )
            cache = og.past_key_values
            nxt = og.logits[0, -1, :].argmax().view(1, 1)
            t = int(nxt[0, 0])
            g.append(t)
            pos += 1
            if t == tok.eos_token_id:
                break
        return tok.decode(g, skip_special_tokens=True), lin_w, skip_frac
    finally:
        rm()
        rl()
        rf()


def total_save(dense_full, frac, win, T, lin_w, skip_frac, deep_moe, n_full, n_layers):
    dc = T * (T + 1) / 2.0
    sc = frac * dc + (1 - frac) * T * min(win, T)
    ns = max(0, n_full - dense_full)
    full_save = 1.0 - (dense_full * dc + ns * sc) / (n_full * dc)
    lin_save = lin_w * 0.5
    deep_moe_layers = len([i for i in range(n_layers) if i >= deep_moe])
    moe_save = skip_frac * (deep_moe_layers / n_layers)
    return (
        SHARE["full"] * full_save
        + SHARE["linear"] * lin_save
        + SHARE["moe"] * moe_save,
        full_save,
        lin_save,
        moe_save,
    )


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        build_full_attention_head_config,
        full_attention_layer_indices,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()
    n_full = len(full_attention_layer_indices(model.config))
    n_layers = getattr(
        getattr(model.config, "text_config", model.config), "num_hidden_layers"
    )
    T = N_CHUNK * CHUNK
    samples = []
    for ds in DATASETS:
        samples += load(ds, tok, N)
    W = 96
    print("=" * W)
    print(
        f" COMPUTE-ACCURACY SWEEP — {Path(MODEL).name} | {N_CHUNK}x{CHUNK}={T} tok | N={len(samples)}"
    )
    print("=" * W)
    # baseline F1
    base = []
    for s in samples:
        base.append(
            (
                s,
                f1(
                    short(
                        std_gen(
                            model, tok, "\n\n".join(s["chunks"]) + QP.format(q=s["q"])
                        )
                    ),
                    s["golds"],
                ),
            )
        )
    bf = sum(b for _, b in base) / len(base)
    print(f" baseline F1={bf:.3f}")
    rows = [("C0_baseline", bf, 0.0, 0.0, 0.0, 0.0)]
    for cfg in CONFIGS:
        head_cfg = build_full_attention_head_config(
            model.config, frac_global=cfg[2], local_window=cfg[3]
        )
        rf_sum = 0.0
        lw_sum = 0.0
        sk_sum = 0.0
        for s, _ in base:
            qt = QP.format(q=s["q"])
            rk, lw, sk = redknot_gen(model, tok, s["chunks"], qt, cfg, head_cfg)
            rf_sum += f1(short(rk), s["golds"])
            lw_sum += lw
            sk_sum += sk
        k = len(base)
        rkf = rf_sum / k
        lw = lw_sum / k
        sk = sk_sum / k
        tot, fs, ls, ms = total_save(
            cfg[1], cfg[2], cfg[3], T, lw, sk, cfg[7], n_full, n_layers
        )
        rows.append((cfg[0], rkf, tot, fs, ls, ms))
        print(
            f" {cfg[0]:11} F1={rkf:.3f} dF1={rkf - bf:+.3f} | TOTAL_save={tot * 100:.1f}% (full {fs * 100:.0f}% lin {ls * 100:.0f}% moe {ms * 100:.0f}%)"
        )
    print("=" * W)
    print(" COMPUTE-ACCURACY CURVE:")
    print(f" {'config':12} {'F1':>6} {'dF1':>7} {'total_compute_save':>18}")
    for name, rkf, tot, fs, ls, ms in rows:
        print(f" {name:12} {rkf:6.3f} {rkf - bf:+7.3f} {tot * 100:16.1f}%")
    print("=" * W)


if __name__ == "__main__":
    main()
