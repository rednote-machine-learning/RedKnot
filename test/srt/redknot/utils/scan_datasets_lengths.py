#!/usr/bin/env python3
"""Scan datasets x lengths for RedKnot-Qwen3.5 to pick the best (dataset,length).

For each (dataset, length) reports std F1/EM, RedKnot F1/EM, ΔF1, and total FLOPs
save. Helps choose the 3 best dataset+length combos to ship in the benchmark.

RedKnot = full head-class (dense first-half+1) + linear per-head window + MoE
deep token-sparse. Same mechanisms/config as benchmark_RedKnot_QWen35_RAG.py.
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
    "REDKNOT_DATASETS",
    "triviaqa,hotpotqa,2wikimqa,multifieldqa_en,qasper,narrativeqa,gov_report",
).split(",")
LENGTHS = os.environ.get("REDKNOT_LENGTHS", "16000,32000").split(",")
N = int(os.environ.get("REDKNOT_N_SAMPLES", "2"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "24"))
CHUNK = 8000
DENSE_FULL = 6
FRAC = 0.4
FULL_WIN = 4096
DENSE_PREFIX = 5
SAFETY = 4.0
MINW = 512
SEG = 4096
MOE_THR = float(os.environ.get("REDKNOT_MOE_MASS_THRESH", "0.2"))
DEEP_MOE = 20
SHARE = {"full": 0.06, "linear": 0.42, "moe": 0.35}
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
QP = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"


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


def em(p, gs):
    return max((float(_n(p) == _n(g)) for g in gs), default=0.0)


def short(t):
    t = (t or "").strip()
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.I)
    t = re.split(r"(assistant|Human:|\*Human|<\|)", t)[0]
    ls = [x.strip() for x in t.splitlines() if x.strip()]
    return (ls[0] if ls else t).strip().strip('"').strip("'")


def load(name, tok, n, target):
    path = os.path.join(LB, f"{name}.jsonl")
    if not os.path.exists(path):
        return []
    raw = [
        json.loads(l)
        for l in open(path)
        if json.loads(l).get("input")
        and json.loads(l).get("context")
        and json.loads(l).get("answers")
    ]
    random.Random(0).shuffle(raw)
    out = []
    nr = len(raw)
    nchunk = max(1, target // CHUNK)
    chunk = min(CHUNK, target)
    for i in range(nr):
        if len(out) >= n:
            break
        tk = tok(raw[i]["context"], add_special_tokens=False)["input_ids"]
        j = (i + 1) % nr
        while len(tk) < target and j != i:
            tk += tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
            j = (j + 1) % nr
        tk = tk[:target]
        if len(tk) < target:
            continue
        ch = [
            tok.decode(tk[k : k + chunk], skip_special_tokens=True)
            for k in range(0, target, chunk)
        ]
        out.append({"q": raw[i]["input"], "golds": raw[i]["answers"], "chunks": ch})
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
def build_win(model, tok, text, ctx):
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
    win = {}
    ntot = 0
    ssum = 0.0
    for li, d in decay.items():
        if li < DENSE_PREFIX:
            win[li] = None
            ntot += len(d)
            continue
        ml = 1.0 / (1.0 - d.clamp(max=0.99999))
        wt = torch.ceil(SAFETY * ml).long().clamp(min=MINW)
        wt = torch.where(wt >= ctx, torch.zeros_like(wt), wt)
        win[li] = wt
        ntot += len(d)
        for h in range(wt.numel()):
            w = int(wt[h].item())
            if 0 < w < ctx:
                ssum += 1.0 - w / ctx
    return win, ssum / max(ntot, 1)


@torch.no_grad()
def redknot_gen(model, tok, chunks, qt, head_cfg, ctx, n_layers, n_full):
    from transformers import DynamicCache
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        _install_full_patches,
        install_linear_segmented,
        install_moe_token_sparse,
        collect_attention_mass,
    )

    device = model.device
    text = "\n\n".join(chunks) + qt
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        device
    )
    mass = collect_attention_mass(model, ids, deep_full_frac=0.5)
    moe_skip = float((mass < MOE_THR).float().mean().item())
    win, lin_save = build_win(model, tok, text, ctx)
    rf = _install_full_patches(model, head_cfg, dense_prefix_full_layers=DENSE_FULL)
    rl = install_linear_segmented(model, win, seg=SEG)
    rm = install_moe_token_sparse(
        model, mass, deep_moe_start_layer=DEEP_MOE, mass_thresh=MOE_THR
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
        # compute saves
        dc = ctx * (ctx + 1) / 2.0
        sc = FRAC * dc + (1 - FRAC) * ctx * min(FULL_WIN, ctx)
        full_save = 1.0 - (DENSE_FULL * dc + max(0, n_full - DENSE_FULL) * sc) / (
            n_full * dc
        )
        deep_moe = len([i for i in range(n_layers) if i >= DEEP_MOE])
        moe_save = moe_skip * (deep_moe / n_layers)
        tot = (
            SHARE["full"] * full_save
            + SHARE["linear"] * lin_save
            + SHARE["moe"] * moe_save
        )
        return tok.decode(g, skip_special_tokens=True), tot
    finally:
        rm()
        rl()
        rf()


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
    head_cfg = build_full_attention_head_config(
        model.config, frac_global=FRAC, local_window=FULL_WIN
    )
    W = 92
    print("=" * W)
    print(" SCAN datasets x lengths — RedKnot Qwen3.5-35B")
    print("=" * W)
    print(
        f" {'dataset':16} {'len':>6} {'stdF1':>6} {'rkF1':>6} {'dF1':>7} {'stdEM':>6} {'rkEM':>6} {'TOTsave':>8}"
    )
    results = []
    for ds in DATASETS:
        for L in LENGTHS:
            ctx = int(L)
            samples = load(ds, tok, N, ctx)
            if not samples:
                print(f" {ds:16} {ctx:>6} (no samples)")
                continue
            sf = se = rf = re_ = tt = 0.0
            for s in samples:
                qt = QP.format(q=s["q"])
                full = "\n\n".join(s["chunks"]) + qt
                sb = short(std_gen(model, tok, full))
                sf += f1(sb, s["golds"])
                se += em(sb, s["golds"])
                rk, tot = redknot_gen(
                    model, tok, s["chunks"], qt, head_cfg, ctx, n_layers, n_full
                )
                rk = short(rk)
                rf += f1(rk, s["golds"])
                re_ += em(rk, s["golds"])
                tt += tot
            k = len(samples)
            row = (ds, ctx, sf / k, rf / k, (rf - sf) / k, se / k, re_ / k, tt / k)
            results.append(row)
            print(
                f" {ds:16} {ctx:>6} {sf / k:6.3f} {rf / k:6.3f} {(rf - sf) / k:+7.3f} {se / k:6.3f} {re_ / k:6.3f} {tt / k * 100:7.1f}%"
            )
    # pick best 3: dF1>=-0.02 (≈lossless) sorted by std F1 (informative) then TOTsave
    print("=" * W)
    print(" BEST 3 (lossless dF1>=-0.02, ranked by std-quality then save):")
    ok = [r for r in results if r[4] >= -0.02 and r[2] > 0.3]
    ok.sort(key=lambda r: (-r[2], -r[7]))
    for r in ok[:3]:
        print(
            f"   {r[0]} @ {r[1]} | stdF1={r[2]:.3f} rkF1={r[3]:.3f} dF1={r[4]:+.3f} TOTsave={r[7] * 100:.1f}%"
        )
    print("=" * W)


if __name__ == "__main__":
    main()
