#!/usr/bin/env python3
"""Aggressive sparsity sweep on the 3 selected datasets @32K to find the
accuracy/speed sweet spot (target theoretical TTFT 1.5-2x = total FLOPs 33-50%).

Knobs swept (conservative -> aggressive):
  full:   dense_full_layers, frac_global, window
  linear: min_window (smaller -> more save), safety
  MoE:    mass_thresh (higher -> more tokens skip routed), deep_moe_start (lower)

Each config reports avg ΔF1 over {multifieldqa_en, triviaqa, hotpotqa}@32K and
total FLOPs save + theoretical TTFT. Picks the config with max save under an
accuracy floor (ΔF1 >= -0.10).
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
DATASETS = ["multifieldqa_en", "triviaqa", "hotpotqa"]
CTX = 32000
CHUNK = 8000
N = int(os.environ.get("REDKNOT_N_SAMPLES", "2"))
MAX_NEW = 24
SHARE = {"full": 0.06, "linear": 0.42, "moe": 0.35}
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
QP = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"

# (name, dense_full, frac, full_win, lin_safety, lin_minw, moe_thr, deep_moe)
CONFIGS = [
    ("base_lossless", 6, 0.4, 4096, 4.0, 512, 0.2, 20),
    ("A_moe0.5", 6, 0.4, 4096, 4.0, 512, 0.5, 16),
    ("B_lin256+moe0.5", 5, 0.4, 4096, 3.0, 256, 0.5, 16),
    ("C_aggr", 4, 0.4, 2048, 2.0, 256, 0.7, 12),
    ("D_very_aggr", 4, 0.3, 1024, 2.0, 128, 1.0, 8),
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
    out = []
    nr = len(raw)
    nchunk = CTX // CHUNK
    for i in range(nr):
        if len(out) >= n:
            break
        tk = tok(raw[i]["context"], add_special_tokens=False)["input_ids"]
        j = (i + 1) % nr
        while len(tk) < CTX and j != i:
            tk += tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
            j = (j + 1) % nr
        tk = tk[:CTX]
        if len(tk) < CTX:
            continue
        ch = [
            tok.decode(tk[k : k + CHUNK], skip_special_tokens=True)
            for k in range(0, CTX, CHUNK)
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
def build_win(model, tok, text, safety, minw):
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
        if li < 5:
            win[li] = None
            ntot += len(d)
            continue
        ml = 1.0 / (1.0 - d.clamp(max=0.99999))
        wt = torch.ceil(safety * ml).long().clamp(min=minw)
        wt = torch.where(wt >= CTX, torch.zeros_like(wt), wt)
        win[li] = wt
        ntot += len(d)
        for h in range(wt.numel()):
            w = int(wt[h].item())
            if 0 < w < CTX:
                ssum += 1.0 - w / CTX
    return win, ssum / max(ntot, 1)


@torch.no_grad()
def redknot_gen(model, tok, chunks, qt, cfg, head_cfg, n_layers, n_full):
    from transformers import DynamicCache
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        _install_full_patches,
        install_linear_segmented,
        install_moe_token_sparse,
        collect_attention_mass,
    )

    name, dense_full, frac, fwin, safety, minw, moe_thr, deep_moe = cfg
    device = model.device
    text = "\n\n".join(chunks) + qt
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        device
    )
    mass = collect_attention_mass(model, ids, deep_full_frac=0.5)
    moe_skip = float((mass < moe_thr).float().mean().item())
    win, lin_save = build_win(model, tok, text, safety, minw)
    rf = _install_full_patches(model, head_cfg, dense_prefix_full_layers=dense_full)
    rl = install_linear_segmented(model, win, seg=4096)
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
        dc = CTX * (CTX + 1) / 2.0
        sc = frac * dc + (1 - frac) * CTX * min(fwin, CTX)
        full_save = 1.0 - (dense_full * dc + max(0, n_full - dense_full) * sc) / (
            n_full * dc
        )
        deep = len([i for i in range(n_layers) if i >= deep_moe])
        moe_save = moe_skip * (deep / n_layers)
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
    # samples + std baseline
    data = {ds: load(ds, tok, N) for ds in DATASETS}
    base = {}
    for ds in DATASETS:
        bs = 0.0
        for s in data[ds]:
            bs += f1(
                short(
                    std_gen(model, tok, "\n\n".join(s["chunks"]) + QP.format(q=s["q"]))
                ),
                s["golds"],
            )
        base[ds] = bs / len(data[ds])
    bf = sum(base.values()) / len(base)
    W = 90
    print("=" * W)
    print(f" AGGRESSIVE SWEEP @32K (3 datasets) | baseline avg F1={bf:.3f}")
    print("=" * W)
    print(f" {'config':16} {'avgF1':>6} {'dF1':>7} {'TOTsave':>8} {'theoTTFT':>9}")
    results = []
    for cfg in CONFIGS:
        head_cfg = build_full_attention_head_config(
            model.config, frac_global=cfg[2], local_window=cfg[3]
        )
        rf_sum = 0.0
        tot_sum = 0.0
        cnt = 0
        for ds in DATASETS:
            for s in data[ds]:
                rk, tot = redknot_gen(
                    model,
                    tok,
                    s["chunks"],
                    QP.format(q=s["q"]),
                    cfg,
                    head_cfg,
                    n_layers,
                    n_full,
                )
                rf_sum += f1(short(rk), s["golds"])
                tot_sum += tot
                cnt += 1
        avgf = rf_sum / cnt
        tot = tot_sum / cnt
        theo = 1.0 / max(1e-6, 1 - tot)
        results.append((cfg[0], avgf, avgf - bf, tot, theo))
        print(
            f" {cfg[0]:16} {avgf:6.3f} {avgf - bf:+7.3f} {tot * 100:7.1f}% {theo:8.2f}x"
        )
    print("=" * W)
    ok = [r for r in results if r[2] >= -0.10]
    ok.sort(key=lambda r: -r[3])
    if ok:
        b = ok[0]
        print(
            f" SWEET SPOT (ΔF1>=-0.10, max save): {b[0]} | avgF1={b[1]:.3f} dF1={b[2]:+.3f} TOTsave={b[3] * 100:.1f}% theoTTFT={b[4]:.2f}x"
        )
    print("=" * W)


if __name__ == "__main__":
    main()
