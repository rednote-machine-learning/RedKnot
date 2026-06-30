#!/usr/bin/env python3
"""Full-attention RedKnot sparsity on Qwen3.5-35B-A3B.

Config: first 4 FULL-attention layers stay DENSE (exact); the remaining full
layers run RedKnot head-class (global/local) sparsity at the verified sweet spot
(frac_global, window). Linear/MoE unchanged. Reports accuracy vs standard +
analytic FULL-ATTENTION FLOPs saving.

FLOPs saving (full-attn prefill, per full layer):
  dense  ~ Hq * T^2 / 2
  sparse ~ Hq * [frac_global * T^2/2 + (1-frac_global) * T*window]
Only the sparsified full layers save; the 4 dense full layers save 0.

Run:
  HF_HUB_OFFLINE=1 PYTHONPATH=python:<venv-sp>:<sys-sp> CUDA_VISIBLE_DEVICES=0,1 \
    .venv_tf5/bin/python test/srt/redknot/bench_full_attn_sparse.py
"""

from __future__ import annotations
import json, os, random, re, string, sys, time
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
FRAC = float(os.environ.get("REDKNOT_FRAC_GLOBAL", "0.40"))
WINDOW = int(os.environ.get("REDKNOT_WINDOW_FIXED", "4096"))
DENSE_FULL = int(os.environ.get("REDKNOT_DENSE_FULL_LAYERS", "4"))
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
def gen_chunked(model, tok, chunks, qt, head_cfg, dense_full):
    """Single online prefill (chunked, full head-class sparse w/ dense prefix)."""
    from transformers import DynamicCache
    from sglang.srt.layers.attention.redknot.driver_qwen35 import _install_full_patches

    device = model.device
    restore = _install_full_patches(
        model, head_cfg, dense_prefix_full_layers=dense_full
    )
    try:
        cache = DynamicCache(config=model.config)
        pos = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        last = None
        for piece in list(chunks) + [qt]:
            ids = tok(piece, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ].to(device)
            pids = torch.arange(pos, pos + ids.shape[1], device=device).unsqueeze(0)
            out = model(
                input_ids=ids, position_ids=pids, past_key_values=cache, use_cache=True
            )
            cache = out.past_key_values
            last = out.logits[0, -1, :]
            pos += ids.shape[1]
        nxt = last.argmax().view(1, 1)
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t0
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
        return tok.decode(g, skip_special_tokens=True), ttft
    finally:
        restore()


@torch.no_grad()
def std(model, tok, text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    o = model(input_ids=ids, use_cache=True)
    nx = o.logits[0, -1, :].argmax().view(1, 1)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
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
    return tok.decode(g, skip_special_tokens=True), ttft


def full_attn_flops_save(n_full, dense_full, frac, window, T):
    """Average full-attn FLOPs saving across ALL full layers (dense prefix saves 0)."""
    dense_cost = T * (T + 1) / 2.0
    sparse_cost = frac * dense_cost + (1 - frac) * T * min(window, T)
    n_sparse = max(0, n_full - dense_full)
    total_dense = n_full * dense_cost
    total_red = dense_full * dense_cost + n_sparse * sparse_cost
    return 1.0 - total_red / total_dense


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
    head_cfg = build_full_attention_head_config(
        model.config, frac_global=FRAC, local_window=WINDOW
    )
    T = N_CHUNK * CHUNK
    samples = []
    for ds in DATASETS:
        samples += load(ds, tok, N)
    W = 92
    print("=" * W)
    print(
        f" FULL-ATTN RedKnot: first {DENSE_FULL} full layers DENSE, rest head-class (frac_g={FRAC} win={WINDOW})"
    )
    print(f" {Path(MODEL).name} | {N_CHUNK}x{CHUNK}={T} tok | full layers={n_full}")
    print("=" * W)
    sf = se = rf = re_ = st = rt = 0.0
    for s in samples:
        qt = QP.format(q=s["q"])
        full = "\n\n".join(s["chunks"]) + qt
        sb, sttft = std(model, tok, full)
        sb = short(sb)
        rk, rttft = gen_chunked(model, tok, s["chunks"], qt, head_cfg, DENSE_FULL)
        rk = short(rk)
        sF = f1(sb, s["golds"])
        rF = f1(rk, s["golds"])
        sf += sF
        rf += rF
        se += em(sb, s["golds"])
        re_ += em(rk, s["golds"])
        st += sttft
        rt += rttft
        print(f" {s['ds']:14} std={sb[:20]!r} F1={sF:.2f} | rk={rk[:20]!r} F1={rF:.2f}")
    k = len(samples)
    save = full_attn_flops_save(n_full, DENSE_FULL, FRAC, WINDOW, T)
    print("-" * W)
    print(
        f" ACCURACY  std F1={sf / k:.3f} EM={se / k:.3f} | RedKnot F1={rf / k:.3f} EM={re_ / k:.3f} (dF1={rf / k - sf / k:+.3f})"
    )
    print(f" TTFT      std={st / k:.2f}s RedKnot={rt / k:.2f}s")
    print(
        f" FULL-ATTN FLOPs SAVE = {save * 100:.1f}%  (over all {n_full} full layers; first {DENSE_FULL} dense)"
    )
    print(
        f"   sparsified full layers: {n_full - DENSE_FULL}/{n_full}, each saves "
        f"{(1 - (FRAC + (1 - FRAC) * min(WINDOW, T) / (T / 2.0))) * 100:.0f}% of its attn FLOPs"
    )
    print("=" * W)


if __name__ == "__main__":
    main()
