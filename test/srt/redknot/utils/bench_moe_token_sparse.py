#!/usr/bin/env python3
"""MoE token-sparse on Qwen3.5-35B-A3B: deep-full attention mass -> deep MoE skips
low-importance tokens' routed experts.

Pass 1: collect per-token attention mass over the DEEP half of full layers.
Pass 2: for deep MoE layers (>= deep_moe_start), LOW-mass tokens skip routed
experts (keep shared); high-mass tokens run full MoE. Shallow MoE dense.

Reports F1/EM vs standard + MoE compute saving (fraction of routed-expert token
work avoided) over the deep MoE layers.

Run:
  HF_HUB_OFFLINE=1 PYTHONPATH=python:<venv-sp>:<sys-sp> CUDA_VISIBLE_DEVICES=0,1 \
    .venv_tf5/bin/python test/srt/redknot/bench_moe_token_sparse.py
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
MASS_THRESH = float(os.environ.get("REDKNOT_MOE_MASS_THRESH", "0.2"))
DEEP_MOE_START = int(os.environ.get("REDKNOT_DEEP_MOE_START", "20"))  # layer idx
DEEP_FULL_FRAC = float(os.environ.get("REDKNOT_DEEP_FULL_FRAC", "0.5"))
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
            {
                "q": raw[i]["input"],
                "golds": raw[i]["answers"],
                "ctx": "\n\n".join(ch),
                "ds": name,
            }
        )
    return out


@torch.no_grad()
def gen(model, tok, text):
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
def gen_moe_sparse(model, tok, text, mass):
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        install_moe_token_sparse,
    )

    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    restore = install_moe_token_sparse(
        model, mass, deep_moe_start_layer=DEEP_MOE_START, mass_thresh=MASS_THRESH
    )
    try:
        o = model(input_ids=ids, use_cache=True)
        nx = o.logits[0, -1, :].argmax().view(1, 1)
        p = o.past_key_values
        g = [int(nx[0, 0])]
        # decode: mass mask size won't match single token -> driver runs all (dense)
        for _ in range(MAX_NEW - 1):
            og = model(input_ids=nx, past_key_values=p, use_cache=True)
            p = og.past_key_values
            nx = og.logits[0, -1, :].argmax().view(1, 1)
            t = int(nx[0, 0])
            g.append(t)
            if t == tok.eos_token_id:
                break
        return tok.decode(g, skip_special_tokens=True)
    finally:
        restore()


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sglang.srt.layers.attention.redknot.driver_qwen35 import collect_attention_mass

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()
    samples = []
    for ds in DATASETS:
        samples += load(ds, tok, N)
    W = 92
    n_layers = (
        model.config.num_hidden_layers
        if not hasattr(model.config, "text_config")
        else model.config.text_config.num_hidden_layers
    )
    n_deep_moe = len([i for i in range(n_layers) if i >= DEEP_MOE_START])
    print("=" * W)
    print(
        f" MoE TOKEN-SPARSE — {Path(MODEL).name} | {N_CHUNK}x{CHUNK}={N_CHUNK * CHUNK} tok"
    )
    print(
        f" deep MoE start=L{DEEP_MOE_START} ({n_deep_moe} layers), mass_thresh={MASS_THRESH} (skip routed for mass<thresh)"
    )
    print("=" * W)
    sf = se = rf = re_ = 0.0
    skipsum = 0.0
    for s in samples:
        qt = QP.format(q=s["q"])
        text = s["ctx"] + qt
        sb = short(gen(model, tok, text))
        sF = f1(sb, s["golds"])
        sE = em(sb, s["golds"])
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            model.device
        )
        mass = collect_attention_mass(model, ids, deep_full_frac=DEEP_FULL_FRAC)
        skip_frac = (
            (mass < MASS_THRESH).float().mean().item()
        )  # fraction of tokens skipping routed
        rk = short(gen_moe_sparse(model, tok, text, mass))
        rF = f1(rk, s["golds"])
        rE = em(rk, s["golds"])
        sf += sF
        rf += rF
        se += sE
        re_ += rE
        skipsum += skip_frac
        print(
            f" {s['ds']:14} std={sb[:20]!r} F1={sF:.2f} | moe-sparse={rk[:20]!r} F1={rF:.2f} | skip={skip_frac * 100:.0f}% tokens"
        )
    k = len(samples)
    avg_skip = skipsum / k
    # MoE compute save: deep MoE layers skip routed experts for avg_skip tokens;
    # routed experts dominate MoE compute (shared is small). Approx MoE routed
    # save over deep layers = avg_skip; weighted by deep/total MoE layers.
    moe_routed_save_deep = avg_skip
    frac_deep = n_deep_moe / n_layers
    moe_total_save = moe_routed_save_deep * frac_deep
    print("-" * W)
    print(
        f" ACCURACY  std F1={sf / k:.3f} EM={se / k:.3f} | MoE-sparse F1={rf / k:.3f} EM={re_ / k:.3f} (dF1={rf / k - sf / k:+.3f})"
    )
    print(
        f" MoE COMPUTE  deep-layer low-mass tokens skipping routed experts = {avg_skip * 100:.0f}%"
    )
    print(
        f"   routed-expert work saved on deep MoE layers ~ {moe_routed_save_deep * 100:.0f}%"
    )
    print(
        f"   over ALL MoE layers ({n_deep_moe}/{n_layers} deep) ~ {moe_total_save * 100:.0f}%"
    )
    print("=" * W)


if __name__ == "__main__":
    main()
