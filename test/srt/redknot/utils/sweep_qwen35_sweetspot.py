#!/usr/bin/env python3
"""Find Qwen3.5 full-attention sweet spot + test the depth hypothesis.

Two questions, one model load (Qwen3.5-35B-A3B), 4-chunk x 4K RAG samples:

PART 1 — sweet spot:  grid over frac_global x window. For each config report
  next-token agreement vs native + mean logit diff (decisive, fast signal).

PART 2 — depth hypothesis:  does "shallow full layers -> local, deep full layers
  -> global" help? At a fixed budget (frac_global), compare global_assign in
  {random, deep, shallow}. If "deep" >> "shallow", the RedKnot depth property
  carries over to Qwen3.5's full layers.

Run:
  HF_ENDPOINT=https://huggingface.co PYTHONPATH=python:<venv-sp>:<sys-sp> \
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv_tf5/bin/python \
    test/srt/redknot/sweep_qwen35_sweetspot.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

MODEL = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B",
)
DATASETS = os.environ.get(
    "REDKNOT_DATASETS", "hotpotqa,2wikimqa,multifieldqa_en"
).split(",")
LB = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
N = int(os.environ.get("REDKNOT_N_SAMPLES", "2"))  # per dataset
SEED = int(os.environ.get("REDKNOT_SEED", "0"))
N_CHUNK = int(os.environ.get("REDKNOT_N_CHUNK", "4"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "4000"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
QP = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"

FRACS = [float(x) for x in os.environ.get("REDKNOT_FRACS", "0.10,0.25,0.50").split(",")]
WINS = [int(x) for x in os.environ.get("REDKNOT_WINS", "4096,8192,16384").split(",")]


def load_4chunk(name, tok, n, seed):
    raw = []
    with open(os.path.join(LB, f"{name}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("input") and r.get("context") and r.get("answers"):
                raw.append(r)
    random.Random(seed).shuffle(raw)
    target = N_CHUNK * CHUNK
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
            tok.decode(toks[k : k + CHUNK], skip_special_tokens=True)
            for k in range(0, target, CHUNK)
        ]
        out.append({"q": raw[i]["input"], "chunks": chunks})
    return out


@torch.no_grad()
def eval_cfg(model, tok, samples, refs, build, **kw):
    """Return (match_rate, mean_logit_diff) of chunked driver vs native refs."""
    from sglang.srt.layers.attention.redknot.driver_qwen35 import _install_full_patches
    from transformers import DynamicCache

    head_cfg = build(model.config, **kw)
    matches, diffs = [], []
    restore = _install_full_patches(model, head_cfg)
    try:
        for s, ref in zip(samples, refs):
            cache = DynamicCache(config=model.config)
            pos = 0
            last = None
            for piece in s["chunks"] + [s["qt"]]:
                ids = tok(piece, return_tensors="pt", add_special_tokens=False)[
                    "input_ids"
                ].to(model.device)
                pids = torch.arange(
                    pos, pos + ids.shape[1], device=model.device
                ).unsqueeze(0)
                out = model(
                    input_ids=ids,
                    position_ids=pids,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = out.past_key_values
                last = out.logits[0, -1, :].float()
                pos += ids.shape[1]
            matches.append(int(last.argmax() == ref.argmax()))
            diffs.append((last - ref).abs().max().item())
    finally:
        restore()
    return sum(matches) / len(matches), sum(diffs) / len(diffs)


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        build_full_attention_head_config,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()

    # Build samples + native reference next-token logits once.
    samples, refs = [], []
    for ds in DATASETS:
        for s in load_4chunk(ds, tok, N, SEED):
            s["qt"] = QP.format(q=s["q"])
            rids = tok(
                "\n\n".join(s["chunks"]) + s["qt"],
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].to(model.device)
            refs.append(model(input_ids=rids, use_cache=False).logits[0, -1, :].float())
            samples.append(s)
    print(f"samples={len(samples)} (4chunk x {CHUNK})  context~{N_CHUNK * CHUNK} tok")

    W = 78
    print("=" * W)
    print(" PART 1 — sweet spot: frac_global x window (match rate / mean logit diff)")
    print("=" * W)
    _hdr = "frac\\win"
    print(f" {_hdr:>10} " + " ".join(f"{w:>12}" for w in WINS))
    for fr in FRACS:
        cells = []
        for w in WINS:
            m, d = eval_cfg(
                model,
                tok,
                samples,
                refs,
                build_full_attention_head_config,
                frac_global=fr,
                local_window=w,
                global_assign="random",
            )
            cells.append(f"{m:.2f}/{d:5.2f}")
        print(f" {fr:>10.2f} " + " ".join(f"{c:>12}" for c in cells))

    print("\n" + "=" * W)
    print(" PART 2 — depth hypothesis: global_assign at frac=0.25, window=8192")
    print("  (deep=deepest full layers global; shallow=shallowest; random=spread)")
    print("=" * W)
    for ga in ["deep", "shallow", "random"]:
        m, d = eval_cfg(
            model,
            tok,
            samples,
            refs,
            build_full_attention_head_config,
            frac_global=0.25,
            local_window=8192,
            global_assign=ga,
        )
        print(f"   {ga:8} : match={m:.2f}  mean_logit_diff={d:.3f}")
    print("=" * W)
    print(" If deep >> shallow -> RedKnot depth property holds on Qwen3.5 full layers.")
    print("=" * W)


if __name__ == "__main__":
    main()
