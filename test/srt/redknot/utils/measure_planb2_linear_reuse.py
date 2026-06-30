#!/usr/bin/env python3
"""Plan-2 measurement: reuse OFFLINE linear-attn outputs for chunk>=2 (no online
linear recompute) vs Plan-1 (linear recomputed online). Quantify the accuracy
loss of skipping linear recompute, on Qwen3.5-35B-A3B, 4-chunk RAG.

Setup:
  OFFLINE pass: run the WHOLE context once with EXACT full attention, capturing,
    per linear layer, its per-token OUTPUT hidden states (this is the offline
    linear product Plan-2 would store).
  ONLINE Plan-2: prefill chunk-by-chunk with SPARSE full attention; for chunk>=2,
    each linear layer does NOT recompute — it returns the OFFLINE outputs for
    those token positions (state reuse / accumulation, no compute). chunk 1 uses
    offline too (exact prefix).
  REF: native full-recompute.
  PLAN1: chunk-by-chunk, sparse full + linear RECOMPUTED online (already
    validated ~lossless beyond full sparsity).

Report next-token match + logit diff vs REF for PLAN1 and PLAN2. The gap PLAN2 -
PLAN1 is the cost of NOT recomputing linear (the upstream-sparse-full coupling).

Run:
  HF_ENDPOINT=https://huggingface.co PYTHONPATH=python:<venv-sp>:<sys-sp> \
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv_tf5/bin/python \
    test/srt/redknot/measure_planb2_linear_reuse.py
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
N = int(os.environ.get("REDKNOT_N_SAMPLES", "3"))
SEED = int(os.environ.get("REDKNOT_SEED", "0"))
N_CHUNK = int(os.environ.get("REDKNOT_N_CHUNK", "4"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "4000"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
FRAC = float(os.environ.get("REDKNOT_FRAC_GLOBAL", "0.40"))
WINDOW = int(os.environ.get("REDKNOT_WINDOW_FIXED", "4096"))
QP = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"


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
        out.append({"q": raw[i]["input"], "toks": toks})
    return out


def linear_layer_indices(config):
    tc = getattr(config, "text_config", config)
    return [i for i, t in enumerate(tc.layer_types) if t == "linear_attention"]


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        _install_full_patches,
        build_full_attention_head_config,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()
    bm = model.model if hasattr(model, "model") else model
    head_cfg = build_full_attention_head_config(
        model.config, frac_global=FRAC, local_window=WINDOW
    )
    lin_idx = linear_layer_indices(model.config)

    samples = []
    for ds in DATASETS:
        for s in load_4chunk(ds, tok, N, SEED):
            samples.append(s)
    print(f"samples={len(samples)} ctx~{N_CHUNK * CHUNK} frac={FRAC} window={WINDOW}")

    def run_planx(toks, qtext, reuse_offline_linear):
        """chunk-by-chunk sparse-full prefill. If reuse_offline_linear: for
        chunk>=2 linear layers return captured OFFLINE outputs (no recompute)."""
        device = model.device
        # split into chunks of CHUNK tokens + query
        qids = tok(qtext, add_special_tokens=False)["input_ids"]
        chunk_tok = [toks[k : k + CHUNK] for k in range(0, len(toks), CHUNK)] + [qids]

        # OFFLINE capture (exact full) of linear-layer outputs per position.
        offline_out = {}  # layer_idx -> [1, L, H] full-seq output
        if reuse_offline_linear:
            cap = {}
            hooks = []
            for li in lin_idx:

                def mk(_li):
                    def hook(mod, inp, out):
                        cap[_li] = out.detach()

                    return hook

                hooks.append(bm.layers[li].linear_attn.register_forward_hook(mk(li)))
            full = torch.tensor([toks + qids], device=device)
            model(input_ids=full, use_cache=False)
            for h in hooks:
                h.remove()
            offline_out = cap

        # ONLINE chunk-by-chunk sparse full.
        restore_full = _install_full_patches(model, head_cfg)
        lin_saved = {}
        # patch linear layers to return offline outputs for chunk>=2
        state = {"pos": 0, "chunk_id": 0}
        if reuse_offline_linear:
            for li in lin_idx:
                m = bm.layers[li].linear_attn
                lin_saved[li] = m.forward

                def mk(_li, _orig):
                    def fwd(hidden_states, **kw):
                        T = hidden_states.shape[1]
                        if state["chunk_id"] >= 1 and _li in offline_out:
                            # reuse offline output slice for these positions
                            seg = offline_out[_li][
                                :, state["pos"] : state["pos"] + T, :
                            ]
                            return seg.to(hidden_states.dtype)
                        return _orig(hidden_states=hidden_states, **kw)

                    return fwd

                m.forward = mk(li, lin_saved[li])

        try:
            cache = DynamicCache(config=model.config)
            pos = 0
            last = None
            for ci, piece_tok in enumerate(chunk_tok):
                state["chunk_id"] = ci
                state["pos"] = pos
                ids = torch.tensor([piece_tok], device=device)
                pids = torch.arange(pos, pos + ids.shape[1], device=device).unsqueeze(0)
                out = model(
                    input_ids=ids,
                    position_ids=pids,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = out.past_key_values
                last = out.logits[0, -1, :].float()
                pos += ids.shape[1]
            return last
        finally:
            restore_full()
            for li, f in lin_saved.items():
                bm.layers[li].linear_attn.forward = f

    W = 80
    print("=" * W)
    print(f" PLAN-1 (linear recompute) vs PLAN-2 (linear offline reuse, chunk>=2)")
    print("=" * W)
    p1m, p2m, p1d, p2d = [], [], [], []
    for s in samples:
        rids = torch.tensor(
            [
                s["toks"]
                + tok(QP.format(q=s["q"]), add_special_tokens=False)["input_ids"]
            ],
            device=model.device,
        )
        ref = model(input_ids=rids, use_cache=False).logits[0, -1, :].float()
        qtext = QP.format(q=s["q"])
        l1 = run_planx(s["toks"], qtext, reuse_offline_linear=False)
        l2 = run_planx(s["toks"], qtext, reuse_offline_linear=True)
        p1m.append(int(l1.argmax() == ref.argmax()))
        p1d.append((l1 - ref).abs().max().item())
        p2m.append(int(l2.argmax() == ref.argmax()))
        p2d.append((l2 - ref).abs().max().item())
        print(
            f"  sample: P1 match={bool(p1m[-1])} diff={p1d[-1]:.2f} | "
            f"P2 match={bool(p2m[-1])} diff={p2d[-1]:.2f}"
        )
    print("-" * W)
    print(
        f" PLAN-1 (recompute) : match={sum(p1m) / len(p1m):.2f} diff={sum(p1d) / len(p1d):.3f}"
    )
    print(
        f" PLAN-2 (reuse)     : match={sum(p2m) / len(p2m):.2f} diff={sum(p2d) / len(p2d):.3f}"
    )
    print("=" * W)
    print(
        " Gap (P2 worse) = cost of NOT recomputing linear (upstream-sparse coupling)."
    )
    print("=" * W)


if __name__ == "__main__":
    main()
