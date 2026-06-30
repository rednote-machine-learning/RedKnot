#!/usr/bin/env python3
"""Analyze MoE token-importance signals for Sparse-FFN (思路1) on Qwen3.5.

思路1: low-importance tokens skip the routed experts (keep only shared expert).
Need a CRITERION to flag low-importance tokens. This script measures, per MoE
layer, candidate signals per token:
  * router top-k weight SUM before normalization (routing confidence): a token
    whose chosen experts have low total prob may be "don't-care".
  * router max prob (peakiness): flat distribution -> token not strongly routed.
  * token hidden-state norm.
We report distributions to see if low-importance tokens are separable (so we can
skip routed experts for them with little quality loss).

Run:
  HF_HUB_OFFLINE=1 PYTHONPATH=python:<venv-sp>:<sys-sp> CUDA_VISIBLE_DEVICES=0,1 \
    .venv_tf5/bin/python test/srt/redknot/analyze_moe_token_importance.py
"""

from __future__ import annotations
import json, os, sys, random
from pathlib import Path
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

MODEL = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B",
)
LB = "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data"
DS = os.environ.get("REDKNOT_DATASETS", "hotpotqa").split(",")[0]
NTOK = int(os.environ.get("REDKNOT_NTOK", "16000"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()
    bm = model.model if hasattr(model, "model") else model

    raw = [json.loads(l) for l in open(os.path.join(LB, f"{DS}.jsonl"))]
    random.Random(0).shuffle(raw)
    toks = tok(raw[0]["context"], add_special_tokens=False)["input_ids"]
    j = 1
    while len(toks) < NTOK:
        toks += tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
        j += 1
    ids = torch.tensor([toks[:NTOK]], device=model.device)
    print(f"analyzing {ids.shape[1]} tokens on {DS}")

    # hook each MoE block's INPUT to capture per-token activation norm + router
    norms = []  # per (layer, token) input hidden-state L2 norm
    handles = []
    for li, layer in enumerate(bm.layers):
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "gate"):
            continue

        def mk(_li):
            def hook(m, args, kwargs):
                hs = (
                    args[0]
                    if args and torch.is_tensor(args[0])
                    else kwargs.get("hidden_states")
                )
                if hs is None:
                    return
                n = hs.reshape(-1, hs.shape[-1]).float().norm(dim=-1)
                norms.append(n.cpu())

            return hook

        handles.append(layer.mlp.register_forward_pre_hook(mk(li), with_kwargs=True))
    model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()

    alln = torch.cat(norms)
    print("=" * 70)
    print(f" MoE input token activation NORM over {alln.numel()} (layer,token) points")
    print("=" * 70)
    print(
        f"   min={alln.min():.2f} p10={alln.quantile(0.1):.2f} p25={alln.quantile(0.25):.2f} "
        f"med={alln.median():.2f} p75={alln.quantile(0.75):.2f} p90={alln.quantile(0.9):.2f} max={alln.max():.2f}"
    )
    # separability: is there a low-norm tail (skip-candidates) distinct from bulk?
    med = alln.median()
    for frac in [0.1, 0.2, 0.3, 0.5]:
        thr = alln.quantile(frac)
        print(
            f"   bottom {int(frac * 100)}% norm threshold = {thr:.2f}  (ratio to median = {thr / med:.2f})"
        )
    # per-layer norm spread (are low-norm tokens consistent across layers?)
    print("-" * 70)
    print(" per-layer norm (med / p10 / spread p90-p10):")
    for k, n in enumerate(norms[:8]):
        print(
            f"   layer#{k}: med={n.median():.2f} p10={n.quantile(0.1):.2f} spread={(n.quantile(0.9) - n.quantile(0.1)):.2f}"
        )
    print("=" * 70)
    print(" If the bottom-X% norm is much smaller than median (ratio<<1), low-norm")
    print(" tokens are separable -> candidates to skip routed experts.")
    print("=" * 70)
    print(f" MoE routing signals over {allconf.numel()} (layer,token) points")
    print("=" * 70)
    print(f" top-k mass (sum of top-{conf and ''}{'k'} probs):")
    print(
        f"   min={allconf.min():.3f} p10={allconf.quantile(0.1):.3f} med={allconf.median():.3f} p90={allconf.quantile(0.9):.3f} max={allconf.max():.3f}"
    )
    print(f" top-1 prob (peak):")
    print(
        f"   min={allpeak.min():.3f} p10={allpeak.quantile(0.1):.3f} med={allpeak.median():.3f} p90={allpeak.quantile(0.9):.3f} max={allpeak.max():.3f}"
    )
    # how separable: fraction of tokens with low top-k mass (candidates to skip)
    for thr in [0.3, 0.5, 0.7]:
        frac = (allconf < thr).float().mean().item()
        print(
            f"   tokens with top-k mass < {thr}: {frac * 100:.1f}%  (candidate skip-routed)"
        )
    print("=" * 70)
    print(" If a meaningful fraction has low top-k mass, those tokens are weakly")
    print(" routed -> skipping their routed experts (keep shared) may be low-loss.")
    print("=" * 70)


if __name__ == "__main__":
    main()
