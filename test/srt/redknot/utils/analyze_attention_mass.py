#!/usr/bin/env python3
"""Analyze attention-mass token importance in Qwen3.5 full-attention layers.

For Sparse-MoE (思路: deep MoE skips low-importance tokens' routed experts) we
need a token-importance signal WITH real spread. router/norm had none. This
measures attention mass = how much each KEY token is attended to (sum of
softmax(QK^T) over queries), per full-attention layer. If mass has a clear
high/low split (a few tokens get most attention, many get little), it's a usable
criterion. We sample query rows to avoid the O(L^2) full matrix.

Run:
  HF_HUB_OFFLINE=1 PYTHONPATH=python:<venv-sp>:<sys-sp> CUDA_VISIBLE_DEVICES=0,1 \
    .venv_tf5/bin/python test/srt/redknot/analyze_attention_mass.py
"""

from __future__ import annotations
import json, os, random, sys
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
N_QSAMPLE = int(os.environ.get("REDKNOT_QSAMPLE", "512"))  # sampled query rows
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        apply_rotary_pos_emb,
    )
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        full_attention_layer_indices,
    )

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
    T = ids.shape[1]
    print(f"analyzing {T} tokens on {DS}, sampling {N_QSAMPLE} query rows/layer")

    full_idx = sorted(full_attention_layer_indices(model.config))
    mass_stats = {}  # layer -> per-key-token mass (sampled queries averaged)

    handles = []
    for li in full_idx:
        attn = bm.layers[li].self_attn

        def mk(_li, _attn):
            def hook(m, args, kwargs):
                hs = (
                    args[0]
                    if args and torch.is_tensor(args[0])
                    else kwargs.get("hidden_states")
                )
                pe = kwargs.get("position_embeddings")
                if hs is None or pe is None:
                    return
                B, Tt, _ = hs.shape
                hd = _attn.head_dim
                q_raw = _attn.q_proj(hs).view(B, Tt, -1, hd * 2)
                q, _gate = torch.chunk(q_raw, 2, dim=-1)
                q = _attn.q_norm(q).transpose(1, 2)  # [B,Hq,T,hd]
                k = _attn.k_norm(_attn.k_proj(hs).view(B, Tt, -1, hd)).transpose(
                    1, 2
                )  # [B,Hkv,T,hd]
                cos, sin = pe
                q, k = apply_rotary_pos_emb(q, k, cos.to(q.device), sin.to(q.device))
                Hq = q.shape[1]
                Hkv = k.shape[1]
                rep = Hq // Hkv
                k = k.repeat_interleave(rep, dim=1)  # [B,Hq,T,hd]
                # sample query rows (causal: each query attends [0..i])
                qi = torch.randperm(Tt, device=q.device)[:N_QSAMPLE].sort().values
                qs = q[:, :, qi, :]  # [B,Hq,S,hd]
                scale = hd**-0.5
                scores = torch.matmul(qs, k.transpose(-1, -2)) * scale  # [B,Hq,S,T]
                # causal mask: query at position qi[s] attends keys <= qi[s]
                keypos = torch.arange(Tt, device=q.device)
                mask = keypos[None, :] > qi[:, None]  # [S,T] True=masked
                scores = scores.masked_fill(mask[None, None], float("-inf"))
                probs = F.softmax(scores.float(), dim=-1)  # [B,Hq,S,T]
                mass = probs.sum(dim=(0, 1, 2))  # [T] total attention received
                mass_stats[_li] = mass.cpu()

            return hook

        handles.append(attn.register_forward_pre_hook(mk(li, attn), with_kwargs=True))
    model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()

    print("=" * 74)
    print(" Attention MASS per key token (sampled queries), per full layer")
    print("=" * 74)
    allm = []
    for li in full_idx:
        if li not in mass_stats:
            continue
        m = mass_stats[li]
        m = m / (m.sum() + 1e-9) * m.numel()  # normalize: mean=1
        allm.append(m)
        print(
            f" L{li:2d}: mass(norm mean=1) min={m.min():.3f} p10={m.quantile(0.1):.3f} "
            f"med={m.median():.3f} p90={m.quantile(0.9):.3f} max={m.max():.2f} | "
            f"top1%share={m.topk(max(1, m.numel() // 100)).values.sum() / m.sum() * 100:.0f}%"
        )
    A = torch.cat(allm)
    print("-" * 74)
    print(
        f" ALL full layers: min={A.min():.3f} p10={A.quantile(0.1):.3f} med={A.median():.3f} "
        f"p90={A.quantile(0.9):.3f} max={A.max():.2f}"
    )
    for thr in [0.2, 0.5]:
        frac = (A < thr).float().mean().item()
        print(
            f"   tokens with mass < {thr}x mean: {frac * 100:.1f}%  (low-importance candidates)"
        )
    print("=" * 74)
    print(" If a meaningful fraction has mass << mean (and a few tokens dominate),")
    print(" attention mass IS a usable importance signal for MoE token sparsity.")
    print("=" * 74)


if __name__ == "__main__":
    main()
