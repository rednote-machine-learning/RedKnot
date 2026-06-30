#!/usr/bin/env python3
"""Real probe: measure Qwen3.5-35B-A3B full-attention head locality.

For each full-attention layer:
  1. Capture hidden states via forward hook
  2. Manually compute Q/K/V with real weights from the model
  3. Apply RoPE, compute softmax attention per sampled query
  4. Measure near-window mass fraction → classify global vs local

Run:
  .venv_tf5/bin/python test/srt/redknot/utils/probe_qwen35_full_attn_heads.py \
      --out test/srt/redknot/head_class/qwen3.5-35B-A3B_full_attn_head_map.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

MODEL_PATH = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B",
)
DEFAULT_TEXT = (
    (
        "The transformer architecture has revolutionized natural language processing. "
        "Machine learning models have achieved remarkable results across many domains. "
        "Attention mechanisms allow models to focus on relevant parts of the input. "
        "Deep learning has transformed computer vision and speech recognition. "
        "Natural language understanding has made significant progress in recent years. "
        "Large language models demonstrate impressive capabilities in reasoning. "
        "The field of artificial intelligence continues to evolve rapidly. "
        "Neural networks have become the foundation of modern AI systems. "
    )
    * 50
)  # ~2200 tokens, longer than window=4096? No, 50 reps * ~44 tokens = 2200 tokens. Need more.
DEFAULT_OUT = (
    REPO
    / "test"
    / "srt"
    / "redknot"
    / "head_class"
    / "qwen3.5-35B-A3B_full_attn_head_map.json"
)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--qsample", type=int, default=256)
    parser.add_argument("--coverage", type=float, default=0.99)
    parser.add_argument("--local-mass-thresh", type=float, default=0.80)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--window-safety", type=float, default=1.5)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        full_attention_layer_indices,
        linear_attention_layer_indices,
    )

    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "70GiB", 1: "70GiB"},
        trust_remote_code=True,
    ).eval()

    tc = getattr(model.config, "text_config", model.config)
    n_q_heads = int(tc.num_attention_heads)
    n_kv_heads = int(tc.num_key_value_heads)
    n_rep = n_q_heads // n_kv_heads
    head_dim = int(getattr(tc, "head_dim", tc.hidden_size // n_q_heads))
    full_layers = full_attention_layer_indices(model.config)
    linear_layers = linear_attention_layer_indices(model.config)
    n_layers = tc.num_hidden_layers

    print(
        f"  Layers: {n_layers}, Full: {len(full_layers)}, Linear: {len(linear_layers)}"
    )
    print(f"  Heads: Q={n_q_heads}, KV={n_kv_heads}, rep={n_rep}, dim={head_dim}")

    ids = tok(args.text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    T = ids.shape[1]
    print(f"  Tokens: {T}")

    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            apply_rotary_pos_emb,
        )
    except Exception:
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    bm = model.model if hasattr(model, "model") else model
    scale = head_dim**-0.5

    # ---- Step 1: capture hidden states + position_embeddings for each full-attn layer ----
    hidden_by_layer: dict[int, torch.Tensor] = {}
    pe_by_layer: dict[int, tuple] = {}
    handles = []

    for li in full_layers:

        def mk(_li):
            def hook(m, args, kwargs):
                hs = (
                    args[0]
                    if args and torch.is_tensor(args[0])
                    else kwargs.get("hidden_states")
                )
                pe = kwargs.get("position_embeddings")
                if hs is not None:
                    hidden_by_layer[_li] = hs.detach()
                if pe is not None:
                    pe_by_layer[_li] = (pe[0].detach(), pe[1].detach())

            return hook

        handles.append(
            bm.layers[li].self_attn.register_forward_pre_hook(mk(li), with_kwargs=True)
        )

    print("Running forward pass to capture hidden states ...")
    model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()

    print(f"  Captured {len(hidden_by_layer)} full-attention layers")

    # ---- Step 2: for each full-attn layer, compute attention and classify heads ----
    n_rows = min(args.qsample, T)
    qi = torch.linspace(0, T - 1, steps=n_rows, device=model.device).long().unique()
    min_qpos = max(8, int(0.25 * T))
    qsel = qi[qi >= min_qpos]
    if qsel.numel() == 0:
        qsel = qi[-max(1, n_rows // 4) :]

    full_head_map: dict[int, list[bool]] = {}
    print(
        f"\nFull-attention head locality (window={args.window_size}, thresh={args.local_mass_thresh}):"
    )
    print(f"{'Layer':>5s} {'global':>6s} {'local':>6s} {'avg_winmass':>12s}")

    for li in full_layers:
        hidden = hidden_by_layer[li]
        attn_mod = bm.layers[li].self_attn
        b, t, d = hidden.shape

        # Qwen3.5: q_proj -> [Q | gate], split
        q_raw = attn_mod.q_proj(hidden).view(b, t, -1, head_dim * 2)
        query_states, _gate = torch.chunk(q_raw, 2, dim=-1)
        query_states = attn_mod.q_norm(query_states.view(b, t, -1, head_dim)).transpose(
            1, 2
        )
        key_states = attn_mod.k_norm(
            attn_mod.k_proj(hidden).view(b, t, -1, head_dim)
        ).transpose(1, 2)

        # RoPE
        pe = pe_by_layer.get(li)
        if pe is not None:
            cos, sin = pe
            cos = cos.to(query_states.device)
            sin = sin.to(query_states.device)
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )

        key_states = key_states.repeat_interleave(n_rep, dim=1)

        head_win_mass_sum = torch.zeros(n_q_heads, device=model.device)
        head_dist_sum = torch.zeros(n_q_heads, device=model.device)
        head_cnt = 0

        # Batch: sample queries at once, compute all attention scores
        qs = query_states[0, :, qsel.to(query_states.device)].float()  # [H, S_rows, D]
        ks = key_states[0].float()  # [H, T, D]
        scores = torch.einsum("hsd,htd->hst", qs, ks) * scale  # [H, S_rows, T]

        # Causal mask
        dev = query_states.device
        qpos = qsel[:, None].to(dev)  # [S_rows, 1]
        kpos = torch.arange(t, device=dev)[None, :]  # [1, T]
        causal = kpos > qpos
        scores = scores.masked_fill(causal[None, :, :], float("-inf"))

        probs = F.softmax(scores.float(), dim=-1)  # [H, S_rows, T]
        tot = probs.sum(-1, keepdim=True).clamp_min(1e-12)
        probs = probs / tot

        # Near-window mass per (head, query)
        dist = (qpos - kpos).float()  # [S_rows, T], positive = in window
        in_win = (dist >= 0) & (dist < args.window_size)  # [S_rows, T]
        head_win_mass_sum = (probs * in_win.float()[None, :, :]).sum(-1).sum(-1)  # [H]
        head_cnt = qsel.numel()

        # Coverage back-distance per (head, query)
        order = torch.argsort(dist, dim=-1, descending=False)  # [S_rows, T]
        ph_sorted = torch.gather(
            probs, -1, order.unsqueeze(0).expand(n_q_heads, -1, -1)
        )
        dist_sorted = torch.gather(
            dist.unsqueeze(0).expand(n_q_heads, -1, -1),
            -1,
            order.unsqueeze(0).expand(n_q_heads, -1, -1),
        )
        cum = torch.cumsum(ph_sorted, dim=-1)
        reach = (cum >= args.coverage).float().argmax(dim=-1)  # [H, S_rows]
        head_dist_sum = (
            dist_sorted.gather(-1, reach.unsqueeze(-1)).squeeze(-1).sum(-1)
        )  # [H]

        cnt = max(1, head_cnt)
        mean_win_mass = (head_win_mass_sum / cnt).cpu().tolist()
        mean_dist = (head_dist_sum / cnt).cpu().tolist()

        row_global = []
        row_max_dist = []
        for h in range(n_q_heads):
            if mean_win_mass[h] >= args.local_mass_thresh:
                row_global.append(False)  # local
                w = int(math.ceil(mean_dist[h] * args.window_safety))
                row_max_dist.append(max(128, w))
            else:
                row_global.append(True)  # global
                row_max_dist.append(-1)

        n_g = sum(row_global)
        n_l = n_q_heads - n_g
        avg_wm = sum(mean_win_mass) / n_q_heads
        print(f"L{li:3d}   {n_g:4d}/{n_q_heads} {n_l:4d}/{n_q_heads} {avg_wm:.3f}")
        full_head_map[li] = row_global

    # ---- Step 3: Linear attention layers (depth-aware) ----
    max_global_heads = 2
    linear_head_map: dict[int, list[bool]] = {}
    for depth_rank, li in enumerate(linear_layers):
        depth_ratio = depth_rank / max(1, len(linear_layers) - 1)
        k = round(depth_ratio * max_global_heads)
        row = [False] * n_q_heads
        for h in range(min(k, n_q_heads)):
            row[h] = True
        linear_head_map[li] = row

    # ---- Step 4: Build final 40x16 head map ----
    head_map_final = []
    for li in range(n_layers):
        if li in full_layers:
            head_map_final.append(full_head_map[li])
        elif li in linear_layers:
            head_map_final.append(linear_head_map[li])
        else:
            head_map_final.append([False] * n_q_heads)

    total = len(head_map_final) * n_q_heads
    n_global_total = sum(sum(row) for row in head_map_final)

    print(f"\n{'=' * 60}")
    print(f" FINAL HEAD MAP")
    print(f"{'=' * 60}")
    print(
        f"Total: {total}, Global: {n_global_total} ({n_global_total / total * 100:.1f}%)"
    )
    print(f"Per-layer:")
    for li in range(n_layers):
        row = head_map_final[li]
        g = sum(row)
        lt = "full" if li in full_layers else ("linear" if li in linear_layers else "?")
        bar = "".join("R" if x else "B" for x in row)
        print(f"  L{li:2d} [{lt:6s}] {bar}  g={g}/{n_q_heads}")

    output = {
        "model": "Qwen3.5-35B-A3B",
        "method": "real_probe_full_attn_head_locality",
        "num_layers": n_layers,
        "num_heads": n_q_heads,
        "num_kv_heads": n_kv_heads,
        "params": {
            "coverage": args.coverage,
            "local_mass_thresh": args.local_mass_thresh,
            "window_size": args.window_size,
            "window_safety": args.window_safety,
            "qsample": args.qsample,
            "tokens": T,
            "linear_max_global_heads": max_global_heads,
        },
        "summary": {
            "global": n_global_total,
            "local": total - n_global_total,
            "total": total,
            "global_pct": f"{n_global_total / total * 100:.1f}%",
            "local_pct": f"{(total - n_global_total) / total * 100:.1f}%",
        },
        "head_map": head_map_final,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
