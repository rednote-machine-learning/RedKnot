#!/usr/bin/env python3
"""Measure layer-wise attention concentration for the RedKnot motivation figure.

Metric (per layer): the fraction of key tokens needed to cover ``coverage`` of
the total attention mass, averaged over sampled query rows. A diffuse layer
needs many tokens (fraction near 1.0); a concentrated layer needs only a few
(fraction near 0.0). The expected trend is high in shallow layers and dropping
toward deep layers.

Two faithful real-weight paths:

* ``Llama-3.3-70B``: real per-layer HuggingFace forward; attention is rebuilt
  per layer from real q/k projections on the real hidden state.
* ``DeepSeek-V4``: real fp8 MLA weights (wq_a -> q_norm -> wq_b -> per-head RMS
  -> RoPE for Q; wkv -> kv_norm -> RoPE for the shared latent K). The hidden
  state evolves via real attn + wo + residual each layer; the DeepSeek hc
  residual-mixing and the MoE/FFN block are skipped (noted in the output).

Supports multiple LongBench datasets and multiple context lengths in one run.
Output JSON is consumed by make_motivation_figures.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))
REDKNOT_DIR = REPO / "test" / "srt" / "redknot"
LONGBENCH_DIR = Path(
    os.environ.get(
        "REDKNOT_LONGBENCH_DIR",
        "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
    )
)
DSV4_PATH = "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/DeepSeek-V4-Flash"


# ──────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────
def _load_longbench_ids(tokenizer, dataset: str, target_tokens: int) -> torch.Tensor:
    """Concatenate LongBench ``context`` fields until ``target_tokens`` is met."""
    path = LONGBENCH_DIR / f"{dataset}.jsonl"
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    random.Random(2026).shuffle(rows)
    toks: list[int] = []
    for row in rows:
        ctx = row.get("context") or row.get("input") or ""
        if not ctx:
            continue
        toks.extend(tokenizer(ctx, add_special_tokens=False)["input_ids"])
        if len(toks) >= target_tokens:
            break
    if len(toks) < target_tokens:
        # Repeat to reach the target length if a single dataset is short.
        if not toks:
            raise ValueError(f"no usable context in {path}")
        while len(toks) < target_tokens:
            toks.extend(toks[: target_tokens - len(toks)])
    return torch.tensor([toks[:target_tokens]], dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────
# Metric
# ──────────────────────────────────────────────────────────────────────────
def _coverage_fraction(
    probs: torch.Tensor, coverage: float, min_visible: int | None = None
) -> float:
    """Mean fraction of *visible* keys needed to reach ``coverage`` mass.

    probs: [H, S, T] per-(head, query) attention distribution (softmaxed,
    causally masked). For each (head, query) we sort keys descending, find how
    many are needed to reach ``coverage`` of the visible mass, and divide by the
    number of visible keys for that query. Computed per head (sharp single-head
    structure is preserved, not averaged away) then averaged over heads/queries.

    Only queries whose visible length is at least ``min_visible`` are counted, to
    avoid the position bias where early queries see only a few keys. When
    ``min_visible`` is None it defaults to 25% of the sequence (dense case).
    """
    if probs.dim() == 2:
        probs = probs.unsqueeze(0)
    H, S, T = probs.shape
    if min_visible is None:
        min_visible = max(8, int(0.25 * T))
    fracs: list[torch.Tensor] = []
    # Process head-by-head to keep the sort/cumsum memory bounded by [S, T].
    for h in range(H):
        ph = probs[h].float()  # [S, T]
        visible = (ph > 0).sum(dim=-1)  # [S]
        keep = visible >= min_visible
        sorted_p, _ = torch.sort(ph, dim=-1, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        cum_frac = cum / cum[:, -1:].clamp_min(1e-12)
        n_needed = (cum_frac >= coverage).float().argmax(dim=-1) + 1  # [S]
        frac = (n_needed.float() / visible.clamp_min(1).float()).clamp(0, 1)
        sel = frac[keep] if bool(keep.any()) else frac
        fracs.append(sel.mean().detach())
        del ph, sorted_p, cum, cum_frac, n_needed, frac
    return float(torch.stack(fracs).mean().item())


# ──────────────────────────────────────────────────────────────────────────
# Llama (real per-layer forward)
# ──────────────────────────────────────────────────────────────────────────
def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


def _apply_llama_rope(q, k, cos, sin):
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


@torch.no_grad()
def measure_llama(
    model_path: str,
    dataset: str,
    target_tokens: int,
    qsample: int,
    coverage: float,
    dtype_mode: str,
) -> list[float]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if dtype_mode == "int4":
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation="sdpa",
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=os.environ.get("REDKNOT_DEVICE_MAP", "auto"),
            trust_remote_code=True,
            attn_implementation="sdpa",
        ).eval()

    input_ids = _load_longbench_ids(tokenizer, dataset, target_tokens).to(model.device)
    base = model.model if hasattr(model, "model") else model
    n_layers = int(model.config.num_hidden_layers)
    n_heads = int(model.config.num_attention_heads)
    n_kv = int(model.config.num_key_value_heads)
    n_rep = n_heads // n_kv
    head_dim = int(
        getattr(model.config, "head_dim", model.config.hidden_size // n_heads)
    )
    out: dict[int, float] = {}
    handles = []

    for layer_id, layer in enumerate(base.layers):
        attn = layer.self_attn

        def make_hook(li, module):
            def hook(_m, args, kwargs):
                hs = (
                    args[0]
                    if args and torch.is_tensor(args[0])
                    else kwargs.get("hidden_states")
                )
                pe = kwargs.get("position_embeddings")
                if hs is None:
                    return
                b, t, _ = hs.shape
                q = module.q_proj(hs).view(b, t, n_heads, head_dim).transpose(1, 2)
                k = module.k_proj(hs).view(b, t, n_kv, head_dim).transpose(1, 2)
                if pe is not None:
                    cos, sin = pe
                    q, k = _apply_llama_rope(q, k, cos.to(q.device), sin.to(q.device))
                k = _repeat_kv(k, n_rep)
                n_rows = min(qsample, t)
                qi = (
                    torch.linspace(0, t - 1, steps=n_rows, device=q.device)
                    .long()
                    .unique()
                )
                qs = q[:, :, qi, :]
                scores = torch.matmul(qs, k.transpose(-1, -2)) * (head_dim**-0.5)
                key_pos = torch.arange(t, device=q.device)
                causal = key_pos[None, :] > qi[:, None]
                scores = scores.masked_fill(causal[None, None], float("-inf"))
                probs = F.softmax(scores.float(), dim=-1)[0]  # [H, S, T]
                out[li] = _coverage_fraction(probs, coverage)

            return hook

        handles.append(
            attn.register_forward_pre_hook(make_hook(layer_id, attn), with_kwargs=True)
        )

    model(input_ids=input_ids, use_cache=False)
    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return [out[i] for i in range(n_layers)]


# ──────────────────────────────────────────────────────────────────────────
# DeepSeek-V4 (real fp8 MLA weights, real per-layer hidden propagation)
# ──────────────────────────────────────────────────────────────────────────
def _dequant_fp8_blockwise(weight, scale, block: int = 128):
    w = weight.to(torch.float32)
    out_dim, in_dim = w.shape
    s = scale.to(torch.float32)
    if scale.dtype == torch.float8_e8m0fnu:
        s = torch.pow(2.0, s)
    n_rb = (out_dim + block - 1) // block
    n_cb = (in_dim + block - 1) // block
    s = s[:n_rb, :n_cb]
    s = s.repeat_interleave(block, 0)[:out_dim].repeat_interleave(block, 1)[:, :in_dim]
    return (w * s).to(torch.bfloat16)


def _precompute_freqs_cis(
    dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow
):
    def find_correction_dim(num_rot, dim, base, max_len):
        return dim * math.log(max_len / (num_rot * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_len))
        return max(low, 0), min(high, dim - 1)

    def ramp(lo, hi, dim):
        if lo == hi:
            hi += 0.001
        f = (torch.arange(dim, dtype=torch.float32) - lo) / (hi - lo)
        return torch.clamp(f, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - ramp(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rope_complex(x, freqs_cis):
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    T = xc.size(0)
    half = xc.size(-1)
    shape = [T] + [1] * (xc.ndim - 2) + [half]
    fc = freqs_cis[:T].view(*shape)
    return torch.view_as_real(xc * fc).flatten(-2).to(x.dtype)


def _rms_norm(x, weight, eps=1e-6):
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight.float()).to(weight.dtype)


def _hadamard_transform(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Pure-PyTorch fast Walsh-Hadamard transform along the last dim.

    Replicates fast_hadamard_transform.hadamard_transform(x, scale): the last
    dim must be a power of two. Used by the DeepSeek-V4 indexer to spread
    information across channels before fp4 quant.
    """
    orig_dtype = x.dtype
    x = x.float()
    *lead, n = x.shape
    assert (n & (n - 1)) == 0, f"hadamard dim {n} is not a power of two"
    y = x.reshape(-1, n).clone()
    h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :].clone()
        b = y[:, :, 1, :].clone()
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        y = y.view(-1, n)
        h *= 2
    y = y * scale
    return y.view(*lead, n).to(orig_dtype)


@torch.no_grad()
def _compress_index_kv(x, wkv_c, wgate_c, ape, norm_w, freqs, ratio, head_dim, rd, eps):
    """Replicate Compressor.forward (prefill, overlap when ratio==4) to build the
    compressed KV blocks that the indexer scores. Returns [n_blocks, head_dim].
    x: [T, dim] float. Produces gated-pooled, normed, RoPE'd compressed blocks.
    """
    T = x.shape[0]
    overlap = ratio == 4
    coff = 1 + overlap
    kv = F.linear(x, wkv_c)  # [T, coff*head_dim]
    score = F.linear(x, wgate_c)  # [T, coff*head_dim]
    cutoff = T - (T % ratio)
    if cutoff < ratio:
        return None
    kv = kv[:cutoff]
    score = score[:cutoff]
    kv = kv.unflatten(0, (-1, ratio))  # [nb, ratio, coff*head_dim]
    score = score.unflatten(0, (-1, ratio)) + ape  # ape: [ratio, coff*head_dim]
    if overlap:
        nb = kv.shape[0]
        d = head_dim
        kv_new = kv.new_zeros((nb, 2 * ratio, d))
        kv_new[:, ratio:] = kv[:, :, d:]
        kv_new[1:, :ratio] = kv[:-1, :, :d]
        sc_new = score.new_full((nb, 2 * ratio, d), float("-inf"))
        sc_new[:, ratio:] = score[:, :, d:]
        sc_new[1:, :ratio] = score[:-1, :, :d]
        kv, score = kv_new, sc_new
    pooled = (kv * score.softmax(dim=1)).sum(dim=1)  # [nb, head_dim]
    pooled = _rms_norm(pooled.to(norm_w.dtype), norm_w, eps).float()
    nb = pooled.shape[0]
    rope_part = _apply_rope_complex(
        pooled[:, -rd:].unsqueeze(1), freqs[:cutoff:ratio][:nb]
    ).squeeze(1)
    pooled = torch.cat([pooled[:, :-rd], rope_part], dim=-1)
    return pooled  # [nb, head_dim]


@torch.no_grad()
def measure_deepseek_v4(
    model_path: str,
    dataset: str,
    target_tokens: int,
    qsample: int,
    coverage: float,
    device: str,
    dense_causal: bool = False,
) -> list[float]:
    from safetensors import safe_open
    from transformers import AutoTokenizer

    cfg = json.load(open(Path(model_path) / "config.json"))
    n_layers = int(cfg["num_hidden_layers"])
    n_heads = int(cfg["num_attention_heads"])
    head_dim = int(cfg["head_dim"])
    rd = int(cfg["qk_rope_head_dim"])
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    compress_ratios = cfg["compress_ratios"]
    rope_theta = float(cfg.get("rope_theta", 10000))
    compress_rope_theta = float(cfg.get("compress_rope_theta", 160000))
    rope_factor = float(cfg["rope_scaling"]["factor"])
    original_seq_len = int(cfg["rope_scaling"]["original_max_position_embeddings"])
    beta_fast = int(cfg["rope_scaling"]["beta_fast"])
    beta_slow = int(cfg["rope_scaling"]["beta_slow"])
    n_groups = int(cfg.get("o_groups", 8))
    o_lora_rank = int(cfg.get("o_lora_rank", 1024))
    window = int(cfg.get("sliding_window", 128))

    index = json.load(open(Path(model_path) / "model.safetensors.index.json"))[
        "weight_map"
    ]

    def _get(name):
        shard = index[name]
        with safe_open(str(Path(model_path) / shard), framework="pt") as f:
            return f.get_tensor(name)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_ids = _load_longbench_ids(tokenizer, dataset, target_tokens)[0].tolist()
    seqlen = len(input_ids)

    embed = _get("embed.weight")
    hidden = embed[torch.tensor(input_ids)].to(device=device, dtype=torch.bfloat16)

    freqs_plain = _precompute_freqs_cis(
        rd, seqlen, 0, rope_theta, rope_factor, beta_fast, beta_slow
    ).to(device)
    freqs_yarn = _precompute_freqs_cis(
        rd,
        seqlen,
        original_seq_len,
        compress_rope_theta,
        rope_factor,
        beta_fast,
        beta_slow,
    ).to(device)

    out: dict[int, float] = {}
    n_rows = min(qsample, seqlen)
    query_idx = (
        torch.linspace(0, seqlen - 1, steps=n_rows, device=device).long().unique()
    )
    key_pos = torch.arange(seqlen, device=device)
    scale = head_dim**-0.5

    for layer in range(n_layers):
        p = f"layers.{layer}.attn."
        attn_norm = _get(f"layers.{layer}.attn_norm.weight").to(device)
        wq_a = _dequant_fp8_blockwise(
            _get(p + "wq_a.weight"), _get(p + "wq_a.scale")
        ).to(device)
        wq_b = _dequant_fp8_blockwise(
            _get(p + "wq_b.weight"), _get(p + "wq_b.scale")
        ).to(device)
        wkv = _dequant_fp8_blockwise(_get(p + "wkv.weight"), _get(p + "wkv.scale")).to(
            device
        )
        q_norm_w = _get(p + "q_norm.weight").to(device)
        kv_norm_w = _get(p + "kv_norm.weight").to(device)
        attn_sink = _get(p + "attn_sink").to(device).float()  # [H]
        wo_a = _dequant_fp8_blockwise(
            _get(p + "wo_a.weight"), _get(p + "wo_a.scale")
        ).to(device)
        wo_b = _dequant_fp8_blockwise(
            _get(p + "wo_b.weight"), _get(p + "wo_b.scale")
        ).to(device)

        x = _rms_norm(hidden, attn_norm, eps)
        qr = _rms_norm(F.linear(x, wq_a), q_norm_w, eps)
        q = F.linear(qr, wq_b).view(seqlen, n_heads, head_dim)
        q = q * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + eps).to(q.dtype)
        kv = _rms_norm(F.linear(x, wkv), kv_norm_w, eps)

        ratio = compress_ratios[layer] if layer < len(compress_ratios) else 0
        freqs = freqs_plain if ratio == 0 else freqs_yarn
        q = torch.cat([q[..., :-rd], _apply_rope_complex(q[..., -rd:], freqs)], dim=-1)
        kv = torch.cat(
            [
                kv[..., :-rd],
                _apply_rope_complex(kv[..., -rd:].unsqueeze(1), freqs).squeeze(1),
            ],
            dim=-1,
        )

        scores = torch.einsum("thd,sd->ths", q.float(), kv.float()) * scale  # [T,H,S]
        qpos = torch.arange(seqlen, device=device)[:, None, None]
        kpos = key_pos[None, None, :]
        causal = kpos > qpos
        # DeepSeek-V4 attention is sparse: each query sees the recent sliding
        # window of `window_size` raw tokens plus compressed history blocks of
        # `ratio` tokens each (ratio==0 layers are pure sliding window). Mask out
        # tokens outside the window so the metric reflects the real visible set.
        out_of_window = kpos < (qpos - window + 1)
        if dense_causal:
            # Dense causal: every query attends to ALL past tokens (no sliding
            # window / block sparsity). Same metric definition as the Llama MHA
            # panel, so the layer-wise token-mass concentration is comparable and
            # the expected "concentrating toward deep layers" trend is visible.
            visible_mask = ~causal
        elif ratio == 0:
            visible_mask = ~(causal | out_of_window)
        else:
            # Compressed history: every `ratio`-th key represents one block;
            # keep window tokens plus block-representative tokens before it.
            is_block_rep = (kpos % ratio) == 0
            visible_mask = ~causal & (~out_of_window | is_block_rep)
        scores = scores.masked_fill(~visible_mask, float("-inf"))

        # attn_sink: a learnable per-head bias term in the softmax denominator
        # (a virtual sink that absorbs part of the mass). Append as an extra
        # column so the normalized token probabilities exclude the sink mass.
        sink_col = attn_sink.view(1, n_heads, 1).expand(seqlen, n_heads, 1)
        scores_aug = torch.cat([scores, sink_col], dim=-1)  # [T,H,S+1]
        probs_aug = F.softmax(scores_aug, dim=-1)
        probs = probs_aug[..., :-1]  # drop the sink column; token mass only

        probs_metric = probs[query_idx].permute(1, 0, 2)  # [H, S_rows, T]
        # Sparse visibility means most queries see ~window..window+blocks keys;
        # count any query that sees at least the full sliding window. In dense
        # causal mode let _coverage_fraction use its default (25% of seq) so we
        # only score queries with enough visible history, matching Llama.
        out[layer] = _coverage_fraction(
            probs_metric, coverage, min_visible=None if dense_causal else window
        )

        attn_out = torch.einsum("ths,sd->thd", probs.to(kv.dtype), kv)
        o = attn_out.reshape(seqlen, n_groups, -1)
        wo_a_g = wo_a.view(n_groups, o_lora_rank, -1)
        o = torch.einsum("tgd,grd->tgr", o.float(), wo_a_g.float())
        o = F.linear(o.reshape(seqlen, -1).to(wo_b.dtype), wo_b)
        hidden = hidden + o

        del wq_a, wq_b, wkv, wo_a, wo_b, q, kv, qr, x, scores, probs, attn_out, o
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return [out[i] for i in range(n_layers)]


@torch.no_grad()
def measure_deepseek_v4_head_locality(
    model_path: str,
    dataset: str,
    target_tokens: int,
    qsample: int,
    coverage: float,
    device: str,
    local_mass_thresh: float = 0.80,
    local_window_mult: int = 8,
    window_safety: float = 1.5,
    min_local_window: int = 128,
    global_distance_frac: float = 0.5,  # kept for CLI compat; unused here
) -> dict:
    """Per-(layer, logical head) global/local classification for MLA RedKnot.

    DeepSeek-V4 MLA stores a single latent KV stream, but each of the
    ``num_attention_heads`` logical heads decompresses its own Q/K and attends
    with its own distribution. This replays the real fp8 MLA forward (DENSE
    causal so every head can see the whole history) and, per (layer, head),
    measures the *near-window mass fraction*: the share of attention mass that
    falls within the most recent ``W = local_window_mult * sliding_window``
    tokens, averaged over sampled queries.

    Classification (near-window mass criterion, robust to long tails):
      * mass_in_window >= ``local_mass_thresh`` -> LOCAL. Its window is set from
        the coverage back-distance (covering ``coverage`` of mass) scaled by
        ``window_safety`` and clamped to >= ``min_local_window``.
      * otherwise the head spreads mass far back -> GLOBAL.

    Returns a dict with per-layer head_class / head_max_distance lists ready to
    feed ``DeepSeekV4MLAHeadConfig``.
    """
    from safetensors import safe_open
    from transformers import AutoTokenizer

    cfg = json.load(open(Path(model_path) / "config.json"))
    n_layers = int(cfg["num_hidden_layers"])
    n_heads = int(cfg["num_attention_heads"])
    head_dim = int(cfg["head_dim"])
    rd = int(cfg["qk_rope_head_dim"])
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    compress_ratios = cfg["compress_ratios"]
    rope_theta = float(cfg.get("rope_theta", 10000))
    compress_rope_theta = float(cfg.get("compress_rope_theta", 160000))
    rope_factor = float(cfg["rope_scaling"]["factor"])
    original_seq_len = int(cfg["rope_scaling"]["original_max_position_embeddings"])
    beta_fast = int(cfg["rope_scaling"]["beta_fast"])
    beta_slow = int(cfg["rope_scaling"]["beta_slow"])
    n_groups = int(cfg.get("o_groups", 8))
    o_lora_rank = int(cfg.get("o_lora_rank", 1024))
    sliding_window = int(cfg.get("sliding_window", cfg.get("window_size", 128)) or 128)
    near_window = max(min_local_window, local_window_mult * sliding_window)

    index = json.load(open(Path(model_path) / "model.safetensors.index.json"))[
        "weight_map"
    ]

    def _get(name):
        shard = index[name]
        with safe_open(str(Path(model_path) / shard), framework="pt") as f:
            return f.get_tensor(name)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_ids = _load_longbench_ids(tokenizer, dataset, target_tokens)[0].tolist()
    seqlen = len(input_ids)

    embed = _get("embed.weight")
    hidden = embed[torch.tensor(input_ids)].to(device=device, dtype=torch.bfloat16)

    freqs_plain = _precompute_freqs_cis(
        rd, seqlen, 0, rope_theta, rope_factor, beta_fast, beta_slow
    ).to(device)
    freqs_yarn = _precompute_freqs_cis(
        rd,
        seqlen,
        original_seq_len,
        compress_rope_theta,
        rope_factor,
        beta_fast,
        beta_slow,
    ).to(device)

    n_rows = min(qsample, seqlen)
    query_idx = (
        torch.linspace(0, seqlen - 1, steps=n_rows, device=device).long().unique()
    )
    key_pos = torch.arange(seqlen, device=device)
    scale = head_dim**-0.5
    # Only score queries that can see enough history for the distance to be
    # meaningful (skip the first 25% positions).
    min_qpos = max(8, int(0.25 * seqlen))

    head_class = [["dense"] * n_heads for _ in range(n_layers)]
    head_max_distance = [[-1] * n_heads for _ in range(n_layers)]

    for layer in range(n_layers):
        p = f"layers.{layer}.attn."
        attn_norm = _get(f"layers.{layer}.attn_norm.weight").to(device)
        wq_a = _dequant_fp8_blockwise(
            _get(p + "wq_a.weight"), _get(p + "wq_a.scale")
        ).to(device)
        wq_b = _dequant_fp8_blockwise(
            _get(p + "wq_b.weight"), _get(p + "wq_b.scale")
        ).to(device)
        wkv = _dequant_fp8_blockwise(_get(p + "wkv.weight"), _get(p + "wkv.scale")).to(
            device
        )
        q_norm_w = _get(p + "q_norm.weight").to(device)
        kv_norm_w = _get(p + "kv_norm.weight").to(device)
        attn_sink = _get(p + "attn_sink").to(device).float()  # [H]
        wo_a = _dequant_fp8_blockwise(
            _get(p + "wo_a.weight"), _get(p + "wo_a.scale")
        ).to(device)
        wo_b = _dequant_fp8_blockwise(
            _get(p + "wo_b.weight"), _get(p + "wo_b.scale")
        ).to(device)

        x = _rms_norm(hidden, attn_norm, eps)
        qr = _rms_norm(F.linear(x, wq_a), q_norm_w, eps)
        q = F.linear(qr, wq_b).view(seqlen, n_heads, head_dim)
        q = q * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + eps).to(q.dtype)
        kv = _rms_norm(F.linear(x, wkv), kv_norm_w, eps)

        ratio = compress_ratios[layer] if layer < len(compress_ratios) else 0
        freqs = freqs_plain if ratio == 0 else freqs_yarn
        q = torch.cat([q[..., :-rd], _apply_rope_complex(q[..., -rd:], freqs)], dim=-1)
        kv = torch.cat(
            [
                kv[..., :-rd],
                _apply_rope_complex(kv[..., -rd:].unsqueeze(1), freqs).squeeze(1),
            ],
            dim=-1,
        )

        # Only score sampled query rows that can see enough history, so memory
        # stays O(n_rows * H * S) instead of O(T * H * S) (which OOMs at long T).
        qsel = query_idx[query_idx >= min_qpos]
        if qsel.numel() == 0:
            qsel = query_idx[-max(1, n_rows // 4) :]
        kv_f = kv.float()
        scale_t = scale
        # Accumulators over sampled queries:
        #  - near-window mass fraction per head (the global/local criterion)
        #  - coverage back-distance per head (used to size the local window)
        head_win_mass_sum = torch.zeros(n_heads, device=device)
        head_dist_sum = torch.zeros(n_heads, device=device)
        head_cnt = 0
        for t in qsel.tolist():
            qrow = q[t].float()  # [H, head_dim]
            kvis = kv_f[: t + 1]  # [t+1, head_dim] causal visible keys
            sc = torch.einsum("hd,sd->hs", qrow, kvis) * scale_t  # [H, t+1]
            sink_col = attn_sink.view(n_heads, 1)  # [H,1]
            sc_aug = torch.cat([sc, sink_col], dim=-1)  # [H, t+2]
            ph = F.softmax(sc_aug, dim=-1)[:, :-1]  # [H, t+1] drop sink
            tot = ph.sum(-1, keepdim=True).clamp_min(1e-12)
            ph = ph / tot
            dist = (t - torch.arange(t + 1, device=device)).float()  # 0=self
            # near-window mass: share of mass on keys within `near_window` back.
            in_win = (dist < near_window).float()  # [t+1]
            head_win_mass_sum += (ph * in_win[None, :]).sum(-1)  # [H]
            # coverage back-distance (nearest-first cumulative) to size window.
            order = torch.argsort(dist, descending=False)
            ph_sorted = ph[:, order]
            dist_sorted = dist[order]
            cum = torch.cumsum(ph_sorted, dim=-1)
            reach = (cum >= coverage).float().argmax(dim=-1)  # [H]
            head_dist_sum += dist_sorted[reach]  # [H]
            head_cnt += 1
            del qrow, kvis, sc, sc_aug, ph, ph_sorted, cum, reach, in_win
        mean_win_mass = (head_win_mass_sum / max(1, head_cnt)).tolist()  # [H]
        mean_dist = (head_dist_sum / max(1, head_cnt)).tolist()  # [H]

        for h in range(n_heads):
            if mean_win_mass[h] >= local_mass_thresh:
                head_class[layer][h] = "local"
                w = int(math.ceil(mean_dist[h] * window_safety))
                head_max_distance[layer][h] = max(min_local_window, w)
            else:
                head_class[layer][h] = "global"
                head_max_distance[layer][h] = -1

        # Propagate hidden via real dense-causal attention, chunked over query
        # rows to keep memory at O(chunk * H * S) instead of O(T * H * S).
        wo_a_g = wo_a.view(n_groups, o_lora_rank, -1)
        chunk = max(64, min(512, seqlen))
        o_full = torch.empty(
            seqlen, hidden.shape[-1], device=device, dtype=hidden.dtype
        )
        for cs in range(0, seqlen, chunk):
            ce = min(cs + chunk, seqlen)
            qc = q[cs:ce].float()  # [c, H, hd]
            sc = torch.einsum("chd,sd->chs", qc, kv_f) * scale_t  # [c, H, S]
            qposc = torch.arange(cs, ce, device=device)[:, None, None]
            kposc = key_pos[None, None, :]
            sc = sc.masked_fill(kposc > qposc, float("-inf"))
            sink_col = attn_sink.view(1, n_heads, 1).expand(ce - cs, n_heads, 1)
            sc_aug = torch.cat([sc, sink_col], dim=-1)
            pc = F.softmax(sc_aug, dim=-1)[..., :-1]  # [c, H, S]
            ao = torch.einsum("chs,sd->chd", pc.to(kv.dtype), kv)  # [c, H, hd]
            oc = ao.reshape(ce - cs, n_groups, -1)
            oc = torch.einsum("cgd,grd->cgr", oc.float(), wo_a_g.float())
            oc = F.linear(oc.reshape(ce - cs, -1).to(wo_b.dtype), wo_b)
            o_full[cs:ce] = oc
            del qc, sc, sc_aug, pc, ao, oc
        hidden = hidden + o_full

        n_glob = sum(1 for h in range(n_heads) if head_class[layer][h] == "global")
        print(
            f"  [head-locality] layer {layer:3d}: global={n_glob:3d}/{n_heads} "
            f"local={n_heads - n_glob:3d} "
            f"avg_win_mass={sum(mean_win_mass) / n_heads:.3f} "
            f"(near_window={near_window})"
        )
        del wq_a, wq_b, wkv, wo_a, wo_b, q, kv, kv_f, qr, x, o_full
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return {
        "num_layers": n_layers,
        "num_attention_heads": n_heads,
        "head_class": head_class,
        "head_max_distance": head_max_distance,
        "seqlen": seqlen,
        "near_window": near_window,
        "local_mass_thresh": local_mass_thresh,
        "window_safety": window_safety,
    }


@torch.no_grad()
def measure_deepseek_v4_indexer(
    model_path: str,
    dataset: str,
    target_tokens: int,
    qsample: int,
    coverage: float,
    device: str,
) -> list[float]:
    """Measure DeepSeek-V4 token-importance concentration via the INDEXER.

    DeepSeek-V4 selects the top-k=512 compressed KV blocks per query with a
    dedicated indexer (independent Q/K + learned head weights). The real
    token-importance signal is the indexer's ``index_score``; softmax attention
    only runs inside the selected set. This faithfully replays the indexer
    (compressor gated pooling + indexer Q + relu(QK)*weights) per layer and
    measures the fraction of candidate blocks needed to cover ``coverage`` of
    the positive index-score mass (averaged over sampled queries).

    fp4 activation quantization is approximated as identity (it perturbs values
    but not the ranking trend); the Hadamard rotation is reproduced exactly.
    Layers without an indexer (compress_ratio==0, pure sliding window) are
    reported as 1.0 (no selection performed).
    """
    from safetensors import safe_open
    from transformers import AutoTokenizer

    cfg = json.load(open(Path(model_path) / "config.json"))
    n_layers = int(cfg["num_hidden_layers"])
    rd = int(cfg["qk_rope_head_dim"])
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    compress_ratios = cfg["compress_ratios"]
    compress_rope_theta = float(cfg.get("compress_rope_theta", 160000))
    rope_factor = float(cfg["rope_scaling"]["factor"])
    original_seq_len = int(cfg["rope_scaling"]["original_max_position_embeddings"])
    beta_fast = int(cfg["rope_scaling"]["beta_fast"])
    beta_slow = int(cfg["rope_scaling"]["beta_slow"])
    q_lora_rank = int(cfg["q_lora_rank"])
    idx_heads = int(cfg["index_n_heads"])
    idx_hd = int(cfg["index_head_dim"])
    idx_topk = int(cfg["index_topk"])

    index = json.load(open(Path(model_path) / "model.safetensors.index.json"))[
        "weight_map"
    ]

    def _get(name):
        with safe_open(str(Path(model_path) / index[name]), framework="pt") as f:
            return f.get_tensor(name)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_ids = _load_longbench_ids(tokenizer, dataset, target_tokens)[0].tolist()
    seqlen = len(input_ids)
    embed = _get("embed.weight")
    hidden = embed[torch.tensor(input_ids)].to(device=device, dtype=torch.bfloat16)

    freqs_yarn = _precompute_freqs_cis(
        rd,
        seqlen,
        original_seq_len,
        compress_rope_theta,
        rope_factor,
        beta_fast,
        beta_slow,
    ).to(device)

    n_rows = min(qsample, seqlen)
    query_idx = (
        torch.linspace(0, seqlen - 1, steps=n_rows, device=device).long().unique()
    )
    out: dict[int, float] = {}

    for layer in range(n_layers):
        ratio = compress_ratios[layer] if layer < len(compress_ratios) else 0
        p = f"layers.{layer}.attn."
        ip = p + "indexer."
        # Only ratio==4 layers carry a learned indexer; ratio==0 (pure sliding
        # window) and ratio==128 (deterministic compress) perform no learned
        # top-k token selection, so they are reported as 1.0.
        if ratio != 4 or (ip + "wq_b.weight") not in index:
            out[layer] = 1.0
            continue
        attn_norm = _get(f"layers.{layer}.attn_norm.weight").to(device)
        wq_a = _dequant_fp8_blockwise(
            _get(p + "wq_a.weight"), _get(p + "wq_a.scale")
        ).to(device)
        q_norm_w = _get(p + "q_norm.weight").to(device)
        idx_wq_b = _dequant_fp8_blockwise(
            _get(ip + "wq_b.weight"), _get(ip + "wq_b.scale")
        ).to(device)
        weights_proj = _get(ip + "weights_proj.weight").to(device)
        c_wkv = _get(ip + "compressor.wkv.weight").float().to(device)
        c_wgate = _get(ip + "compressor.wgate.weight").float().to(device)
        c_ape = _get(ip + "compressor.ape").float().to(device)
        c_norm = _get(ip + "compressor.norm.weight").to(device)

        x = _rms_norm(hidden, attn_norm, eps)
        qr = _rms_norm(F.linear(x, wq_a), q_norm_w, eps)  # [T, q_lora_rank]

        # Indexer Q: wq_b -> heads -> RoPE(rope dims) -> Hadamard rotate.
        q = F.linear(qr, idx_wq_b).view(seqlen, idx_heads, idx_hd)
        q = torch.cat(
            [q[..., :-rd], _apply_rope_complex(q[..., -rd:], freqs_yarn)], dim=-1
        )
        q = _hadamard_transform(q, idx_hd**-0.5)

        # Compressed KV blocks the indexer scores against.
        kv_blocks = _compress_index_kv(
            x.float(), c_wkv, c_wgate, c_ape, c_norm, freqs_yarn, ratio, idx_hd, rd, eps
        )
        if kv_blocks is None:
            out[layer] = 1.0
            continue
        kv_blocks = _hadamard_transform(kv_blocks, idx_hd**-0.5)
        nb = kv_blocks.shape[0]

        # weights: per-(token, head) scalar; scale per official code.
        softmax_scale = idx_hd**-0.5
        weights = F.linear(x, weights_proj).float() * (
            softmax_scale * idx_heads**-0.5
        )  # [T, H]

        # index_score[t, block] = sum_h relu(q[t,h] . kv[block]) * weights[t,h]
        qs = q[query_idx].float()  # [S, H, D]
        ws = weights[query_idx]  # [S, H]
        # causal mask: block b (covers tokens up to (b+1)*ratio) visible to token t
        # if (b+1)*ratio <= t+1  <=> b < (t+1)//ratio
        score = torch.einsum("shd,bd->shb", qs, kv_blocks.float()).relu_()
        index_score = (score * ws.unsqueeze(-1)).sum(dim=1)  # [S, nb]
        block_idx = torch.arange(nb, device=device)
        vis = block_idx[None, :] < ((query_idx[:, None] + 1) // ratio)
        index_score = index_score.masked_fill(~vis, 0.0)

        # Concentration: fraction of visible candidate blocks needed to cover
        # `coverage` of the positive index-score mass per query.
        fracs = []
        for s in range(index_score.shape[0]):
            row = index_score[s]
            n_vis = int(vis[s].sum())
            if n_vis < 8:
                continue
            vals = row[:n_vis]
            tot = vals.sum().clamp_min(1e-9)
            sv, _ = torch.sort(vals, descending=True)
            cum = torch.cumsum(sv, 0) / tot
            n_need = int((cum >= coverage).float().argmax()) + 1
            # Cap selection at the model's real top-k budget.
            n_need = min(n_need, min(idx_topk, n_vis))
            fracs.append(n_need / n_vis)
        out[layer] = float(np.mean(fracs)) if fracs else 1.0

        del wq_a, idx_wq_b, weights_proj, c_wkv, c_wgate, q, kv_blocks, index_score
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return [out[i] for i in range(n_layers)]


# ──────────────────────────────────────────────────────────────────────────
def _run_head_locality(args, model_path, datasets, lengths) -> None:
    """Run per-(layer, head) global/local classification across configs, vote,
    and export a DeepSeekV4MLAHeadConfig JSON consumable by RedKnot MLA."""
    runs = []
    configs = []
    for dataset in datasets:
        for length in lengths:
            print(f"\n=== head_locality: {dataset} @ {length} ===")
            res = measure_deepseek_v4_head_locality(
                model_path,
                dataset,
                length,
                args.qsample,
                args.coverage,
                args.device,
                local_mass_thresh=args.local_mass_thresh,
                local_window_mult=args.local_window_mult,
                window_safety=args.window_safety,
            )
            runs.append(res)
            configs.append(f"{dataset}@{length}")

    n_layers = runs[0]["num_layers"]
    n_heads = runs[0]["num_attention_heads"]

    # Aggregate: a head is GLOBAL if a MAJORITY of configs call it global
    # (global is the conservative/accuracy-preserving choice). Local window =
    # max coverage distance over configs that called it local.
    head_class = [["dense"] * n_heads for _ in range(n_layers)]
    head_max_distance = [[-1] * n_heads for _ in range(n_layers)]
    n_cfg = len(runs)
    for layer in range(n_layers):
        for h in range(n_heads):
            votes_global = sum(1 for r in runs if r["head_class"][layer][h] == "global")
            if votes_global * 2 >= n_cfg:
                head_class[layer][h] = "global"
                head_max_distance[layer][h] = -1
            else:
                head_class[layer][h] = "local"
                wins = [
                    r["head_max_distance"][layer][h]
                    for r in runs
                    if r["head_class"][layer][h] == "local"
                ]
                head_max_distance[layer][h] = int(max(wins)) if wins else 128

    # Force the first dense_prefix_layers to dense to preserve early signal.
    for layer in range(min(args.dense_prefix_layers, n_layers)):
        for h in range(n_heads):
            head_class[layer][h] = "dense"
            head_max_distance[layer][h] = -1

    from sglang.srt.layers.attention.redknot.deepseek_v4_mla import (
        DeepSeekV4MLAHeadConfig,
    )

    hc = DeepSeekV4MLAHeadConfig(
        head_class=head_class,
        head_max_distance=head_max_distance,
        num_layers=n_layers,
        num_attention_heads=n_heads,
        physical_kv_heads=1,
        dense_prefix_layers=int(args.dense_prefix_layers),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    hc.to_json(str(args.out))

    summary = hc.summary()
    n_total = n_layers * n_heads
    n_g = summary.get("global", 0)
    n_l = summary.get("local", 0)
    n_d = summary.get("dense", 0)
    print("\n" + "=" * 72)
    print(" DeepSeek-V4 MLA HEAD LOCALITY (per logical head global/local)")
    print(f" model: {model_path}")
    print(f" configs: {configs}")
    print(
        f" heads: total={n_total} global={n_g} ({n_g / n_total:.1%}) "
        f"local={n_l} ({n_l / n_total:.1%}) dense={n_d} ({n_d / n_total:.1%})"
    )
    print(f" exported: {args.out}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", required=True, choices=["Llama-3.3-70B", "DeepSeek-V4"]
    )
    parser.add_argument("--model-path", default="")
    parser.add_argument(
        "--datasets",
        default="hotpotqa,multifieldqa_en,gov_report",
        help="Comma-separated LongBench dataset names.",
    )
    parser.add_argument(
        "--lengths",
        default="2048,8192",
        help="Comma-separated context lengths in tokens.",
    )
    parser.add_argument("--qsample", type=int, default=256)
    parser.add_argument("--coverage", type=float, default=0.99)
    parser.add_argument("--dtype", choices=["int4", "bf16"], default="int4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dsv4-mode",
        choices=["softmax", "indexer", "dense", "head_locality"],
        default="indexer",
        help="DeepSeek-V4 metric: 'indexer' measures the top-k selection "
        "concentration (only indexer layers); 'softmax' measures the in-window "
        "(sparse) attention distribution; 'dense' measures full causal "
        "token-level attention-mass concentration over ALL layers (comparable "
        "to the Llama MHA panel, shows the concentrate-with-depth trend); "
        "'head_locality' classifies every (layer, logical head) as global/local "
        "and exports a DeepSeekV4MLAHeadConfig JSON for RedKnot MLA.",
    )
    parser.add_argument(
        "--local-mass-thresh",
        type=float,
        default=0.80,
        help="head_locality: a head is LOCAL if the attention mass within the "
        "near window (>= this fraction) is concentrated there; else GLOBAL.",
    )
    parser.add_argument(
        "--local-window-mult",
        type=int,
        default=8,
        help="head_locality: near window = local_window_mult * sliding_window.",
    )
    parser.add_argument(
        "--global-distance-frac",
        type=float,
        default=0.5,
        help="(deprecated) kept for CLI compatibility; unused by head_locality.",
    )
    parser.add_argument(
        "--window-safety",
        type=float,
        default=1.5,
        help="head_locality: local window = ceil(coverage_distance * safety).",
    )
    parser.add_argument(
        "--dense-prefix-layers",
        type=int,
        default=2,
        help="head_locality: first N layers forced dense in the exported config.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]

    if args.model_name == "Llama-3.3-70B":
        model_path = (
            args.model_path
            or "/mnt/tidal-alsh01/dataset/redone/096/models/Llama-3.3-70B-Instruct"
        )
    else:
        model_path = args.model_path or DSV4_PATH

    if args.dsv4_mode == "head_locality":
        _run_head_locality(args, model_path, datasets, lengths)
        return

    series: list[dict[str, Any]] = []
    n_layers = None
    for dataset in datasets:
        for length in lengths:
            if args.model_name == "Llama-3.3-70B":
                values = measure_llama(
                    model_path, dataset, length, args.qsample, args.coverage, args.dtype
                )
            elif args.dsv4_mode == "indexer":
                values = measure_deepseek_v4_indexer(
                    model_path,
                    dataset,
                    length,
                    args.qsample,
                    args.coverage,
                    args.device,
                )
            else:
                values = measure_deepseek_v4(
                    model_path,
                    dataset,
                    length,
                    args.qsample,
                    args.coverage,
                    args.device,
                    dense_causal=(args.dsv4_mode == "dense"),
                )
            n_layers = len(values)
            series.append({"dataset": dataset, "length": length, "values": values})
            print(
                f"[{args.model_name}] {dataset} @ {length}: L0={values[0]:.3f} Llast={values[-1]:.3f}"
            )

    measured = {
        args.model_name: {
            "metric": f"frac_tokens_for_{int(args.coverage * 100)}pct_mass",
            "method": "real_per_layer_forward"
            if args.model_name == "Llama-3.3-70B"
            else "real_weight_attn_residual_forward_ffn_skipped",
            "coverage": args.coverage,
            "qsample": args.qsample,
            "num_layers": n_layers,
            "layers": list(range(n_layers)) if n_layers else [],
            "series": series,
        }
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(measured, f, indent=2)
        f.write("\n")
    print(args.out)


if __name__ == "__main__":
    main()
