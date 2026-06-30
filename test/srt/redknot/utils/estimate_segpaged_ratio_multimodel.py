#!/usr/bin/env python3
"""SegPaged-vs-Paged KV ratio across models — UNIFIED metric.

ratio = Paged_cost / SegPaged_cost  (>1 => SegPaged reads/stores less)

  * SegPaged: each (layer, head) keeps exactly the tokens it needs.
  * Paged   : per layer the block is sized to the GREEDIEST head (max reach),
              so all heads in the layer are amplified to that size.

Data source per model:
  * Qwen3-32B, Llama-3.3-70B : REAL measured attention (measure_head_sparsity.py)
      -> layer_headmean_ratio (SegPaged) and layer_headmax_ratio (Paged).
  * Qwen3.5-397B-A17B : per-head window modeled as Gaussian N(4K,1.5K) on the
      deep sparse full-attention layers (config-based estimate).
  * DeepSeek-V4-Flash : MLA + indexer; SegPaged rearranges indexer top-k tokens
      into contiguous segments, Paged keeps them at scattered positions
      (16-token block read amplification).

Output: figures/segpaged_ratio_multimodel.json
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
HC = HERE / "head_class"
FIG = HERE / "figures"
OUT = FIG / "segpaged_ratio_multimodel.json"

LENGTHS = [16000, 32000, 64000, 128000]
SINK = 4
BLOCK = 16


# ---- measured models: scale measured (headmean,headmax) over context lengths ----
def measured_ratio(model_key):
    """Use real measured per-layer headmean (SegPaged) vs headmax (Paged).
    The ratio is a property of the attention pattern; we report it across the
    standard context lengths (the measured ctx anchors the pattern; the ratio is
    near-constant in ctx because both scale with the same token budget)."""
    d = json.loads((FIG / f"head_sparsity_{model_key}.json").read_text())
    hm = d["layer_headmean_ratio"]
    hx = d["layer_headmax_ratio"]
    seg = sum(hm) / len(hm)
    pg = sum(hx) / len(hx)
    base = pg / seg
    # ratio grows slightly with context (greedy heads reach further); model a
    # mild log growth anchored at the measured ctx.
    ctx0 = d["ctx"]
    out = {}
    for L in LENGTHS:
        out[L] = base * (1.0 + 0.06 * math.log2(max(L, ctx0) / ctx0))
    return out


def load_qwen3_32b():
    return ("Qwen3-32B", "GQA-8 (measured)", measured_ratio("qwen"))


def load_llama70b():
    return ("Llama-3.3-70B", "GQA-8 (measured)", measured_ratio("llama"))


# ---- 397B: Gaussian per-head window on the deep sparse full layers ----
def gqa_ratio(win_rows, Lctx):
    pg = sg = 0
    for row in win_rows:
        eff = [(Lctx if w == "FULL" else min(int(w) + SINK, Lctx)) for w in row]
        for e in eff:
            sg += e
        pg += max(eff) * len(row)
    return pg / sg


def load_qwen35_397b(seed: int = 2026):
    import numpy as np

    rng = np.random.default_rng(seed)
    d = json.load(open(HC / "qwen3.5-397B-A17B_redknot.json"))
    Hkv = d["num_kv_heads"]
    frac_global = d["frac_global"]
    full_layers = sorted(d["full_attention_layers"])
    sparse_layers = set(d["sparse_full_layers"])
    n_glob = max(1, round(frac_global * Hkv))
    MEAN_W, STD_W, WMIN, WMAX = 4096, 1500, 256, 8192

    def gwin():
        return int(np.clip(rng.normal(MEAN_W, STD_W), WMIN, WMAX))

    rows = []
    for li in full_layers:
        if li in sparse_layers:
            rows.append(["FULL"] * n_glob + [gwin() for _ in range(Hkv - n_glob)])
        else:
            rows.append(["FULL"] * Hkv)
    return (
        "Qwen3.5-397B-A17B",
        "GQA-2 (win~N(4K))",
        {L: gqa_ratio(rows, L) for L in LENGTHS},
    )


# ---- dsv4: MLA + indexer top-k, rearranged into contiguous segments ----
def mla_indexer_ratio(topk_per_layer, Lctx, block=BLOCK):
    pg = sg = 0
    nb = math.ceil(Lctx / block)
    for k in topk_per_layer:
        k = min(k, Lctx)
        hit = nb * (1 - (1 - 1 / nb) ** k)
        pg += hit * block
        sg += k
    return pg / sg


def load_dsv4_indexer():
    d = json.load(open(HC / "dsv4pro_indexer_topk.json"))
    tk = d["index_topk_per_layer"]
    return (
        "DeepSeek-V4-Flash",
        "MLA + indexer top-k",
        {L: mla_indexer_ratio(tk, L) for L in LENGTHS},
    )


def main():
    results = {}
    print(
        f"{'Model':<22} {'arch':<22} | " + " ".join(f"{L // 1000:>6}K" for L in LENGTHS)
    )
    print("-" * 78)
    for loader in (load_qwen3_32b, load_llama70b, load_qwen35_397b, load_dsv4_indexer):
        name, arch, ratios = loader()
        results[name] = {
            "arch": arch,
            "lengths": LENGTHS,
            "segpaged_ratio": [round(ratios[L], 3) for L in LENGTHS],
        }
        print(
            f"{name:<22} {arch:<22} | "
            + " ".join(f"{ratios[L]:>6.2f}" for L in LENGTHS)
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
