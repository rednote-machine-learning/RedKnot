#!/usr/bin/env python3
"""Estimate the heterogeneous-window SegPaged-vs-Paged KV ratio for multiple
models, using each model's REAL per-head window/distance configuration.

This reproduces fig3's `hetero` ratio (analytic KV-cache volume) but generalizes
it beyond Qwen3-32B to Llama-3.3-70B, DeepSeek-V4-Flash, and Qwen3.5-397B-A17B,
by reading each model's head-class config and extracting a per-(layer, kv-head)
effective window.

KV-volume model (identical to hetero_window_bandwidth_qwen3.py):
  SegPaged: sum over heads of  eff_h * D * 2 * dtype   (each head exact)
  Paged   : per layer  max_h(eff_h) * Hkv * D * 2 * dtype  (block == max head)
  ratio   = Paged / SegPaged   (higher => SegPaged saves more)
where eff_h = Lctx if head is FULL else min(window_h + sink, Lctx).

The more heterogeneous the per-head windows, the larger Paged's max() read
amplification, hence the larger SegPaged's advantage.

Output: figures/hetero_ratio_multimodel.json  (+ printed table)
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
HC = HERE / "head_class"
OUT = HERE / "figures" / "hetero_ratio_multimodel.json"

LENGTHS = [16000, 32000, 64000, 128000]
DT = 2  # kv dtype bytes (bf16)
SINK = 4  # sink tokens


def kv_ratio(win_rows, Hkv, D, Lctx):
    """win_rows: list[layer] of list[Hkv] windows ('FULL' or int)."""
    pg = sg = 0
    for row in win_rows:
        eff = [(Lctx if w == "FULL" else min(int(w) + SINK, Lctx)) for w in row]
        for e in eff:
            sg += e * D * 2 * DT
        pg += max(eff) * len(row) * D * 2 * DT
    return pg, sg


def _norm_dist(val, Lctx):
    """Map a per-head max_distance value to a window or FULL.
    Convention in the configs: -1 (or >= context) means a FULL/global head."""
    if val is None or val < 0:
        return "FULL"
    return int(val)


# ---------- per-model loaders: return (win_rows, Hkv, D, name, dist_stats) ----
def load_llama70b():
    d = json.load(open(HC / "llama-70B_optimal_g15_lf_ret.json"))
    md = d["kv_head_max_distance"]  # [L][Hkv]
    Hkv = d["num_kv_heads"]
    D = d.get("head_dim", 128)
    rows = [[_norm_dist(v, 10**9) for v in layer] for layer in md]
    return rows, Hkv, D, "Llama-3.3-70B"


def kv_ratio_mla(md_rows, Lctx):
    """MLA-specific KV ratio. DeepSeek-V4 uses a SINGLE shared latent KV per
    token (physical_kv_heads = 1), not per-head pages. A token's latent must be
    kept if ANY logical head still attends to it; at the decode position head h
    reaches its most-recent eff_h tokens, so the kept set is the UNION over
    heads = max_h(eff_h). A uniform token-block Paged cache keeps the same set
    (block == furthest head). Hence SegPaged cannot drop per-head here and the
    ratio is 1.0 — head-window heterogeneity yields no benefit under MLA."""
    paged = seg = 0
    for layer in md_rows:
        effs = [(Lctx if v < 0 else min(int(v) + SINK, Lctx)) for v in layer]
        paged += max(effs)
        seg += max(effs)  # shared latent: union == max
    return paged, seg


def load_dsv4_mla(Lctx):
    d = json.load(open(HC / "dsv4_extreme_local128.json"))
    md = d["mla_head_max_distance"]  # [L][64] logical-head distances
    pg, sg = kv_ratio_mla(md, Lctx)
    return pg, sg, "DeepSeek-V4-Flash (MLA)"


def load_qwen35_397b():
    """Parameter-style config: expand to per-(layer,head) windows.
    full_attention layers: heads are global/local per frac_global; window for
    local heads = `window`. linear/other layers are not standard KV -> skip."""
    d = json.load(open(HC / "qwen3.5-397B-A17B_redknot.json"))
    Hkv = d.get("num_kv_heads", 2)
    D = d.get("head_dim", 128)
    win = d.get("window", 2048)
    frac_global = d.get("frac_global", 0.1)
    full_layers = set(d.get("full_attention_layers", []))
    nL = d["num_layers"]
    # deterministic head assignment: first ceil(frac_global*Hkv) heads global
    import math

    n_glob = max(0, math.ceil(frac_global * Hkv))
    rows = []
    for li in range(nL):
        if full_layers and li not in full_layers:
            continue  # only full-attention layers carry explicit KV here
        row = []
        for h in range(Hkv):
            row.append("FULL" if h < n_glob else win)
        rows.append(row)
    return rows, Hkv, D, "Qwen3.5-397B-A17B"


def load_qwen3_32b():
    d = json.load(open(HC / "qwen3-32B_optimal_g15_lf_ret.json"))
    Hkv = d["num_kv_heads"]
    D = d.get("head_dim", 128)
    win = d.get("window", 4096)
    cls = d["kv_head_classification"]  # [L][Hkv] of 'global'/'local_full'/'retrieval'
    md = d.get("kv_head_max_distance")
    rows = []
    for li, layer in enumerate(cls):
        row = []
        for h, c in enumerate(layer):
            if c == "global":
                row.append("FULL")
            else:
                # use measured max distance if available, else nominal window
                v = md[li][h] if md else win
                row.append("FULL" if (v is None or v < 0) else min(int(v), win))
        rows.append(row)
    return rows, Hkv, D, "Qwen3-32B"


# standard GQA models use per-head paging; MLA model handled separately
GQA_LOADERS = [load_qwen3_32b, load_qwen35_397b]


def main():
    results = {}
    print(
        f"{'Model':<26} | "
        + " ".join(f"{L // 1000:>5}K" for L in LENGTHS)
        + "   (hetero SegPaged-vs-Paged KV ratio)"
    )
    print("-" * 76)

    # --- standard GQA models ---
    for loader in GQA_LOADERS:
        rows, Hkv, D, name = loader()
        wins = [w for r in rows for w in r if w != "FULL"]
        full_frac = sum(1 for r in rows for w in r if w == "FULL") / max(
            1, sum(len(r) for r in rows)
        )
        ratios = [
            kv_ratio(rows, Hkv, D, L)[0] / kv_ratio(rows, Hkv, D, L)[1] for L in LENGTHS
        ]
        results[name] = {
            "arch": "GQA",
            "Hkv": Hkv,
            "head_dim": D,
            "window_median": int(statistics.median(wins)) if wins else None,
            "full_head_frac": round(full_frac, 4),
            "lengths": LENGTHS,
            "hetero_ratio": [round(r, 3) for r in ratios],
        }
        print(
            f"{name + ' (GQA)':<26} | "
            + " ".join(f"{r:>5.2f}" for r in ratios)
            + f"   (median win={results[name]['window_median']})"
        )

    # --- MLA model (DeepSeek-V4): shared latent => ratio == 1.0 ---
    mla_ratios = []
    for L in LENGTHS:
        pg, sg, mname = load_dsv4_mla(L)
        mla_ratios.append(pg / sg)
    results["DeepSeek-V4-Flash"] = {
        "arch": "MLA",
        "physical_kv_heads": 1,
        "lengths": LENGTHS,
        "hetero_ratio": [round(r, 3) for r in mla_ratios],
        "note": "MLA shares a single latent KV across heads; per-head window "
        "heterogeneity gives no SegPaged benefit (ratio == 1).",
    }
    print(
        f"{'DeepSeek-V4-Flash (MLA)':<26} | "
        + " ".join(f"{r:>5.2f}" for r in mla_ratios)
        + "   (shared latent -> no per-head gain)"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
