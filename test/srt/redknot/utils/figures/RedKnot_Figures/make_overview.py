#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RedKnot system overview – component-architecture style (2.5:1).

v3: less whitespace, fewer black borders, larger mini-diagrams.
"""

from __future__ import annotations
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch, Polygon

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# ── palette ──
RED = "#EFA59C"
BLUE = "#A7B5E0"
ORANGE = "#F4B43C"
GRAY = "#C9CCD2"
LIGHT = "#F5F5F8"
INK = "#111111"
WHITE = "#FFFFFF"

CTRL_BG = "#E8EBF4"
DATA_BG = "#F6F6F9"
STORE_BG = "#E4E5EB"
RED_TINT = "#FDF1EF"
BLUE_TINT = "#EEF0F8"
ORANGE_TINT = "#FEF5E0"

LW_OUT = 2.0  # outermost boxes only
LW_MID = 1.2  # planes
LW_IN = 0.9  # components
LW_THIN = 0.6  # mini rectangles

# ── canvas 2.5 : 1 ──
W, H = 15.0, 6.0
fig, ax = plt.subplots(figsize=(12.5, 5.0), facecolor="white")
ax.set_facecolor("white")
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════


def box(x, y, w, h, fc=WHITE, ec=INK, lw=LW_MID, z=2, ls="-", rounded=False):
    if rounded:
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=z,
            linestyle=ls,
        )
    else:
        p = Rectangle(
            (x, y),
            w,
            h,
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=z,
            linestyle=ls,
        )
    ax.add_patch(p)


def lab(
    x,
    y,
    s,
    size=7.0,
    weight="normal",
    style="normal",
    ha="center",
    va="center",
    color=INK,
    z=10,
    rot=0,
):
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        fontweight=weight,
        fontstyle=style,
        ha=ha,
        va=va,
        color=color,
        zorder=z,
        rotation=rot,
    )


def arrow(p0, p1, lw=1.3, ms=9, color=INK, ls="-", z=8):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=ms,
            lw=lw,
            color=color,
            linestyle=ls,
            zorder=z,
            shrinkA=0,
            shrinkB=0,
        )
    )


def elbow(pts, lw=1.2, ms=8, color=INK, ls="-", z=8):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="-",
                lw=lw,
                color=color,
                linestyle=ls,
                zorder=z,
                shrinkA=0,
                shrinkB=0,
            )
        )
    ax.add_patch(
        FancyArrowPatch(
            pts[-2],
            pts[-1],
            arrowstyle="-|>",
            mutation_scale=ms,
            lw=lw,
            color=color,
            linestyle=ls,
            zorder=z,
            shrinkA=0,
            shrinkB=0,
        )
    )


def badge(x, y, n, r=0.14, size=6.5, z=14):
    ax.add_patch(
        plt.Circle((x, y), r, facecolor="#222", edgecolor=WHITE, lw=0.85, zorder=z)
    )
    ax.text(
        x,
        y,
        str(n),
        fontsize=size,
        fontweight="bold",
        ha="center",
        va="center",
        color="white",
        zorder=z + 1,
    )


def srect(x, y, w, h, fc, ec=INK, lw=LW_THIN, z=6):
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    )


def prow(x, y, n, w, h, gap, colors, z=6):
    if isinstance(colors, str):
        colors = [colors] * n
    for i in range(n):
        srect(x + i * (w + gap), y, w, h, colors[i % len(colors)], z=z)


def doc_icon(cx, cy, sc=1.0, z=5):
    w, h = 0.42 * sc, 0.56 * sc
    x0, y0 = cx - w / 2, cy - h / 2
    fold = 0.11 * sc
    ax.add_patch(
        Polygon(
            [
                (x0, y0),
                (x0 + w, y0),
                (x0 + w, y0 + h - fold),
                (x0 + w - fold, y0 + h),
                (x0, y0 + h),
            ],
            closed=True,
            facecolor=WHITE,
            edgecolor=INK,
            linewidth=0.9,
            zorder=z,
        )
    )
    ax.add_patch(
        Polygon(
            [
                (x0 + w - fold, y0 + h),
                (x0 + w, y0 + h - fold),
                (x0 + w - fold, y0 + h - fold),
            ],
            closed=True,
            facecolor=LIGHT,
            edgecolor=INK,
            linewidth=0.6,
            zorder=z + 1,
        )
    )
    for i in range(3):
        ly = y0 + h - 0.18 * sc - i * 0.12 * sc
        ax.plot(
            [x0 + 0.07 * sc, x0 + w - 0.10 * sc],
            [ly, ly],
            color=GRAY,
            lw=0.7,
            zorder=z + 2,
        )


def chip_icon(cx, cy, sc=1.0, z=5):
    w, h = 0.50 * sc, 0.38 * sc
    x0, y0 = cx - w / 2, cy - h / 2
    box(x0, y0, w, h, fc=LIGHT, ec=INK, lw=0.9, z=z)
    dw, dh = 0.20 * sc, 0.14 * sc
    box(cx - dw / 2, cy - dh / 2, dw, dh, fc=BLUE, ec=INK, lw=0.6, z=z + 1)
    pin = 0.06 * sc
    for i in range(3):
        py = y0 + 0.07 * sc + i * 0.11 * sc
        ax.plot([x0 - pin, x0], [py, py], color=INK, lw=0.65, zorder=z)
        ax.plot([x0 + w, x0 + w + pin], [py, py], color=INK, lw=0.65, zorder=z)
    for i in range(2):
        px = x0 + 0.13 * sc + i * 0.22 * sc
        ax.plot([px, px], [y0 - pin, y0], color=INK, lw=0.65, zorder=z)
        ax.plot([px, px], [y0 + h, y0 + h + pin], color=INK, lw=0.65, zorder=z)


# ═══════════════════════════════════════════════════════════
# 1  LEFT COLUMN  (narrower, tighter)
# ═══════════════════════════════════════════════════════════

left_w = 2.65

# ── ❶ Offline Profiling ──
off_x, off_y = 0.15, 3.60
off_w, off_h = left_w, 2.20
box(off_x, off_y, off_w, off_h, fc=LIGHT, ec=INK, lw=LW_OUT, z=1)
lab(
    off_x + off_w / 2,
    off_y + off_h - 0.22,
    "Offline Profiling",
    size=7.8,
    weight="bold",
)
badge(off_x + off_w - 0.20, off_y + off_h - 0.20, 1)

# document icon (larger)
doc_icon(0.80, 4.65, sc=1.2, z=3)
lab(0.80, 4.00, "Documents", size=5.8)

# head map
hm_x, hm_y = 1.62, 4.12
box(hm_x, hm_y, 1.00, 0.75, fc=WHITE, ec=GRAY, lw=LW_IN, z=3)
lab(hm_x + 0.50, hm_y + 0.55, r"$M(\ell,h)$", size=6.5, weight="bold")
for i, c in enumerate([RED, BLUE, BLUE, RED]):
    srect(hm_x + 0.08 + i * 0.22, hm_y + 0.10, 0.18, 0.18, c, z=4)
lab(hm_x + 0.50, hm_y - 0.14, "Head Map", size=5.8)


# ── ❷ Online Request ──
on_x, on_y = 0.15, 0.20
on_w, on_h = left_w, 2.95
box(on_x, on_y, on_w, on_h, fc=LIGHT, ec=INK, lw=LW_OUT, z=1)
lab(on_x + on_w / 2, on_y + on_h - 0.22, "Online Request", size=7.8, weight="bold")
badge(on_x + on_w - 0.20, on_y + on_h - 0.20, 2)

lab(on_x + on_w / 2, 2.48, r"prefix $P$  +  chunk $C$", size=5.8, style="italic")

# token blocks — LARGER
bw, bh, bg = 0.24, 0.28, 0.06
prow(0.38, 1.92, 3, bw, bh, bg, GRAY, z=3)
lab(0.38 + 1.5 * (bw + bg), 1.78, "prefix", size=5.0, style="italic")
prow(1.38, 1.92, 5, bw, bh, bg, [RED, BLUE, BLUE, RED, BLUE], z=3)
lab(1.38 + 2.5 * (bw + bg), 1.78, "chunk (cached)", size=5.0, style="italic")

# RoPE Align — no extra border, just tinted background
rope_x, rope_y = 0.35, 0.42
rope_w, rope_h = left_w - 0.40, 1.05
box(
    rope_x, rope_y, rope_w, rope_h, fc=ORANGE_TINT, ec=GRAY, lw=LW_IN, z=3, rounded=True
)
lab(rope_x + rope_w / 2, rope_y + 0.72, "RoPE Align", size=7.0, weight="bold")
lab(
    rope_x + rope_w / 2,
    rope_y + 0.38,
    r"$K(p_{on}) = R(p_{on})R(p_{off})^{-1}K(p_{off})$",
    size=5.5,
)


# ═══════════════════════════════════════════════════════════
# 2  CENTRAL – RedKnot Runtime  (wider, taller, less padding)
# ═══════════════════════════════════════════════════════════

rt_x = on_x + on_w + 0.35
rt_y = 0.12
rt_w = 15.0 - rt_x - 1.95  # leave room for output
rt_h = 5.76
# No black border — use a subtle dark-gray border for the outermost runtime
box(rt_x, rt_y, rt_w, rt_h, fc=WHITE, ec="#444444", lw=LW_OUT + 0.6, z=0)
lab(rt_x + rt_w / 2, rt_y + rt_h - 0.24, "RedKnot Runtime", size=9.8, weight="bold")
lab(
    rt_x + rt_w / 2,
    rt_y + rt_h - 0.54,
    "Elastic Sparsity  +  SegPagedAttention",
    size=5.8,
    style="italic",
)
badge(rt_x + rt_w - 0.22, rt_y + rt_h - 0.22, 3)

pad = 0.18  # internal padding

# ── Control Plane (no black border, just background) ──
cp_x = rt_x + pad
cp_y = rt_y + rt_h - 1.68
cp_w = rt_w - 2 * pad
cp_h = 0.86
box(cp_x, cp_y, cp_w, cp_h, fc=CTRL_BG, ec=GRAY, lw=LW_IN, z=2)
lab(
    cp_x + 0.06,
    cp_y + cp_h - 0.13,
    "Control Plane",
    size=5.0,
    weight="bold",
    ha="left",
    color="#555555",
)

# Elastic Sparsity Controller — no border, just white fill
esc_x = cp_x + 0.80
esc_w = 3.60
esc_h = cp_h - 0.22
box(esc_x, cp_y + 0.11, esc_w, esc_h, fc=WHITE, ec=GRAY, lw=LW_IN, z=4, rounded=True)
chip_icon(esc_x + 0.40, cp_y + 0.11 + esc_h / 2, sc=0.78, z=5)
lab(
    esc_x + 2.15,
    cp_y + 0.11 + esc_h / 2 + 0.10,
    "Elastic Sparsity Controller",
    size=6.5,
    weight="bold",
)
lab(
    esc_x + 2.15,
    cp_y + 0.11 + esc_h / 2 - 0.14,
    r"queries $M(\ell,h)$  ·  selects recovery policy",
    size=4.8,
    style="italic",
)

# Layer-wise Policy
lp_x = cp_x + cp_w - 2.95
lp_w = 2.75
box(lp_x, cp_y + 0.11, lp_w, esc_h, fc=WHITE, ec=GRAY, lw=LW_IN, z=4, rounded=True)
lab(
    lp_x + lp_w / 2,
    cp_y + 0.11 + esc_h / 2 + 0.10,
    "Layer-wise Policy",
    size=6.3,
    weight="bold",
)
lab(
    lp_x + lp_w / 2,
    cp_y + 0.11 + esc_h / 2 - 0.14,
    "shallow → dense FFN  ·  deep → sparse FFN",
    size=4.6,
    style="italic",
)


# ── Data Plane ──
dp_x = rt_x + pad
dp_y = rt_y + 1.68
dp_w = rt_w - 2 * pad
dp_h = 2.26
box(dp_x, dp_y, dp_w, dp_h, fc=DATA_BG, ec=GRAY, lw=LW_IN, z=2)
lab(
    dp_x + 0.06,
    dp_y + dp_h - 0.13,
    "Data Plane",
    size=5.0,
    weight="bold",
    ha="left",
    color="#555555",
)

comp_y0 = dp_y + 0.14
comp_h = dp_h - 0.32
comp_start = dp_x + 0.65
comp_gap = 0.18

# ── Component A: SegPagedAttention ──
cA_w = 2.35
box(
    comp_start,
    comp_y0,
    cA_w,
    comp_h,
    fc=BLUE_TINT,
    ec=GRAY,
    lw=LW_IN,
    z=4,
    rounded=True,
)
lab(
    comp_start + cA_w / 2,
    comp_y0 + comp_h - 0.20,
    "SegPagedAttention",
    size=6.8,
    weight="bold",
)
lab(
    comp_start + cA_w / 2,
    comp_y0 + comp_h - 0.50,
    r"$(\ell,h,s)$ → virtual pages",
    size=5.2,
    style="italic",
)
# LARGER mini diagram
sw, sh, sg = 0.24, 0.28, 0.05
seg_y = comp_y0 + 0.72
prow(comp_start + 0.18, seg_y, 3, sw, sh, sg, [RED, BLUE, BLUE], z=6)
arrow(
    (comp_start + 0.18 + 3 * (sw + sg) + 0.02, seg_y + sh / 2),
    (comp_start + 0.18 + 3 * (sw + sg) + 0.28, seg_y + sh / 2),
    lw=1.0,
    ms=7,
)
prow(comp_start + 0.18 + 3 * (sw + sg) + 0.36, seg_y, 3, 0.20, sh, 0.04, WHITE, z=6)
lab(
    comp_start + cA_w / 2,
    comp_y0 + 0.28,
    "fused varlen kernel",
    size=5.0,
    style="italic",
)

# ── Component B: Head Recovery ──
cB_x = comp_start + cA_w + comp_gap
cB_w = 2.48
box(cB_x, comp_y0, cB_w, comp_h, fc=RED_TINT, ec=GRAY, lw=LW_IN, z=4, rounded=True)
lab(cB_x + cB_w / 2, comp_y0 + comp_h - 0.20, "Head Recovery", size=6.8, weight="bold")
lab(
    cB_x + cB_w / 2,
    comp_y0 + comp_h - 0.48,
    "head-aware attention",
    size=5.2,
    style="italic",
)
# LARGER global + local boxes
gb_w, gb_h = 0.88, 0.62
box(cB_x + 0.20, comp_y0 + 0.52, gb_w, gb_h, fc=RED, ec=INK, lw=LW_IN, z=6)
lab(
    cB_x + 0.20 + gb_w / 2,
    comp_y0 + 0.52 + gb_h / 2 + 0.08,
    "global",
    size=6.0,
    weight="bold",
    color=WHITE,
)
lab(
    cB_x + 0.20 + gb_w / 2,
    comp_y0 + 0.52 + gb_h / 2 - 0.14,
    "recompute",
    size=4.5,
    style="italic",
    color=WHITE,
)
box(cB_x + 1.35, comp_y0 + 0.52, gb_w, gb_h, fc=BLUE, ec=INK, lw=LW_IN, z=6)
lab(
    cB_x + 1.35 + gb_w / 2,
    comp_y0 + 0.52 + gb_h / 2 + 0.08,
    "local",
    size=6.0,
    weight="bold",
    color=WHITE,
)
lab(
    cB_x + 1.35 + gb_w / 2,
    comp_y0 + 0.52 + gb_h / 2 - 0.14,
    "reuse",
    size=4.5,
    style="italic",
    color=WHITE,
)
lab(cB_x + cB_w / 2, comp_y0 + 0.22, r"sink $\cup$ window", size=5.0, style="italic")

# ── Component C: Sparse FFN ──
cC_x = cB_x + cB_w + comp_gap
cC_w = 1.88
box(cC_x, comp_y0, cC_w, comp_h, fc=ORANGE_TINT, ec=GRAY, lw=LW_IN, z=4, rounded=True)
lab(cC_x + cC_w / 2, comp_y0 + comp_h - 0.20, "Sparse FFN", size=6.8, weight="bold")
lab(
    cC_x + cC_w / 2,
    comp_y0 + comp_h - 0.48,
    "token-level sparsity",
    size=5.2,
    style="italic",
)
# LARGER token blocks
prow(
    cC_x + 0.18,
    comp_y0 + 0.72,
    5,
    0.22,
    0.28,
    0.05,
    [ORANGE, GRAY, ORANGE, GRAY, ORANGE],
    z=6,
)
lab(cC_x + cC_w / 2, comp_y0 + 0.46, "top-k → FFN", size=5.0, weight="bold")
lab(cC_x + cC_w / 2, comp_y0 + 0.22, "rest → residual", size=5.0, style="italic")

# arrows between data-plane components
arrow(
    (comp_start + cA_w, comp_y0 + comp_h / 2),
    (cB_x, comp_y0 + comp_h / 2),
    lw=1.4,
    ms=9,
)
arrow((cB_x + cB_w, comp_y0 + comp_h / 2), (cC_x, comp_y0 + comp_h / 2), lw=1.4, ms=9)


# ── Storage Plane ──
sp_x = rt_x + pad
sp_y = rt_y + 0.14
sp_w = rt_w - 2 * pad
sp_h = 1.22
box(sp_x, sp_y, sp_w, sp_h, fc=STORE_BG, ec=GRAY, lw=LW_IN, z=2)
lab(
    sp_x + 0.06,
    sp_y + sp_h - 0.13,
    "Storage",
    size=5.0,
    weight="bold",
    ha="left",
    color="#555555",
)

lab(
    sp_x + sp_w / 2,
    sp_y + sp_h - 0.16,
    "Head-Segmented KV Store  (HBM)",
    size=6.8,
    weight="bold",
)

# LARGER page grids
pw2, ph2, pg2 = 0.22, 0.20, 0.05

# global pages (left half)
gp_x0 = sp_x + 0.55
gp_y0 = sp_y + sp_h - 0.50
prow(gp_x0, gp_y0, 7, pw2, ph2, pg2, RED_TINT, z=4)
prow(gp_x0, gp_y0 - ph2 - pg2, 7, pw2, ph2, pg2, RED_TINT, z=4)
lab(
    gp_x0 + 7 * (pw2 + pg2) / 2,
    sp_y + sp_h - 0.32,
    "global pages",
    size=5.0,
    weight="bold",
)

# separator
sep_x2 = gp_x0 + 7 * (pw2 + pg2) + 0.10
ax.plot(
    [sep_x2, sep_x2],
    [sp_y + 0.08, sp_y + sp_h - 0.28],
    color=INK,
    lw=0.8,
    ls="--",
    zorder=5,
)

# local pages (right half)
lp_x0 = sep_x2 + 0.12
prow(lp_x0, gp_y0, 7, pw2, ph2, pg2, BLUE_TINT, z=4)
prow(lp_x0, gp_y0 - ph2 - pg2, 7, pw2, ph2, pg2, BLUE_TINT, z=4)
lab(
    lp_x0 + 7 * (pw2 + pg2) / 2,
    sp_y + sp_h - 0.32,
    "local pages",
    size=5.0,
    weight="bold",
)

lab(
    sp_x + sp_w / 2,
    sp_y + 0.06,
    "virtual page indirection  ·  paged KV allocator",
    size=4.6,
    style="italic",
)


# ── vertical arrows: data ↔ storage ──
spa_cx = comp_start + cA_w / 2
arrow((spa_cx - 0.15, comp_y0), (spa_cx - 0.15, sp_y + sp_h), lw=1.0, ms=7)
arrow((spa_cx + 0.15, sp_y + sp_h), (spa_cx + 0.15, comp_y0), lw=1.0, ms=7)

hr_cx = cB_x + cB_w / 2
arrow((hr_cx, sp_y + sp_h), (hr_cx, comp_y0), lw=1.0, ms=7)

# ── control → data (dashed policy arrows) ──
arrow(
    (esc_x + esc_w / 2, cp_y),
    (comp_start + cA_w / 2, dp_y + dp_h),
    lw=0.85,
    ms=6,
    ls=(0, (3, 2)),
)
arrow((lp_x + 0.6, cp_y), (cB_x + cB_w / 2, dp_y + dp_h), lw=0.85, ms=6, ls=(0, (3, 2)))
arrow(
    (lp_x + lp_w - 0.4, cp_y),
    (cC_x + cC_w / 2, dp_y + dp_h),
    lw=0.85,
    ms=6,
    ls=(0, (3, 2)),
)


# ═══════════════════════════════════════════════════════════
# 3  INTER-COLUMN ARROWS
# ═══════════════════════════════════════════════════════════

# Offline → Control plane
elbow(
    [
        (off_x + off_w, 4.65),
        (rt_x + 0.06, 4.65),
        (rt_x + 0.06, cp_y + cp_h / 2),
        (cp_x, cp_y + cp_h / 2),
    ],
    lw=1.0,
    ms=7,
    ls=(0, (4, 2)),
)
lab(off_x + off_w + 0.12, 4.82, r"$M(\ell,h)$", size=5.2, style="italic", ha="left")

# Online → Data plane (main)
mid_arrow_y = dp_y + dp_h / 2
arrow((on_x + on_w, mid_arrow_y), (rt_x, mid_arrow_y), lw=1.5, ms=10)
lab(
    (on_x + on_w + rt_x) / 2,
    mid_arrow_y + 0.18,
    "aligned KV + query",
    size=5.2,
    style="italic",
)

# Online → Storage (position metadata)
arrow((on_x + on_w, 0.95), (rt_x, 0.95), lw=1.0, ms=7, ls=(0, (3, 2)))
lab((on_x + on_w + rt_x) / 2, 1.12, "position metadata", size=4.8, style="italic")


# ═══════════════════════════════════════════════════════════
# 4  OUTPUT (right side, compact)
# ═══════════════════════════════════════════════════════════

out_x = rt_x + rt_w + 0.20
out_y = 1.70
out_w = 1.32
out_h = 2.20
box(out_x, out_y, out_w, out_h, fc=LIGHT, ec=INK, lw=LW_OUT, z=1)
lab(out_x + out_w / 2, out_y + out_h - 0.30, "Recovered", size=6.8, weight="bold")
lab(out_x + out_w / 2, out_y + out_h / 2, "Hidden", size=6.5, weight="bold")
lab(out_x + out_w / 2, out_y + out_h / 2 - 0.25, "States", size=6.5, weight="bold")
lab(out_x + out_w / 2, out_y + 0.30, r"$X_{\ell+1}$", size=6.2, style="italic")
badge(out_x + out_w - 0.16, out_y + out_h - 0.16, 4)

arrow((rt_x + rt_w, mid_arrow_y), (out_x, mid_arrow_y), lw=1.5, ms=10)

# Y
y_x = out_x + out_w + 0.18
y_y = 2.25
y_w, y_h = 0.58, 0.90
box(y_x, y_y, y_w, y_h, fc=WHITE, ec=INK, lw=LW_OUT, z=1)
lab(y_x + y_w / 2, y_y + y_h / 2, "Y", size=9.0, weight="bold")
badge(y_x + y_w - 0.12, y_y + y_h - 0.12, 5)
arrow((out_x + out_w, out_y + out_h / 2), (y_x, y_y + y_h / 2), lw=1.3, ms=8)


# ═══════════════════════════════════════════════════════════
# 5  SAVE
# ═══════════════════════════════════════════════════════════

fig.subplots_adjust(left=0.003, right=0.997, top=0.997, bottom=0.003)
fig.savefig("OverView.pdf", facecolor="white")
fig.savefig("OverView.png", dpi=240, facecolor="white")
plt.close(fig)
print("saved OverView.pdf + OverView.png")
