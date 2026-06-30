#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RedKnot Figure 1 (teaser / overview) -- SINGLE-COLUMN version.

Vertical, stacked layout sized for \columnwidth (tall-narrow aspect).
Content is faithful to the paper:

  RedKnot decouples the KV cache along KV heads, classifies heads into
  global / local, and co-optimizes three ORTHOGONAL mechanisms:
    (A) Head-class sparse attention  [heads]   global->recompute, local->reuse
    (B) SegPagedAttention            [storage] head segments -> pages, varlen
    (C) Sparse FFN                   [tokens]  top-k -> dense FFN, else residual
  Combined: 1.6-3.5x lower TTFT, 4.7-7.8x more concurrency, 67-79% fewer FLOPs.

Design goals for this revision:
  * single-column friendly (narrow, taller than wide)
  * minimal whitespace, fewer unit blocks
  * all connectors orthogonal (horizontal / vertical only)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

RED = "#EFA59C"
BLUE = "#A7B5E0"
GRAY = "#C9CCD2"
SEL = "#F4B43C"
INK = "#000000"
R_E = "#B23A2E"
B_E = "#3B4A86"
O_E = "#B8810A"

LW_OUT = 2.0
LW_MID = 1.5
LW_IN = 1.1


def box(ax, x, y, w, h, fc="white", ec=INK, lw=LW_MID, z=2):
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    )


def lab(
    ax,
    x,
    y,
    s,
    size=9,
    weight="normal",
    style="normal",
    color=INK,
    ha="center",
    va="center",
    z=8,
    rot=0,
):
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        fontweight=weight,
        fontstyle=style,
        color=color,
        ha=ha,
        va=va,
        zorder=z,
        rotation=rot,
    )


def row(ax, x, y, n, uw, uh, gap, fcs, ec=INK, lw=LW_IN, z=6):
    if not isinstance(fcs, (list, tuple)):
        fcs = [fcs] * n
    for c in range(n):
        box(ax, x + c * (uw + gap), y, uw, uh, fc=fcs[c], ec=ec, lw=lw, z=z)


def varrow(ax, x, y0, y1, color=INK, lw=2.0, ms=12, ls="-", z=7):
    ax.add_patch(
        FancyArrowPatch(
            (x, y0),
            (x, y1),
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


def harrow(ax, x0, x1, y, color=INK, lw=2.0, ms=12, ls="-", z=7):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
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


def varrow_open(ax, x, y0, y1, color=INK, lw=1.6, ms=20, z=7):
    """Open (hollow, white-filled head) vertical arrow -- marks non-flow
    transitions (entry into the pipeline / final aggregated output)."""
    ax.add_patch(
        FancyArrowPatch(
            (x, y0),
            (x, y1),
            arrowstyle="-|>",
            mutation_scale=ms,
            lw=lw,
            edgecolor=color,
            facecolor="white",
            zorder=z,
            shrinkA=0,
            shrinkB=0,
        )
    )


def badge(ax, x, y, n, r=0.17, size=8.0, z=14, fc="#222"):
    ax.add_patch(
        plt.Circle((x, y), r, facecolor=fc, edgecolor="white", lw=1.0, zorder=z)
    )
    lab(ax, x, y, str(n), size=size, weight="bold", color="white", z=z + 1)


# ---------------------------------------------------------------------------
# Canvas: narrow + tall  (aspect ~ 1 : 1.28), single-column
# ---------------------------------------------------------------------------
W, H = 10.0, 12.8
fig, ax = plt.subplots(figsize=(5.0, 6.4))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")

M = 0.25
FW = W - 2 * M  # full content width
CX = W / 2

# ===========================================================================
# SECTION 1 (top): Decoupled KV heads -> Global / Local classification
# ===========================================================================
s1y, s1h = 10.05, 2.45
box(ax, M, s1y, FW, s1h, lw=LW_OUT)
badge(ax, M + FW - 0.28, s1y + s1h - 0.28, 1)
lab(
    ax,
    CX,
    s1y + s1h - 0.27,
    "KV Cache decoupled along heads, classified offline",
    size=8.8,
    weight="bold",
)

# 4 head tiles (reduced count)
nh = 4
hw, hg = 1.35, 0.22
hx0 = M + 0.55
hy = s1y + 1.18
hh = 0.62
grays = ["#3a3a3a", "#5f5f5f", "#8a8a8a", "#b3b3b3"]
for i in range(nh):
    box(ax, hx0 + i * (hw + hg), hy, hw, hh, fc=grays[i], lw=LW_IN, z=6)
    lab(
        ax,
        hx0 + i * (hw + hg) + hw / 2,
        hy + hh / 2,
        f"$h_{{{i}}}$",
        size=9,
        weight="bold",
        color=("white" if i < 3 else "#1a1a1a"),
        z=7,
    )

# classification labels under tiles: Global (red block) vs Local (blue block)
cy = s1y + 0.5
lab(ax, M + 0.95, cy + 0.2, "Global", size=8.5, weight="bold")
box(ax, M + 1.85, cy, 0.55, 0.4, fc=RED, lw=LW_IN, z=6)
lab(ax, M + 2.62, cy + 0.2, "12–15%", size=8, weight="bold", ha="left")
lab(ax, M + 4.85, cy + 0.2, "Local", size=8.5, weight="bold")
row(ax, M + 5.75, cy, 3, 0.55, 0.4, 0.12, BLUE, z=6)
lab(
    ax,
    M + 5.75 + 3 * 0.67 + 0.05,
    cy + 0.2,
    "85–88%",
    size=8,
    weight="bold",
    ha="left",
)

# ===========================================================================
# SECTION 2 (middle): three orthogonal mechanisms, STACKED vertically
# ===========================================================================
mech_x = M
mech_w = FW
mh = 1.78  # height of each mechanism band
mg = 0.42  # vertical gap between bands
# top of first band
b_top = 9.05

# ---- (A) Head-class Sparse Attention ----
Ay = b_top - mh
box(ax, mech_x, Ay, mech_w, mh, lw=LW_OUT)
badge(ax, mech_x + mech_w - 0.28, Ay + mh - 0.28, 2)
lab(ax, mech_x + 0.18, Ay + mh - 0.28, "(A)", size=9.5, weight="bold", ha="left")
lab(
    ax,
    mech_x + 0.95,
    Ay + mh - 0.28,
    "Head-class Sparse Attention",
    size=9,
    weight="bold",
    ha="left",
)
lab(
    ax,
    mech_w + M - 0.62,
    Ay + mh - 0.28,
    "head-level",
    size=7.5,
    style="italic",
    ha="right",
)
# global -> recompute  (left)
gax = mech_x + 0.55
lab(ax, gax + 0.95, Ay + 0.95, "Global", size=8, weight="bold")
row(ax, gax, Ay + 0.35, 3, 0.5, 0.4, 0.1, RED, z=6)
lab(ax, gax + 0.85, Ay + 0.18, "recompute", size=7, style="italic")
# arrow ->
harrow(ax, gax + 2.05, gax + 2.95, Ay + 0.55, color="#666", lw=1.8, ms=11)
# local -> reuse window (right)
lax = gax + 3.25
lab(ax, lax + 1.4, Ay + 0.95, "Local: reuse + window", size=8, weight="bold")
row(ax, lax, Ay + 0.35, 2, 0.5, 0.4, 0.1, BLUE, z=6)  # sink
row(ax, lax + 2 * 0.6 + 0.1, Ay + 0.35, 2, 0.5, 0.4, 0.1, GRAY, z=6)  # reuse
row(ax, lax + 4 * 0.6 + 0.2, Ay + 0.35, 1, 0.5, 0.4, 0.1, BLUE, z=6)  # win
lab(
    ax,
    lax + 1.55,
    Ay + 0.18,
    r"$W(i)=S_{\mathrm{sink}}\!\cup\![i\!-\!w,i]$",
    size=6.8,
)

# ---- (B) SegPagedAttention ----
By = Ay - mg - mh
box(ax, mech_x, By, mech_w, mh, lw=LW_OUT)
badge(ax, mech_x + mech_w - 0.28, By + mh - 0.28, 3)
lab(ax, mech_x + 0.18, By + mh - 0.28, "(B)", size=9.5, weight="bold", ha="left")
lab(
    ax,
    mech_x + 0.95,
    By + mh - 0.28,
    "SegPagedAttention",
    size=9,
    weight="bold",
    ha="left",
)
lab(
    ax,
    mech_w + M - 0.62,
    By + mh - 0.28,
    "storage-level",
    size=7.5,
    style="italic",
    ha="right",
)
# head segments -> virtual pages -> physical pages (horizontal pipeline)
seg_x = mech_x + 0.55
lab(ax, seg_x + 0.95, By + 0.95, "segments $(\\ell,h,s)$", size=7.6, weight="bold")
row(ax, seg_x, By + 0.4, 3, 0.55, 0.42, 0.1, [RED, BLUE, BLUE], z=6)
harrow(ax, seg_x + 2.05, seg_x + 2.75, By + 0.61, color="#666", lw=1.8, ms=10)
# virtual pages
vp_x = seg_x + 3.0
lab(ax, vp_x + 0.85, By + 0.95, "virtual pages", size=7.6, weight="bold")
row(ax, vp_x, By + 0.4, 3, 0.5, 0.42, 0.1, "white", z=6)
harrow(ax, vp_x + 1.9, vp_x + 2.55, By + 0.61, color="#666", lw=1.8, ms=10)
# physical pages (gray)
pp_x = vp_x + 2.8
box(ax, pp_x - 0.12, By + 0.32, mech_w + M - pp_x - 0.05, 0.95, fc=GRAY, lw=LW_MID, z=4)
lab(ax, pp_x + 0.55, By + 1.02, "HBM", size=7.4, weight="bold", z=7)
row(ax, pp_x + 0.05, By + 0.42, 3, 0.4, 0.42, 0.08, "white", z=6)
lab(
    ax,
    CX,
    By + 0.16,
    "fused varlen kernel · FlashAttention fast path (no attn\\_mask)",
    size=6.6,
    style="italic",
)

# ---- (C) Sparse FFN ----
Cy = By - mg - mh
box(ax, mech_x, Cy, mech_w, mh, lw=LW_OUT)
badge(ax, mech_x + mech_w - 0.28, Cy + mh - 0.28, 4)
lab(ax, mech_x + 0.18, Cy + mh - 0.28, "(C)", size=9.5, weight="bold", ha="left")
lab(ax, mech_x + 0.95, Cy + mh - 0.28, "Sparse FFN", size=9, weight="bold", ha="left")
lab(
    ax,
    mech_w + M - 0.62,
    Cy + mh - 0.28,
    "token-level",
    size=7.5,
    style="italic",
    ha="right",
)
# tokens row (4): selected vs skipped
tk_x = mech_x + 0.55
toks = [SEL, GRAY, SEL, GRAY]
lab(ax, tk_x + 0.95, Cy + 1.02, "tokens", size=8, weight="bold")
row(ax, tk_x, Cy + 0.45, 4, 0.5, 0.42, 0.1, toks, z=6)
lab(ax, tk_x + 0.95, Cy + 0.27, "top-k / skip", size=6.6, style="italic")

# two outcome boxes, stacked; widened to fill the band and exceed text length.
# centers are placed symmetrically about the token-row midline (jy) so the
# fork arrows are vertically symmetric.
fw_, fh_ = 3.85, 0.5
fx = mech_w + M - 0.2 - fw_  # right-align to band, small right margin
jy = Cy + 0.66  # token-row midline (split point)
d_sym = 0.40  # half vertical separation between the two box centers
cy_hi = jy + d_sym  # Dense FFN box center
cy_lo = jy - d_sym  # Residual identity box center
fy_hi = cy_hi - fh_ / 2
fy_lo = cy_lo - fh_ / 2
box(ax, fx, fy_hi, fw_, fh_, fc="#FFF6E6", ec=O_E, lw=LW_MID, z=5)
lab(
    ax,
    fx + fw_ / 2,
    cy_hi,
    "Dense FFN  (top-k tokens)",
    size=7.4,
    weight="bold",
    z=7,
)
box(ax, fx, fy_lo, fw_, fh_, fc=GRAY, ec=INK, lw=LW_MID, z=5)
lab(
    ax,
    fx + fw_ / 2,
    cy_lo,
    "Residual identity  (FFN = 0)",
    size=7.4,
    weight="bold",
    z=7,
)

# orthogonal split: tokens -> junction -> two boxes (symmetric about jy)
jx = tk_x + 2.05 + 0.45  # junction x (right of token row)
ax.plot([tk_x + 2.05, jx], [jy, jy], color="#666", lw=1.8, zorder=7)  # stub
ax.plot([jx, jx], [cy_lo, cy_hi], color="#666", lw=1.8, zorder=7)  # riser
harrow(ax, jx, fx - 0.02, cy_hi, color="#666", lw=1.8, ms=10)
harrow(ax, jx, fx - 0.02, cy_lo, color="#666", lw=1.8, ms=10)

# vertical arrows between bands  (orthogonal)
# entry into the pipeline (non-flow) -> open/hollow arrow
varrow_open(ax, CX, s1y - 0.02, b_top + 0.02, color=INK, lw=1.8, ms=22)
# pipeline flow A -> B -> C -> solid arrows
varrow(ax, CX, Ay - 0.02, By + mh + 0.02, color=INK, lw=1.8, ms=12)
varrow(ax, CX, By - 0.02, Cy + mh + 0.02, color=INK, lw=1.8, ms=12)

# ===========================================================================
# SECTION 3 (bottom): combined results vs dense
# ===========================================================================
r_h = 1.55
ry = M + 0.02
box(ax, M, ry, FW, r_h, fc="#FAFAFC", lw=LW_OUT)
badge(ax, M + FW - 0.28, ry + r_h - 0.28, 5)
lab(
    ax,
    CX,
    ry + r_h - 0.27,
    "Orthogonal axes compose multiplicatively  vs. dense",
    size=8.2,
    weight="bold",
)
r1 = M + FW * 0.5 / 3
r2 = M + FW * 1.5 / 3
r3 = M + FW * 2.5 / 3
yy = ry + 0.62
for cx, big, small, col in [
    (r1, r"$\mathbf{1.6\text{-}3.5\times}$", "TTFT", R_E),
    (r2, r"$\mathbf{4.7\text{-}7.8\times}$", "concurrency", B_E),
    (r3, r"$\mathbf{67\text{-}79\%}$", "FLOPs", O_E),
]:
    lab(ax, cx, yy + 0.18, big, size=13, color=col, z=7)
    lab(ax, cx, yy - 0.42, small, size=8.5, weight="bold", z=7)

# aggregated output to results (non-flow) -> open/hollow arrow (same as entry)
varrow_open(ax, CX, Cy - 0.02, ry + r_h + 0.02, color=INK, lw=1.8, ms=22)

# ===========================================================================
# (caption removed -- the LaTeX \caption{} will be used instead)
# ===========================================================================
fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)

out_pdf = "Introduce.pdf"
out_png = "Introduce_preview.png"
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
fig.savefig(out_png, dpi=240, bbox_inches="tight", pad_inches=0.03)
print("saved", out_pdf, out_png)
