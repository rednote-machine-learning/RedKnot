#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RedKnot Figure 6 – Elastic Sparsity WorkFlow (Method.pdf).

v7 fixes:
  - Layer labels pulled away from left bracket
  - Ellipsis lowered + bolder
  - Badges near their arrows/boxes
  - Dense FFN → gray fill (not orange)
  - All arrows use orthogonal elbow lines, never overlap boxes
"""

from __future__ import annotations
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
ORANGE = "#F4B43C"
GRAY = "#C9CCD2"
LIGHT = "#FAFAFC"
INK = "#111111"
EDGE = "#8F949E"

LW_OUT = 2.0
LW_MID = 1.4

W, H = 13.0, 12.0
fig, ax = plt.subplots(figsize=(6.5, 6.0))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")


def box(x, y, w, h, fc="white", ec=EDGE, lw=LW_MID, z=2):
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    )


def lab(
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
    )


def badge(x, y, n, r=0.17, size=8.0, z=14):
    ax.add_patch(
        plt.Circle((x, y), r, facecolor="#222", edgecolor="white", lw=1.0, zorder=z)
    )
    lab(x, y, str(n), size=size, weight="bold", color="white", z=z + 1)


def varrow(x, y0, y1, lw=1.6, ms=11, color=INK, z=7):
    ax.add_patch(
        FancyArrowPatch(
            (x, y0),
            (x, y1),
            arrowstyle="-|>",
            mutation_scale=ms,
            lw=lw,
            color=color,
            zorder=z,
            shrinkA=0,
            shrinkB=0,
        )
    )


def harrow(x0, x1, y, lw=1.4, ms=9, color=INK, z=7):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle="-|>",
            mutation_scale=ms,
            lw=lw,
            color=color,
            zorder=z,
            shrinkA=0,
            shrinkB=0,
        )
    )


def elbow_arrow(pts, lw=1.4, ms=9, color=INK, z=7):
    """Orthogonal polyline, arrow head on last segment only."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.add_patch(
            FancyArrowPatch(
                a, b, arrowstyle="-", lw=lw, color=color, zorder=z, shrinkA=0, shrinkB=0
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
            zorder=z,
            shrinkA=0,
            shrinkB=0,
        )
    )


def line_seg(p0, p1, lw=1.4, color=INK, z=7):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, zorder=z)


# ═══════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════
kv_w, kv_h = 4.50, 0.72
kv_x = 2.20  # KV left edge (pulled right to leave room for bracket+label)

ffn_w, ffn_h = 3.60, 0.65
ffn_x = kv_x + kv_w + 0.60  # FFN left edge (well to the right)

kv_cx = kv_x + kv_w / 2
ffn_cx = ffn_x + ffn_w / 2

# right-side gutter for elbow lines (between KV right edge and FFN left edge)
gutter_x = kv_x + kv_w + 0.30  # vertical run x for KV→FFN elbows

# left-side gutter for return elbows (FFN→next KV)
return_x = ffn_x + ffn_w + 0.35  # vertical run x for FFN→KV elbows

step = 2.05
top_y = H - 0.35


# ═══════════════════════════════════════════════════════════
# TOP: two sparsity-dimension labels + badges + arrows
# ═══════════════════════════════════════════════════════════
lab(kv_cx, top_y, "head-level sparsity", size=10.5, weight="bold")
lab(ffn_cx, top_y, "token-level sparsity", size=10.5, weight="bold")

# ❶ badge right below "head-level" text, beside arrow
varrow(kv_cx, top_y - 0.22, top_y - 0.80)
badge(kv_cx + 0.30, top_y - 0.52, 1)

# ❷ badge right below "token-level" text, beside arrow
varrow(ffn_cx, top_y - 0.22, top_y - 0.80)
badge(ffn_cx + 0.30, top_y - 0.52, 2)


# ═══════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════


def draw_kv(num, cy, kv_type):
    fc = BLUE if kv_type == "local" else RED
    txt = "Local-head KV Cache" if kv_type == "local" else "Global-head KV Cache"
    box(kv_x, cy - kv_h / 2, kv_w, kv_h, fc=fc, lw=LW_MID, z=4)
    lab(kv_x + kv_w / 2, cy, txt, size=10.5, weight="bold", z=6)
    # Layer label — left of KV box with clear gap
    lab(kv_x - 0.20, cy, f"Layer {num}", size=10.5, weight="bold", ha="right")
    return cy


def draw_ffn(cy, ffn_type):
    if ffn_type == "dense":
        # Dense FFN → GRAY fill
        box(ffn_x, cy - ffn_h / 2, ffn_w, ffn_h, fc=GRAY, lw=LW_MID, z=4)
        lab(ffn_x + ffn_w / 2, cy, "Dense FFN", size=10.5, weight="bold", z=6)
    else:
        box(ffn_x, cy - ffn_h / 2, ffn_w, ffn_h, fc=LIGHT, ec=INK, lw=LW_MID, z=4)
        lab(ffn_x + ffn_w / 2, cy, "Sparse FFN", size=10.5, weight="bold", z=6)
    return cy


def kv_to_ffn_elbow(kv_cy, ffn_cy):
    """Orthogonal: KV right edge → right → down → FFN left edge."""
    elbow_arrow(
        [
            (kv_x + kv_w, kv_cy),  # start at KV right
            (gutter_x, kv_cy),  # go right to gutter
            (gutter_x, ffn_cy),  # go down
            (ffn_x, ffn_cy),  # go right into FFN left edge
        ]
    )


def ffn_to_kv_elbow(ffn_cy, kv_cy):
    """Orthogonal: FFN right edge → right → down → right to KV right edge."""
    elbow_arrow(
        [
            (ffn_x + ffn_w, ffn_cy),  # start at FFN right
            (return_x, ffn_cy),  # go right
            (return_x, kv_cy),  # go down
            (kv_x + kv_w, kv_cy),  # go left into KV right edge
        ]
    )


# ═══════════════════════════════════════════════════════════
# SHALLOW LAYERS
# ═══════════════════════════════════════════════════════════

# Layer 0
y0 = top_y - 1.55
kv0 = draw_kv(0, y0, "local")
f0 = draw_ffn(y0 - 0.90, "dense")
kv_to_ffn_elbow(kv0, f0)

# Layer 1
y1 = y0 - step
kv1 = draw_kv(1, y1, "local")
f1 = draw_ffn(y1 - 0.90, "dense")
kv_to_ffn_elbow(kv1, f1)

# ❸ badge near the FFN→KV return arrow
ffn_to_kv_elbow(f0, kv1)
badge(return_x + 0.25, (f0 + kv1) / 2, 3)

# Layer 2: KV only (last shown shallow layer)
y2 = y1 - step
kv2 = draw_kv(2, y2, "local")
ffn_to_kv_elbow(f1, kv2)


# ═══════════════════════════════════════════════════════════
# DIVIDER
# ═══════════════════════════════════════════════════════════
div_line_y = y2 - kv_h / 2 - 0.80

# ellipsis — lowered, bold
lab(kv_cx, div_line_y + 0.48, "···    ···", size=14, weight="bold", color="#666666")

ax.plot([1.8, W - 1.8], [div_line_y, div_line_y], color=INK, lw=2.5, zorder=5)

# ❹ Shallow Layers
lab(kv_cx + 1.2, div_line_y + 0.22, "Shallow Layers", size=10.5, weight="bold")
badge(kv_cx + 3.30, div_line_y + 0.22, 4)

# ❺ Deep Layers
lab(kv_cx + 1.2, div_line_y - 0.25, "Deep Layers", size=10.5, weight="bold")
badge(kv_cx + 3.30, div_line_y - 0.25, 5)


# ═══════════════════════════════════════════════════════════
# DEEP LAYERS
# ═══════════════════════════════════════════════════════════

# Layer 79
y79 = div_line_y - 1.35
kv79 = draw_kv(79, y79, "global")
f79 = draw_ffn(y79 - 0.90, "sparse")
kv_to_ffn_elbow(kv79, f79)

# Layer 80: KV only
y80 = y79 - step
kv80 = draw_kv(80, y80, "global")
ffn_to_kv_elbow(f79, kv80)


# ═══════════════════════════════════════════════════════════
# LEFT BRACKETS — pulled further left with gap to Layer labels
# ═══════════════════════════════════════════════════════════
bx = 0.30  # bracket x
bx2 = 0.42  # bracket tick end

# Shallow bracket
sh_t = kv0 + kv_h / 2 + 0.12
sh_b = kv2 - kv_h / 2 - 0.12
ax.plot([bx, bx2], [sh_t, sh_t], color=INK, lw=1.2, zorder=5)
ax.plot([bx, bx], [sh_t, sh_b], color=INK, lw=1.2, zorder=5)
ax.plot([bx, bx2], [sh_b, sh_b], color=INK, lw=1.2, zorder=5)

# Deep bracket
dp_t = kv79 + kv_h / 2 + 0.12
dp_b = kv80 - kv_h / 2 - 0.12
ax.plot([bx, bx2], [dp_t, dp_t], color=INK, lw=1.2, zorder=5)
ax.plot([bx, bx], [dp_t, dp_b], color=INK, lw=1.2, zorder=5)
ax.plot([bx, bx2], [dp_b, dp_b], color=INK, lw=1.2, zorder=5)


# ═══════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════
fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig("Method.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig("Method.png", dpi=240, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("saved Method.pdf + Method.png")
