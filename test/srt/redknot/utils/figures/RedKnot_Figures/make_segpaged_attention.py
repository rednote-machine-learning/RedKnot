#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RedKnot SegPagedAttention figure."""

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
LW_IN = 1.0

W, H = 16.0, 9.0
fig, ax = plt.subplots(figsize=(8.0, 4.5))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")


def box(x, y, w, h, fc="white", ec=EDGE, lw=LW_MID, z=2):
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    )


def lab(x, y, s, size=10.5, weight="normal", color=INK, ha="center", va="center", z=8):
    ax.text(
        x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z
    )


def badge(x, y, n, r=0.18, size=8.5, z=14):
    ax.add_patch(
        plt.Circle((x, y), r, facecolor="#222", edgecolor="white", lw=1.0, zorder=z)
    )
    lab(x, y, str(n), size=size, weight="bold", color="white", z=z + 1)


def elbow_arrow(pts, lw=1.45, ms=10, color=INK, z=7):
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


def token_row(x, y, labels, colors, cell_w=0.46, cell_h=0.46, gap=0.04):
    for i, (txt, fc) in enumerate(zip(labels, colors)):
        xx = x + i * (cell_w + gap)
        box(xx, y, cell_w, cell_h, fc=fc, ec=EDGE, lw=LW_IN, z=4)
        lab(xx + cell_w / 2, y + cell_h / 2, txt, size=7.8, weight="bold", z=6)


def page_grid(x, y, name, color, hot_cols):
    lab(x + 1.05, y + 1.02, name, size=9.6, weight="bold")
    for r in range(2):
        for c in range(4):
            fc = color if c in hot_cols else GRAY
            xx = x + c * 0.50
            yy = y + (1 - r) * 0.43
            box(xx, yy, 0.42, 0.35, fc=fc, ec=EDGE, lw=LW_IN, z=4)


# Outer frame.
box(0.25, 0.25, W - 0.50, H - 0.50, fc="white", ec=INK, lw=LW_OUT, z=1)

# Header labels.
lab(2.45, 8.22, "Segmented Request", size=11.5, weight="bold")
lab(7.10, 8.22, "Segment Page Table", size=11.5, weight="bold")
lab(12.70, 8.22, "Paged KV Cache", size=11.5, weight="bold")

# Left: logical token stream split into segments.
box(0.65, 5.65, 3.55, 1.62, fc=LIGHT, ec=EDGE, lw=LW_MID)
lab(2.43, 6.95, "logical tokens", size=10.6, weight="bold")
token_row(1.00, 6.35, ["Q0", "Q1", "Q2", "Q3", "Q4"], [ORANGE] * 5)
token_row(1.00, 5.84, ["D0", "D1", "D2", "D3", "D4"], [BLUE] * 5)
lab(1.00, 5.58, "query", size=8.8, weight="bold", ha="left")
lab(2.52, 5.58, "retrieved", size=8.8, weight="bold", ha="left")
badge(3.93, 6.98, 1)

box(0.65, 3.02, 3.55, 1.55, fc=LIGHT, ec=EDGE, lw=LW_MID)
lab(2.43, 4.27, "segment metadata", size=10.6, weight="bold")
lab(1.00, 3.78, "seg id", size=9.5, weight="bold", ha="left")
lab(2.15, 3.78, "S0", size=10.0, weight="bold")
lab(2.85, 3.78, "S1", size=10.0, weight="bold")
lab(3.55, 3.78, "S2", size=10.0, weight="bold")
lab(1.00, 3.34, "length", size=9.5, weight="bold", ha="left")
lab(2.15, 3.34, "5", size=10.0, weight="bold")
lab(2.85, 3.34, "5", size=10.0, weight="bold")
lab(3.55, 3.34, "3", size=10.0, weight="bold")
badge(4.34, 4.42, 2)

# Middle: segment page table.
box(5.00, 2.40, 4.10, 4.90, fc=LIGHT, ec=EDGE, lw=LW_MID)
rows = [
    ("S0", ["P0", "P1", "P2"], ORANGE),
    ("S1", ["P8", "P9", "P10"], BLUE),
    ("S2", ["P4", "P5", "--"], RED),
]
for i, (name, pages, color) in enumerate(rows):
    yy = 6.22 - i * 1.38
    lab(5.42, yy + 0.18, name, size=10.5, weight="bold")
    for j, p in enumerate(pages):
        xx = 5.85 + j * 0.82
        fc = color if p != "--" else GRAY
        box(xx, yy, 0.62, 0.42, fc=fc, ec=EDGE, lw=LW_IN, z=4)
        lab(xx + 0.31, yy + 0.21, p, size=8.8, weight="bold", z=6)
    lab(8.35, yy + 0.20, "logical\nview", size=8.4, weight="bold")
badge(8.82, 7.04, 3)

# Right: physical KV pages.
box(10.00, 4.75, 5.25, 2.55, fc=LIGHT, ec=EDGE, lw=LW_MID)
page_grid(10.35, 5.78, "global heads", RED, hot_cols={0, 1, 2, 3})
page_grid(12.65, 5.78, "local heads", BLUE, hot_cols={1, 2})
badge(14.98, 7.04, 4)

box(10.00, 1.20, 5.25, 2.55, fc=LIGHT, ec=EDGE, lw=LW_MID)
lab(12.62, 3.30, "SegPagedAttention Kernel", size=10.8, weight="bold")
box(10.45, 2.46, 1.08, 0.44, fc=RED, ec=EDGE, lw=LW_IN)
lab(10.99, 2.68, "full", size=8.8, weight="bold")
box(12.05, 2.46, 1.08, 0.44, fc=BLUE, ec=EDGE, lw=LW_IN)
lab(12.59, 2.68, "local", size=8.8, weight="bold")
box(13.65, 2.46, 1.08, 0.44, fc=ORANGE, ec=EDGE, lw=LW_IN)
lab(14.19, 2.68, "active", size=8.8, weight="bold")
lab(12.62, 1.78, "gather pages once,\nattend by segment", size=9.2, weight="bold")
badge(15.00, 3.42, 5)

# Orthogonal dataflow arrows.
elbow_arrow([(4.20, 6.35), (4.58, 6.35), (4.58, 6.05), (5.00, 6.05)])
elbow_arrow([(4.20, 3.78), (4.58, 3.78), (4.58, 4.12), (5.00, 4.12)])
elbow_arrow([(9.10, 5.55), (9.55, 5.55), (9.55, 6.02), (10.00, 6.02)])
elbow_arrow([(12.62, 4.75), (12.62, 4.28), (12.62, 3.75)])
elbow_arrow([(9.10, 3.40), (9.55, 3.40), (9.55, 2.50), (10.00, 2.50)])

# Small legend, no caption.
box(0.65, 0.75, 9.25, 0.72, fc="white", ec=EDGE, lw=LW_IN)
for x, c, t in [
    (0.90, RED, "global KV"),
    (2.95, BLUE, "local KV"),
    (4.85, ORANGE, "active tokens"),
    (7.25, GRAY, "inactive pages"),
]:
    box(x, 0.88, 0.36, 0.28, fc=c, ec=EDGE, lw=LW_IN)
    lab(x + 0.47, 1.02, t, size=8.5, weight="bold", ha="left")

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig("SegPagedAttention.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig("SegPagedAttention.png", dpi=240, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("saved SegPagedAttention.pdf + SegPagedAttention.png")
