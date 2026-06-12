"""Illustrative score-overlap concept figure for the diagnosis chapter.

Original schematic (no data values reproduced); the qualitative geometry follows
the score-overlap study summarised in the hard-case overlap table.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2] / 'draft' / 'aauReportTemplate'
OUT = ROOT / "AAUgraphics"
AAUBLUE = (33 / 255, 26 / 255, 82 / 255)

x = np.linspace(-1, 9, 700)


def g(mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2)


groups = [
    ("far-OOD", 1.8, 0.85, "#b5402f"),
    ("near-OOD", 4.4, 1.0, "#e08a2e"),
    ("cs-ID", 5.5, 0.95, "#3a9aa0"),
    ("clean ID", 6.4, 0.8, "#2f6fb0"),
]

fig, ax = plt.subplots(figsize=(9.4, 4.4), dpi=200)
for name, mu, sd, col in groups:
    y = g(mu, sd)
    ax.plot(x, y, color=col, lw=2)
    ax.fill_between(x, y, color=col, alpha=0.18)
    ax.text(mu, 1.05, name, ha="center", color=col, fontsize=10, fontweight="bold")

ax.axvspan(4.55, 5.4, color="#888888", alpha=0.12)
ax.annotate("hard boundary:\ncs-ID vs near-OOD overlap", xy=(4.97, 0.45), xytext=(2.4, 0.80),
            fontsize=9, color=AAUBLUE, ha="center",
            arrowprops=dict(arrowstyle="-|>", color=AAUBLUE, lw=1.1))
ax.annotate("benign shift:\nclean and cs-ID should overlap", xy=(5.95, 0.52), xytext=(7.6, 0.84),
            fontsize=9, color=AAUBLUE, ha="center",
            arrowprops=dict(arrowstyle="-|>", color=AAUBLUE, lw=1.1))
ax.annotate("far-OOD: easily separated", xy=(1.8, 0.30), xytext=(0.0, 0.62),
            fontsize=9, color="#7a2a20", ha="left",
            arrowprops=dict(arrowstyle="-|>", color="#7a2a20", lw=1.0))

ax.set_xlabel(r"detector score   (low $=$ OOD-like   $\rightarrow$   high $=$ ID-compatible)",
              color=AAUBLUE, fontsize=10)
ax.set_yticks([]); ax.set_xticks([]); ax.set_ylim(0, 1.2); ax.set_xlim(-1, 9)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(AAUBLUE)
fig.tight_layout()
fig.savefig(OUT / "score_overlap_concept.pdf")
fig.savefig(OUT / "score_overlap_concept.png", dpi=200)
print("ok overlap")
