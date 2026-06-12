"""Detector-family map for the background chapter (original schematic)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2] / 'draft' / 'aauReportTemplate'
OUT = ROOT / "AAUgraphics"
AAUBLUE = (33 / 255, 26 / 255, 82 / 255)

fig = plt.figure(figsize=(10.0, 3.7), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.text(50, 93, "Post-hoc OOD detectors on a fixed classifier", ha="center",
        fontsize=12, fontweight="bold", color=AAUBLUE)

boxes = [
    ("Output-based", "reads the softmax / logits", "MSP, Energy,\nGEN", "#eef1f6"),
    ("Distance-based", "feature / logit geometry", "MDS, RMDS,\nKNN", "#eef1f6"),
    ("Activation-based", "shapes internal activations", "ReAct, ViM, SCALE", "#eef1f6"),
    ("Relative (this thesis)", "score vs an internal reference", "RTSS, RLSM,\nRKNN++", "#dde6f6"),
]
xs = [5.5, 30.0, 54.5, 79.0]; w, y, h = 19.5, 40, 35
for i, ((t, sub, meth, fc), xp) in enumerate(zip(boxes, xs)):
    ec = AAUBLUE if i == 3 else "#9aa6b5"
    lw = 2.4 if i == 3 else 1.2
    ax.add_patch(FancyBboxPatch((xp, y), w, h, boxstyle="round,pad=0.4,rounding_size=2.0",
                 fc=fc, ec=ec, lw=lw))
    ax.text(xp + w / 2, y + h - 6, t, ha="center", fontsize=10, fontweight="bold", color=AAUBLUE)
    ax.text(xp + w / 2, y + h - 12.5, sub, ha="center", fontsize=7.5, style="italic", color="#5b6b80")
    ax.text(xp + w / 2, y + 9, meth, ha="center", fontsize=9, color="#333")
    if i < 3:
        ax.add_patch(FancyArrowPatch((xp + w, y + h / 2), (xs[i + 1], y + h / 2),
                     arrowstyle="-|>", color="#9aa6b5", lw=1.3, mutation_scale=12))

ax.text(50, 20, "the thesis adds relative, reference-based scoring on top of these families",
        ha="center", fontsize=9, style="italic", color="#5b6b80")
fig.savefig(OUT / "detector_family_map.pdf")
fig.savefig(OUT / "detector_family_map.png", dpi=200)
print("ok family")
