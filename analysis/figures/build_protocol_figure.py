"""Build the Standard vs Full-Spectrum protocol concept figure.

Original schematic (no third-party figures reproduced). Example photographs are
the Creative Commons Wikimedia files already used elsewhere in the report; the
cs-ID panel applies a synthetic ImageNet-C-style corruption to the clean-ID image.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2] / 'draft' / 'aauReportTemplate'
SRC = ROOT / "AAUgraphics" / "example_sources"
OUT_PDF = ROOT / "AAUgraphics" / "standard_vs_fullspectrum_concept.pdf"
OUT_PNG = ROOT / "AAUgraphics" / "standard_vs_fullspectrum_concept.png"

AAUBLUE = (33 / 255, 26 / 255, 82 / 255)
GREEN = "#2f8f5b"
RED = "#c0392b"
GREY = "#9aa0a6"
GOLD = "#c08a00"


def fit(path, size=(360, 360)):
    im = Image.open(path).convert("RGB")
    return ImageOps.fit(im, size, method=Image.Resampling.LANCZOS)


def corrupt(im):
    """Emulate a covariate (ImageNet-C-style) shift: blur + noise + lower contrast."""
    out = im.filter(ImageFilter.GaussianBlur(radius=2.2))
    arr = np.asarray(out).astype(np.float32)
    arr += np.random.default_rng(0).normal(0, 28, arr.shape)
    arr = (arr - 128) * 0.82 + 128 + 10
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


clean = fit(SRC / "clean_id_dog.jpg")
imgs = [clean, corrupt(clean), fit(SRC / "near_ood_fox.jpg"), fit(SRC / "far_ood_airplane.jpg")]
glabels = ["Clean ID", "Covariate-shifted ID", "Near-OOD", "Far-OOD"]
gdesc = ["golden retriever\n(known class)", "same class,\ncorrupted",
         "red fox\n(nearby class)", "airplane\n(unrelated)"]

cx = [0.285, 0.490, 0.695, 0.900]
IW, IH, IB = 0.145, 0.280, 0.600

fig = plt.figure(figsize=(9.4, 5.3), dpi=240)
for c, im in zip(cx, imgs):
    axi = fig.add_axes([c - IW / 2, IB, IW, IH])
    axi.imshow(im)
    axi.set_xticks([]); axi.set_yticks([])
    for s in axi.spines.values():
        s.set_edgecolor(AAUBLUE); s.set_linewidth(1.4)

ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

for c, gl, gd in zip(cx, glabels, gdesc):
    ax.text(c, 0.905, gl, ha="center", va="bottom", fontsize=14.2, fontweight="bold", color=AAUBLUE)
    ax.text(c, 0.578, gd, ha="center", va="top", fontsize=10.8, color="#333333")

ax.annotate("", xy=(0.965, 0.505), xytext=(0.205, 0.505),
            arrowprops=dict(arrowstyle="-|>", color=AAUBLUE, lw=1.5))
ax.text(0.205, 0.488, "more in-distribution", ha="left", va="top", fontsize=10.2, style="italic", color="#555")
ax.text(0.965, 0.488, "more semantically novel", ha="right", va="top", fontsize=10.2, style="italic", color="#555")


def token(c, y, kind):
    colors = {"accept": GREEN, "reject": RED, "excl": GREY}
    label = {"accept": "Accept", "reject": "Reject", "excl": "Not in protocol"}
    w, h = 0.145, 0.086
    ax.add_patch(FancyBboxPatch((c - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.006,rounding_size=0.02", linewidth=0,
                 facecolor=colors[kind], alpha=(0.28 if kind == "excl" else 0.93)))
    ax.text(c, y, label[kind], ha="center", va="center", fontsize=10.8,
            color=("#555" if kind == "excl" else "white"), fontweight="bold")


ax.text(0.022, 0.375, "Standard\nOOD", ha="left", va="center", fontsize=14.4, fontweight="bold", color=AAUBLUE)
ax.text(0.022, 0.205, "Full-spectrum\nOOD", ha="left", va="center", fontsize=14.4, fontweight="bold", color=AAUBLUE)
for c, k in zip(cx, ["accept", "excl", "reject", "reject"]):
    token(c, 0.375, k)
for c, k in zip(cx, ["accept", "accept", "reject", "reject"]):
    token(c, 0.205, k)

ax.add_patch(FancyBboxPatch((cx[1] - 0.086, 0.158), 0.172, 0.272,
             boxstyle="round,pad=0.004,rounding_size=0.02", linewidth=1.3,
             edgecolor=GOLD, facecolor="none", linestyle=(0, (4, 2))))
ax.text(cx[1], 0.135, "the only group that changes", ha="center", va="top",
        fontsize=9.6, style="italic", color=GOLD)

x0 = 0.205
for fc, lab in [(GREEN, "Accepted (scored as ID)"), (RED, "Rejected (scored as OOD)"),
                (GREY, "Excluded from the protocol")]:
    ax.add_patch(FancyBboxPatch((x0, 0.035), 0.022, 0.03,
                 boxstyle="round,pad=0.002,rounding_size=0.006", linewidth=0,
                 facecolor=fc, alpha=(0.28 if fc == GREY else 0.93)))
    ax.text(x0 + 0.03, 0.05, lab, ha="left", va="center", fontsize=9.2, color="#333")
    x0 += 0.265

fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=200)
print("wrote", OUT_PDF.name, "and", OUT_PNG.name)
