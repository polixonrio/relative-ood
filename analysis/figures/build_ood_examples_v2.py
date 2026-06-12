"""Refreshed four-panel full-spectrum grouping figure (clean style).

Uses the Creative Commons Wikimedia example photographs already in the repo; the
cs-ID panel applies a synthetic corruption to the clean-ID image. Overwrites the
older ood_examples output that used the bitmap default font.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2] / 'draft' / 'aauReportTemplate'
SRC = ROOT / "AAUgraphics" / "example_sources"
OUT = ROOT / "AAUgraphics"
AAUBLUE = (33 / 255, 26 / 255, 82 / 255)


def fit(p, s=(360, 360)):
    return ImageOps.fit(Image.open(p).convert("RGB"), s, method=Image.Resampling.LANCZOS)


def corrupt(im):
    out = im.filter(ImageFilter.GaussianBlur(2.2))
    a = np.asarray(out).astype(np.float32)
    a += np.random.default_rng(0).normal(0, 28, a.shape)
    a = (a - 128) * 0.82 + 128 + 10
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


clean = fit(SRC / "clean_id_dog.jpg")
imgs = [clean, corrupt(clean), fit(SRC / "near_ood_fox.jpg"), fit(SRC / "far_ood_airplane.jpg")]
labels = ["Clean ID", "Covariate-shifted ID", "Near-OOD", "Far-OOD"]
desc = ["golden retriever\n(known class)", "same class, corrupted\n(kept under full-spectrum)",
        "red fox\n(nearby novel class)", "airplane\n(unrelated)"]
is_id = [True, True, False, False]

fig, axs = plt.subplots(1, 4, figsize=(10.2, 3.3), dpi=200)
for ax, im, lab, d, idflag in zip(axs, imgs, labels, desc, is_id):
    ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    col = "#2f6fb0" if idflag else "#b5402f"
    for s in ax.spines.values():
        s.set_edgecolor(col); s.set_linewidth(2.4)
    ax.set_title(lab, color=AAUBLUE, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel(d, color="#444444", fontsize=8.5)

fig.text(0.5, 0.03, "accepted (in-distribution)        |        rejected (out-of-distribution)",
         ha="center", fontsize=8.5, color="#5b6b80")
fig.tight_layout(rect=[0, 0.055, 1, 1])
fig.savefig(OUT / "ood_examples.pdf")
fig.savefig(OUT / "ood_examples.png", dpi=200)
print("ok ood_examples")
