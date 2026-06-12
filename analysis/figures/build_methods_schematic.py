"""Three-panel schematic of the relative-scoring designs (original diagram)."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2] / 'draft' / 'aauReportTemplate'
OUT = ROOT / "AAUgraphics"
AAUBLUE = (33 / 255, 26 / 255, 82 / 255)
GREEN, RED, TEAL, ORANGE = "#2f8f5b", "#c0392b", "#3a9aa0", "#e08a2e"

fig, axs = plt.subplots(1, 3, figsize=(10.6, 3.9), dpi=200)
for a in axs:
    a.set_xticks([]); a.set_yticks([]); a.set_xlim(0, 10); a.set_ylim(0, 10)
    for s in a.spines.values():
        s.set_visible(False)

# (a) rTSS
a = axs[0]
a.set_title("rTSS: normalise per predicted class", color=AAUBLUE, fontsize=10, fontweight="bold")
xx = np.linspace(0, 10, 400); mu, sd = 4.6, 1.25
a.plot(xx, 3 + 2.4 * np.exp(-0.5 * ((xx - mu) / sd) ** 2), color=TEAL, lw=2)
a.fill_between(xx, 3, 3 + 2.4 * np.exp(-0.5 * ((xx - mu) / sd) ** 2), color=TEAL, alpha=0.15)
a.axvline(mu, color=TEAL, ls="--", lw=1)
a.text(mu, 5.8, r"$\mu_{\hat y},\,\sigma_{\hat y}$", ha="center", color=TEAL, fontsize=10)
a.plot([8.4], [3], marker="o", color=RED, ms=8)
a.text(8.4, 2.2, "raw TSS\nof sample", ha="center", color=RED, fontsize=8)
a.add_patch(FancyArrowPatch((mu, 3.05), (8.3, 3.05), arrowstyle="<->", color=AAUBLUE, lw=1.1, mutation_scale=10))
a.text((mu + 8.4) / 2, 3.3, r"$(\mathrm{TSS}-\mu_{\hat y})/\sigma_{\hat y}$", ha="center", color=AAUBLUE, fontsize=9)
a.text(5, 0.7, "score relative to the predicted\nclass's own TSS distribution", ha="center", color="#444", fontsize=8.5)

# (b) rLSM
b = axs[1]
b.set_title("rLSM: class vs background (logits)", color=AAUBLUE, fontsize=10, fontweight="bold")
b.add_patch(Ellipse((5, 5.2), 8.6, 6.4, fc="#dfe6ee", ec="#9aa6b5", lw=1.2))
b.text(5, 8.8, "background (all logits)", ha="center", color="#5b6b80", fontsize=8)
for cxp, cyp in [(3.3, 6.3), (6.7, 6.5), (5.1, 3.6)]:
    b.add_patch(Ellipse((cxp, cyp), 2.2, 1.6, fc=GREEN, ec=GREEN, alpha=0.22, lw=1))
    b.plot([cxp], [cyp], marker="x", color=GREEN, ms=7)
q = (4.2, 5.1); b.plot([q[0]], [q[1]], marker="o", color=RED, ms=8)
b.add_patch(FancyArrowPatch(q, (3.3, 6.3), arrowstyle="-|>", color=GREEN, lw=1.5, mutation_scale=11))
b.add_patch(FancyArrowPatch(q, (5, 5.2), arrowstyle="-|>", color="#5b6b80", lw=1.2, ls="--", mutation_scale=11))
b.text(3.3, 6.9, "nearest class", color=GREEN, fontsize=7.5, ha="center")
b.text(5, 0.7, r"$s^{\mathrm{rel}}=$ class score $-$ background score", ha="center", color=AAUBLUE, fontsize=8.5)

# (c) RkNN++
c = axs[2]
c.set_title("RkNN++: global vs class neighbours", color=AAUBLUE, fontsize=10, fontweight="bold")
rng = np.random.default_rng(1)
c.scatter(rng.uniform(1, 9, 44), rng.uniform(1, 9, 44), s=8, color="#b9c2cf")
c.scatter(rng.normal(3.1, 0.7, 16), rng.normal(6.6, 0.7, 16), s=13, color=ORANGE)
c.text(3.1, 8.2, "predicted class", color=ORANGE, fontsize=7.5, ha="center")
q = (5.6, 4.0); c.plot([q[0]], [q[1]], marker="o", color=RED, ms=8)
c.add_patch(Circle(q, 1.4, fill=False, ec="#5b6b80", lw=1.3, ls=":"))
c.text(q[0] + 1.5, q[1] - 0.5, r"$d_{\mathrm{global}}$", color="#5b6b80", fontsize=8.5)
c.add_patch(Circle(q, 3.1, fill=False, ec=ORANGE, lw=1.3, ls="--"))
c.text(q[0] - 0.1, q[1] + 3.2, r"$d_{\mathrm{class}}$", color=ORANGE, fontsize=8.5, ha="center")
c.text(5, 0.7, r"score $= d_{\mathrm{global}} - d_{\mathrm{class}}$", ha="center", color=AAUBLUE, fontsize=8.5)

fig.tight_layout()
fig.savefig(OUT / "relative_scoring_schematic.pdf")
fig.savefig(OUT / "relative_scoring_schematic.png", dpi=200)
print("ok methods")
