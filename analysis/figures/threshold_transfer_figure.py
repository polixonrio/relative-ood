"""Generate Figure 5.2 (all_method_threshold_transfer.png) for the thesis.

Threshold-transfer scatter at the 95% clean-ID operating point, under the
full-spectrum protocol, for the two ImageNet settings: x = cs-ID wrongly
rejected, y = near-OOD wrongly accepted, one point per detector. Marker shape
encodes the detector family and color identifies the method.

Run:  python analysis/figures/threshold_transfer_figure.py
Reads results/thesis_artifacts/boundary_inversion_panel.csv and writes into the
thesis AAUgraphics. Repo-relative, so it is reproducible (this is Figure 5.2,
which previously had no tracked generator).
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV = os.path.join(REPO, "results", "thesis_artifacts", "boundary_inversion_panel.csv")
OUT = os.path.join(REPO, "draft", "aauReportTemplate", "AAUgraphics", "thesis_artifacts", "all_method_threshold_transfer.png")

FAMILY = {}
for m in ["MSP", "MaxLogit", "Energy", "GEN", "TSS", "rTSS", "RTSS"]:
    FAMILY[m] = "output"
for m in ["KNN", "RkNN++", "RKNN++", "Mahalanobis", "MDS++", "RMDS", "RMDS++", "LSM", "LSM++", "rLSM", "RLSM", "rLSM++", "RLSM++"]:
    FAMILY[m] = "distance"
for m in ["ReAct", "ASH", "DICE", "SCALE", "VIM"]:
    FAMILY[m] = "feature/activation"
FAMILY["GradNorm"] = "gradient"
FAM_MARKER = {"output": "o", "distance": "s", "feature/activation": "^", "gradient": "D"}
# Data keys stay as-is; only the legend label is remapped to the thesis display names.
DISPLAY = {"Mahalanobis": "MDS", "VIM": "ViM", "rTSS": "RTSS", "RkNN++": "RKNN++", "rLSM": "RLSM", "rLSM++": "RLSM++"}

panel = pd.read_csv(CSV)
methods = panel["method"].tolist()
COLORS = list(plt.cm.tab20.colors)
color = {m: COLORS[i % 20] for i, m in enumerate(methods)}
marker = {m: FAM_MARKER[FAMILY[m]] for m in methods}
SIZE_TIERS = [200, 140, 95, 60]
size = {m: SIZE_TIERS[i % len(SIZE_TIERS)] for i, m in enumerate(methods)}

DS = [("in200", "ImageNet-200"), ("in1K", "ImageNet-1K")]
fig, axes = plt.subplots(1, 2, figsize=(14, 7.4))
for ax, (tag, name) in zip(axes, DS):
    xs = panel[f"{tag}_csid_rejected"].to_numpy() * 100
    ys = panel[f"{tag}_nearood_accepted"].to_numpy() * 100
    ax.fill_between([0, 100], [100, 0], 100, color="0.93", zorder=0)
    ax.plot([0, 100], [100, 0], "--", color="0.55", linewidth=1, zorder=1)
    for m, x, y in zip(methods, xs, ys):
        ax.scatter(x, y, s=size[m], color=color[m], marker=marker[m], edgecolor="black",
                   linewidth=0.5, alpha=0.8, zorder=3 + (200 - size[m]) / 20.0)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("cs-ID wrongly rejected (%)", fontsize=14)
    ax.set_ylabel("near-OOD wrongly accepted (%)", fontsize=14)
    ax.set_title(name, fontsize=16)
    ax.tick_params(labelsize=12)
    ax.grid(True, linewidth=0.3, alpha=0.5)

fam_handles = [Line2D([0], [0], marker=FAM_MARKER[f], linestyle="", markerfacecolor="0.6",
                      markeredgecolor="black", markersize=11, label=f) for f in FAM_MARKER]
method_handles = [Line2D([0], [0], marker=marker[m], linestyle="", markerfacecolor=color[m],
                         markeredgecolor="black", markersize=10, label=DISPLAY.get(m, m)) for m in methods]

fig.subplots_adjust(left=0.06, right=0.80, bottom=0.09, top=0.94, wspace=0.22)
fig.legend(handles=fam_handles, loc="upper left", bbox_to_anchor=(0.815, 0.965),
           fontsize=11.5, title="Family (shape)", title_fontsize=12, framealpha=0.95)
fig.legend(handles=method_handles, loc="upper left", bbox_to_anchor=(0.815, 0.80),
           fontsize=10.5, title="Method", title_fontsize=12, handletextpad=0.4, labelspacing=0.32)
fig.savefig(OUT, dpi=190)
plt.close(fig)
print("wrote", OUT)
