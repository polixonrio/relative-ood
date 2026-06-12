"""Generate Figure 4.1 (metric_roc_illustration.png) for the thesis.

Illustrates AUROC (the shaded area) and FPR95 (the 95%-TPR operating point) on
one ROC curve, for MSP on ImageNet-200 under the standard protocol. To stay
consistent with the rest of the thesis, the AUROC and FPR95 are the OpenOOD
per-source averages over the five OOD sources, averaged over the three
ImageNet-200 seeds (computed with OpenOOD's own metric function), and the drawn
curve is the seed-averaged source-averaged ROC.

Run:  python analysis/figures/metric_roc_illustration.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from openood.evaluators.metrics import compute_all_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT = os.path.abspath(os.path.join(REPO, 'draft', 'aauReportTemplate', 'AAUgraphics', 'thesis_artifacts'))
CACHE_ROOT = os.path.join(REPO, 'results', 'cache', 'model', 'imagenet200', 'resnet18_224x224')
SEEDS = ['s0__125431080d469cdc', 's1__9cecb1851838f26d', 's2__9c4a892af8c0560a']
SOURCES = ['ood_near__ssb_hard', 'ood_near__ninco', 'ood_far__inaturalist', 'ood_far__textures', 'ood_far__openimage_o']


def msp(cache, split):
    z = np.load(os.path.join(cache, split, 'logits.npy')).astype(np.float64)
    z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    return p.max(1)


def npy(cache, split, name):
    return np.load(os.path.join(cache, split, name + '.npy'))


# Average over the three ImageNet-200 seeds, matching the seed-averaged AUROC and
# FPR95 the thesis tables report. Within each seed the metrics are the OpenOOD
# per-source averages over the five OOD sources (OpenOOD's own metric function).
grid = np.linspace(0, 1, 1000)
seed_tpr, seed_auroc, seed_fpr95 = [], [], []
for seed in SEEDS:
    cache = os.path.join(CACHE_ROOT, seed)
    idm, idl, idp = msp(cache, 'id__test'), npy(cache, 'id__test', 'labels'), npy(cache, 'id__test', 'preds')
    tprs, aurocs, fpr95s = [], [], []
    for s in SOURCES:
        om = msp(cache, s); n = len(om)
        # OpenOOD convention: OOD is the positive class, score = -confidence
        fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(idm)), np.ones(n)], np.r_[-idm, -om])
        tprs.append(np.interp(grid, fpr, tpr))
        m = compute_all_metrics(np.r_[idm, om], np.r_[idl, -np.ones(n)], np.r_[idp, np.zeros(n)])
        fpr95s.append(m[0] * 100.0); aurocs.append(m[1] * 100.0)
    seed_tpr.append(np.mean(tprs, axis=0))
    seed_auroc.append(float(np.mean(aurocs)))
    seed_fpr95.append(float(np.mean(fpr95s)))

mtpr = np.mean(seed_tpr, axis=0); mtpr[0] = 0.0
A = float(np.mean(seed_auroc))      # seed-averaged OpenOOD AUROC (matches the thesis tables)
F = float(np.mean(seed_fpr95))      # seed-averaged OpenOOD FPR95 (matches the thesis tables)

plt.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(figsize=(4.8, 4.5))
ax.fill_between(grid, mtpr, color='#4C78A8', alpha=0.18, label=f'AUROC = {A:.1f}')
ax.plot(grid, mtpr, color='#4C78A8', lw=2.6, zorder=3)
ax.plot([0, 1], [0, 1], ls=(0, (4, 4)), color='#999999', lw=1.3, label='chance (AUROC 50)')
# FPR95 operating point (source-averaged)
f = F / 100.0
ax.plot([0, f], [0.95, 0.95], ls=':', color='#d62728', lw=1.6)
ax.plot([f, f], [0, 0.95], ls=':', color='#d62728', lw=1.6)
ax.scatter([f], [0.95], color='#d62728', zorder=5, s=60)
ax.annotate(f'FPR95 = {F:.1f}', xy=(f, 0.95), xytext=(f + 0.14, 0.70), fontsize=13,
            color='#b01e1e', arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.4))
ax.text(0.57, 0.27, f'AUROC\n= shaded area\n= {A:.1f}', fontsize=13.5, color='#2a4d6e', ha='center')
ax.set_xlabel('false positive rate', fontsize=13.5)
ax.set_ylabel('true positive rate', fontsize=13.5)
ax.tick_params(labelsize=12)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
ax.set_aspect('equal')
ax.legend(loc='lower right', fontsize=12, framealpha=0.95)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, 'metric_roc_illustration.png')
fig.savefig(out, dpi=200)
print('saved', out, '| source-avg AUROC', round(A, 1), '| FPR95', round(F, 1))
