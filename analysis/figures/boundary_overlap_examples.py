"""Generate Figure 1.2 (boundary_overlap_examples.png) for the thesis.

Shows the covariate-shifted ID vs near-OOD MSP-score overlap on ImageNet-200,
with three predicted-class-matched example pairs (a shifted ID image and a
near-OOD image the classifier assigns to the same class at nearly the same
score) and the 95% clean-ID operating threshold.

Run:  python analysis/figures/boundary_overlap_examples.py
Reads the OpenOOD cache under results/cache and the benchmark imglists/images;
writes into the thesis draft/aauReportTemplate/AAUgraphics/thesis_artifacts.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from scipy.stats import gaussian_kde
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))            # openood repo root
OUT = os.path.abspath(os.path.join(REPO, 'draft', 'aauReportTemplate', 'AAUgraphics', 'thesis_artifacts'))
CACHE = os.path.join(REPO, 'results', 'cache', 'model', 'imagenet200', 'resnet18_224x224', 's0__125431080d469cdc')
ILDIR = os.path.join(REPO, 'data', 'benchmark_imglist', 'imagenet200')
IMG = os.path.join(REPO, 'data', 'images_largescale')


def load(split, il):
    z = np.load(os.path.join(CACHE, split, 'logits.npy')).astype(np.float64)
    z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    msp = p.max(1); pred = p.argmax(1)
    rows = [ln.split() for ln in open(os.path.join(ILDIR, il + '.txt'))]
    paths = np.array([r[0] for r in rows], dtype=object)
    labels = np.array([int(r[1]) for r in rows])
    return msp, pred, paths, labels


def load_msp(split):
    """Return just the MSP scores for a cached split (no imglist needed)."""
    z = np.load(os.path.join(CACHE, split, 'logits.npy')).astype(np.float64)
    z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    return p.max(1)


clean = load('id__test', 'test_imagenet200')
cs_splits = [load('csid__imagenet_r', 'test_imagenet200_r'), load('csid__imagenet_v2', 'test_imagenet200_v2')]
no_splits = [load('ood_near__ninco', 'test_ninco'), load('ood_near__ssb_hard', 'test_ssb_hard')]

cs_m = np.concatenate([s[0] for s in cs_splits]); no_m = np.concatenate([s[0] for s in no_splits])
far_m = np.concatenate([load_msp(s) for s in ('ood_far__inaturalist', 'ood_far__textures', 'ood_far__openimage_o')])
cs_by, no_by = defaultdict(list), defaultdict(list)
for msp, pred, paths, labels in cs_splits:
    for m, p, lab in zip(msp, paths, labels): cs_by[int(lab)].append((m, p))
for msp, pred, paths, labels in no_splits:
    for m, pr, p in zip(msp, pred, paths): no_by[int(pr)].append((m, p))


def best_pair(C, lo=0.43, hi=0.85):
    na = sorted([(m, p) for (m, p) in no_by[C] if lo <= m <= hi])
    ca = sorted([(m, p) for (m, p) in cs_by[C] if lo <= m <= hi])
    cms = [c[0] for c in ca]; best = None
    for (mn, pn) in na:
        k = np.searchsorted(cms, mn)
        for kk in (k - 1, k):
            if 0 <= kk < len(ca):
                gap = abs(mn - ca[kk][0])
                if best is None or gap < best[0]: best = (gap, (mn, pn), ca[kk])
    return best


def load_sq(rel, size=180):
    im = Image.open(os.path.join(IMG, *rel.split('/'))).convert('RGB')
    w, h = im.size; s = min(w, h)
    return im.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2)).resize((size, size))


# (class idx, accepted name, rejected near-OOD name)
TRIPLES = [(1, 'great white shark', 'dolphin'), (99, 'gazelle', 'wild goat'), (72, 'tabby', 'wildcat')]

BLUE, ORANGE, GREEN, RED = '#3b6fb0', '#e08a1e', '#2ca25f', '#c0392b'
xs = np.linspace(0, 1, 400)
kcl = gaussian_kde(clean[0])(xs); kc = gaussian_kde(cs_m)(xs); kn = gaussian_kde(no_m)(xs); kf = gaussian_kde(far_m)(xs)
ymax = max(kc.max(), kn.max(), kf.max())

fig, ax = plt.subplots(figsize=(9.2, 6.0))
kcl_c = np.minimum(kcl, ymax * 1.04)
ax.fill_between(xs, kcl_c, color=GREEN, alpha=0.10, lw=0)
ax.plot(xs, kcl_c, color=GREEN, lw=1.5, alpha=0.8)
ax.text(0.985, ymax * 0.62, 'clean-ID\n(easily accepted)', color=GREEN, fontsize=8.6, ha='right', va='center', fontweight='bold')
ax.fill_between(xs, kc, color=BLUE, alpha=0.30, lw=0); ax.plot(xs, kc, color=BLUE, lw=2.2, label='covariate-shifted ID  (accept)')
ax.fill_between(xs, kn, color=ORANGE, alpha=0.30, lw=0); ax.plot(xs, kn, color=ORANGE, lw=2.2, label='near-OOD  (reject)')
ax.fill_between(xs, kf, color=RED, alpha=0.20, lw=0); ax.plot(xs, kf, color=RED, lw=2.0, label='far-OOD  (reject)')
ax.text(0.12, np.interp(0.12, xs, kf) + ymax * 0.10, 'far-OOD\n(lowest, easier)', color=RED, fontsize=8.2, ha='center', va='bottom', fontweight='bold')

# 95% clean-ID operating threshold (accept everything to its right)
thr = float(np.percentile(clean[0], 5))
ax.plot([thr, thr], [0, ymax * 1.18], color='#222', lw=1.6, ls=(0, (5, 2)))
ax.text(thr, ymax * 1.205, f'95% clean-ID threshold\n(MSP $\\approx$ {thr:.2f})', fontsize=8.0, ha='center', va='bottom', color='#222', fontweight='bold')
ax.annotate('reject $\\leftarrow$', xy=(thr - 0.02, ymax * 0.12), ha='right', va='center', fontsize=8.5, color='#222')
ax.annotate('$\\rightarrow$ accept', xy=(thr + 0.02, ymax * 0.12), ha='left', va='center', fontsize=8.5, color='#222')

y_no, y_cs, zoom = ymax * 1.5, ymax * 2.15, 0.27
for C, accname, rejname in TRIPLES:
    b = best_pair(C)
    (mn, pn), (mc, pc) = b[1], b[2]
    t = 0.5 * (mn + mc)
    imc, imn = load_sq(pc), load_sq(pn)
    yk = np.interp(t, xs, np.maximum(kc, kn))
    ax.annotate('', xy=(t, yk + ymax * 0.02), xytext=(t, y_no - ymax * 0.42),
                arrowprops=dict(arrowstyle='-', color='#999', lw=1.0, ls=(0, (3, 2))))
    ax.add_artist(AnnotationBbox(OffsetImage(imc, zoom=zoom), (t, y_cs), frameon=True, bboxprops=dict(edgecolor=BLUE, lw=2.6)))
    ax.add_artist(AnnotationBbox(OffsetImage(imn, zoom=zoom), (t, y_no), frameon=True, bboxprops=dict(edgecolor=ORANGE, lw=2.6)))
    ax.text(t, y_cs + ymax * 0.30, accname, ha='center', va='bottom', fontsize=8.5, color=BLUE, fontweight='bold')
    ax.text(t, y_no - ymax * 0.30, rejname, ha='center', va='top', fontsize=8.5, color=ORANGE, fontweight='bold')
    ax.text(t, y_no - ymax * 0.50, f'MSP $\\approx$ {t:.2f}', ha='center', va='top', fontsize=8, color='#444')

ax.set_ylim(0, y_cs + ymax * 0.62); ax.set_xlim(0, 1); ax.set_yticks([])
ax.set_xlabel('MSP score  (softmax confidence, higher = more ID-like)', fontsize=10.5)
ax.set_ylabel('density', fontsize=10.5)
ax.legend(loc='upper left', frameon=True, fontsize=9)
for sp in ('top', 'right', 'left'): ax.spines[sp].set_visible(False)
fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, 'boundary_overlap_examples.png')
fig.savefig(out, dpi=200)
print('saved', out, '| 95% clean-ID threshold MSP =', round(thr, 3))
