"""Generate Figure 3.1 (group_examples_real.png) for the thesis.

Two coherent rows (gazelle, tabby): clean ID, an ImageNet-R rendition for
cs-ID, a near-OOD image the classifier assigns to the same class (a genuine
lookalike of a novel class), and an unrelated far-OOD image.

Run:  python analysis/figures/group_examples.py
Reads the OpenOOD cache/images; writes into the thesis AAUgraphics.
Needs timm installed (for the ImageNet wnid -> class-name map).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT = os.path.abspath(os.path.join(REPO, 'draft', 'aauReportTemplate', 'AAUgraphics', 'thesis_artifacts'))
CACHE = os.path.join(REPO, 'results', 'cache', 'model', 'imagenet200', 'resnet18_224x224', 's0__125431080d469cdc')
ILDIR = os.path.join(REPO, 'data', 'benchmark_imglist', 'imagenet200')
IMG = os.path.join(REPO, 'data', 'images_largescale')
IMG_CLASSIC = os.path.join(REPO, 'data', 'images_classic')   # Textures (DTD) live here
import timm  # required: provides the imagenet synset-to-lemma mapping
LEMMA = os.path.join(os.path.dirname(timm.__file__), 'data', '_info', 'imagenet_synset_to_lemma.txt')


def load(split, il):
    z = np.load(os.path.join(CACHE, split, 'logits.npy')).astype(np.float64)
    z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    msp = p.max(1); pred = p.argmax(1)
    rows = [ln.split() for ln in open(os.path.join(ILDIR, il + '.txt'))]
    paths = np.array([r[0] for r in rows], dtype=object)
    labels = np.array([int(r[1]) for r in rows])
    return msp, pred, paths, labels


_, _, rpaths, rlabels = load('csid__imagenet_r', 'test_imagenet200_r')
idx2wnid = {}
for pth, lb in zip(rpaths, rlabels):
    idx2wnid.setdefault(int(lb), pth.split('/')[1])
wnid2name = {}
for ln in open(LEMMA, encoding='utf-8'):
    w, nm = ln.rstrip('\n').split('\t'); wnid2name[w] = nm
idx2name = {i: wnid2name.get(idx2wnid.get(i, ''), f'class {i}').split(',')[0] for i in range(200)}

clean = load('id__test', 'test_imagenet200')
csr = load('csid__imagenet_r', 'test_imagenet200_r')
near = [('SSB-hard', load('ood_near__ssb_hard', 'test_ssb_hard')), ('NINCO', load('ood_near__ninco', 'test_ninco'))]
far = [load('ood_far__textures', 'test_textures'), load('ood_far__inaturalist', 'test_inaturalist')]

ROOTS = [IMG, IMG_CLASSIC]


def load_sq(rel, size=200):
    parts = rel.split('/'); im = None
    for root in ROOTS:
        try: im = Image.open(os.path.join(root, *parts)).convert('RGB'); break
        except Exception: im = None
    if im is None: raise FileNotFoundError(rel)
    w, h = im.size; s = min(w, h)
    return im.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2)).resize((size, size))


def pick_true(split, C):
    msp, pred, paths, labels = split
    idx = np.where(labels == C)[0]; idx = idx[np.argsort(msp[idx])[::-1]]
    for i in idx:
        try: return load_sq(paths[i])
        except Exception: continue


def pick_pred(splits, C):
    cand = []
    for dsname, (msp, pred, paths, labels) in splits:
        for i in np.where(pred == C)[0]: cand.append((msp[i], paths[i], dsname))
    for m, p, ds in sorted(cand, reverse=True):
        try: return load_sq(p), p, ds
        except Exception: continue
    return None, None, None


def near_name(path):
    f = path.replace('\\', '/').split('/')[1]
    return wnid2name.get(f, f).split(',')[0] if f.startswith('n') and f[1:].isdigit() else f


def pick_far(splits, want):
    msp, pred, paths, labels = splits[0] if want == 'Textures' else splits[1]
    for i in np.argsort(msp)[::-1]:
        try: return load_sq(paths[i])
        except Exception: continue


def find(name):
    return next(k for k, v in idx2name.items() if name in v.lower())


def safe(im):
    return np.asarray(im) if im is not None else np.full((200, 200, 3), 225, np.uint8)


ROWS = [find('gazelle'), find('tabby')]
BLUE, CYAN, ORANGE, RED = '#2c6fb0', '#23a0b8', '#e08a1e', '#cc3b3b'
HEAD = [('clean-ID', BLUE), ('covariate-shifted ID', CYAN), ('near-OOD', ORANGE), ('far-OOD', RED)]

fig, axes = plt.subplots(2, 4, figsize=(11.2, 6.4))
for r, C in enumerate(ROWS):
    inr, npath, nds = pick_pred(near, C)
    cells = [(pick_true(clean, C), f'ImageNet ({idx2name[C]})', BLUE),
             (pick_true(csr, C), f'ImageNet-R ({idx2name[C]})', CYAN),
             (inr, f'{nds} ({near_name(npath)})', ORANGE),
             (pick_far(far, 'Textures' if r == 0 else 'iNaturalist'), 'Textures' if r == 0 else 'iNaturalist', RED)]
    for c, (im, lab, col) in enumerate(cells):
        if im is None: print('MISSING row', r, 'col', c, lab)
        ax = axes[r, c]; ax.imshow(safe(im)); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(col); sp.set_linewidth(3.4)
        ax.set_xlabel(lab, fontsize=11, color='#333')
        if r == 0:
            ax.set_title(HEAD[c][0], fontsize=14, color=HEAD[c][1], fontweight='bold', pad=9)

fig.text(0.30, 0.96, 'accepted as ID (full-spectrum)', ha='center', fontsize=12.5, color='#2c6fb0', style='italic')
fig.text(0.72, 0.96, 'rejected as OOD', ha='center', fontsize=12.5, color='#cc3b3b', style='italic')
fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=3.2)
os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, 'group_examples_real.png')
fig.savefig(out, dpi=190)
print('saved', out, '| rows:', [idx2name[C] for C in ROWS])
