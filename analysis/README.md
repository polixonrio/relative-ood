# Thesis Analyses

This directory contains the thesis analyses built on top of OpenOOD.

Runnable entrypoints live under `pipelines/`. They use the fixed thesis
datasets, backbones, cache paths, and output paths declared in
`core/benchmarks.py`.

- `pipelines/step00_build_cache.py`
  Builds fixed benchmark model-output caches: logits, features, predictions,
  labels, and classifier weights.
- `pipelines/step01_method_ranking.py`
  Runs the detector panel from existing model caches and writes
  ranking-stability summaries.
- `pipelines/step02_score_overlap.py`
  Extracts per-sample detector scores and analyzes overlap across clean ID,
  covariate-shifted ID, near-OOD, and far-OOD groups.
  When reusing an existing `raw_metrics.parquet`, it resolves checkpoint paths into
  the stock OpenOOD `results/` tree.
- `pipelines/step03_full_spectrum_collapse.py`
  Reuses step 01 outputs to decompose standard-to-full-spectrum collapse by
  method and family.
- `pipelines/step04_class_conditioning_gap.py`
  Compares global, predicted-class, max-class, and top-2 score structures in
  feature and logit space to test whether class-conditional modeling is really
  where the useful separation lives.
- `pipelines/step05_logit_score_geometry.py`
  Compares logit-score families such as max logit, energy, entropy, raw TSS,
  and rTSS across `clean_id`, `cs_id`, `near_ood`, and `far_ood`, and records
  classwise raw-TSS baseline spread on correct ID train samples.
- `pipelines/step06_neighbor_geometry.py`
  Compares global kNN, predicted-class kNN, and relative-kNN neighbor scores
  across the full-spectrum sample groups to support `RkNN++`-style detector
  design.
- `pipelines/step07_tss_temperature_selection.py`
  Sweeps TSS/rTSS temperature choices over existing model caches and validates
  them on cached ID validation logits.
- `pipelines/step08_threshold_transfer.py`
  Tests whether clean-ID score thresholds transfer to cs-ID, near-OOD, and
  far-OOD score groups.
- `pipelines/step09_near_ood_hardness.py`
  Characterizes near-OOD hardness with feature-covariance alignment, centroid
  distance, and kNN overlap metrics.
- `pipelines/step10_ranking_change_explanation.py`
  Joins ranking changes with existing overlap, threshold-transfer,
  and near-OOD-hardness drivers.

Reusable implementation code lives in `analysis/core/`:

- `benchmarks.py`
  Fixed thesis datasets, paths, protocols, backbones, and sample groups.
- `method_scoring.py`
  Per-method cached scoring through the adapter registry.
- `model_cache.py`
  Unified `ModelCache` reader/builder for logits, final features, labels,
  predictions, and classifier weights.
- `table_store.py`
  Shared Parquet table IO helpers for the analysis pipelines.
- `methods/`
  Single source of truth for the detector panel, family assignments, method
  ordering, plotting colors, method adapters, and rTSS/TSS method code.
- `scoring.py`
  Shared score-overlap, group-summary, and activation-comparison helpers.

Persistent caches are written under:

- `results/cache/model/<dataset>/<network>/<checkpoint_name>__<fingerprint>/`

Within each model cache folder, the analyses also persist:

- `method_setup/`
  Cached train-derived method state such as FAISS indices, Mahalanobis-family
  statistics and VIM eigenspaces.
- `method_scores/`
  Cached per-method per-split score arrays, so warm reruns can skip rescoring.

The cache path encodes the dataset, network, and checkpoint name. The short
fingerprint is derived from checkpoint identity metadata so changed checkpoints
do not collide with old cache folders.
Each split is a directory of plain `.npy` arrays (`logits.npy`,
`features.npy`, `labels.npy`, `preds.npy`), which keeps large train arrays
streamable without custom ZIP parsing.

Datasets are expected in the stock OpenOOD location:

- `data/`

Official download-script checkpoints are expected in the stock OpenOOD
location:

- `results/`

The analysis pipelines follow those OpenOOD paths by default so the repo can be
repopulated with `scripts/download/download.sh` without extra path surgery.

## OpenOOD Source

The thesis analysis code is owned by this repository. The upstream OpenOOD
source tree is treated as restorable vendor code. A fresh clone can restore the
OpenOOD package and supporting scaffold with:

```powershell
python -m analysis.tools.bootstrap_openood
```

Use `--force` when intentionally replacing an existing local OpenOOD checkout:

```powershell
python -m analysis.tools.bootstrap_openood --force
```

The command clones `Jingkang50/OpenOOD`, checks out the requested ref, applies
any patches in `analysis/openood_patches/`, and copies the needed OpenOOD paths
into the repository root. No patches are currently required because the active
analysis code does not modify OpenOOD itself.

## Setup

Run `python -m analysis.tools.bootstrap_openood` first (see [OpenOOD Source](#openood-source) above). The setup scripts below depend on the OpenOOD scaffold it restores, including `scripts/download/download.py` and the root `setup.py`/`pyproject.toml`, none of which are tracked in this repository, so a fresh clone must bootstrap before running setup.

The analysis workflow expects a repo-local environment plus repo-local
`data/` and `results/` trees.

Use the setup scripts with explicit dataset/checkpoint arguments rather than
relying on script defaults.

The setup scripts now auto-select the official PyTorch wheel backend:
- NVIDIA + supported driver/runtime: CUDA wheel (`cu118`, `cu126`, or `cu128`)
- no detectable NVIDIA CUDA runtime: CPU wheel

The setup scripts also install `pyarrow`, so the analysis scripts can emit
Parquet-backed tabular artifacts by default.

ImageNet-200 + ImageNet-1K on Windows:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup\setup_openood_uv.ps1 -Datasets imagenet-200,imagenet-1k -Checkpoints imagenet200_res18_v1.5,imagenet_res50_v1.5
powershell -ExecutionPolicy Bypass -File .\scripts\setup\setup_openood_pip.ps1 -Datasets imagenet-200,imagenet-1k -Checkpoints imagenet200_res18_v1.5,imagenet_res50_v1.5
```

Optional CIFAR-100 standard OpenOOD auxiliary setup:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup\setup_openood_uv.ps1 -Datasets cifar-100 -Checkpoints cifar100_res18_v1.5
```

ImageNet-200 + ImageNet-1K on Linux:
```bash
./scripts/setup/setup_openood.sh --installer pip --datasets imagenet-200,imagenet-1k --checkpoints imagenet200_res18_v1.5,imagenet_res50_v1.5
./scripts/setup/setup_openood.sh --installer uv --datasets imagenet-200,imagenet-1k --checkpoints imagenet200_res18_v1.5,imagenet_res50_v1.5
```

Optional CIFAR-100 standard OpenOOD auxiliary setup:
```bash
./scripts/setup/setup_openood.sh --installer uv --datasets cifar-100 --checkpoints cifar100_res18_v1.5
```

On reruns, the setup scripts reuse a verified existing `.venv` by default.
Use `ForceReinstall` only when you intentionally want to reinstall the repo
dependencies.

Useful flags:
```text
-ForceRecreateVenv / --force-recreate-venv
-ForceReinstall / --force-reinstall
-SkipInstall / --skip-install
-SkipDownload / --skip-download
-SkipLibMR / --skip-libmr
-TorchBackend auto|cpu|cu118|cu126|cu128 / --torch-backend auto|cpu|cu118|cu126|cu128
-PythonVersion 3.11 / --python 3.11
```

For the ImageNet-200 setup in this repo, the downloader also fetches an extra
train-subset tarball so train-statistics methods such as KNN,
Mahalanobis-family variants, VIM, and DICE can run from the same setup.

Useful environment overrides:
```text
OPENOOD_IMAGENET200_TRAIN_SUBSET_URL=https://pub-10540a2ddc5f48e897747ae38578d26a.r2.dev/imagenet200_train_subset.tar
OPENOOD_SKIP_IMAGENET200_TRAIN_SUBSET=1
```

Approximate disk usage in this layout:
- ImageNet-200 benchmark assets + train subset + released checkpoints: about `46.3 GB`
- ImageNet-200 + ImageNet-1K benchmark assets + train subset + released checkpoints is substantially larger and depends on whether the full ImageNet train split is already present locally.

## Run Commands

Step 00: build model-output cache
```powershell
.\.venv\Scripts\python.exe -m analysis.pipelines.step00_build_cache
```

Step 00 uses the fixed thesis layout: `data/`, `results/`, and
`results/cache/model/`.

Step 01: checkpoint ranking stability from cache
```powershell
.\.venv\Scripts\python.exe -m analysis.pipelines.step01_method_ranking
```

Step 01 writes the active ranking table to
`results/ranking_stability/raw_metrics.parquet`.

Step 02: score overlap
```powershell
.\.venv\Scripts\python.exe -m analysis.pipelines.step02_score_overlap
```

Step 02 stores its large per-sample score table in:
- `results/score_overlap/<dataset>/per_sample_scores_parts/`

Those shards are written as Parquet.

Step 03: full-spectrum collapse
```powershell
.\.venv\Scripts\python.exe -m analysis.pipelines.step03_full_spectrum_collapse
```

Step 04: class conditioning gap
```powershell
.\.venv\Scripts\python.exe -m analysis.pipelines.step04_class_conditioning_gap
```

Step 05: logit score geometry
```powershell
.\.venv\Scripts\python.exe -m analysis.pipelines.step05_logit_score_geometry
```

Step 06: neighbor geometry
```powershell
.\.venv\Scripts\python.exe -m analysis.pipelines.step06_neighbor_geometry
```

Fixed thesis backbones are `cifar100 -> resnet18_32x32`,
`imagenet200 -> resnet18_224x224`, and `imagenet -> resnet50`.

Remaining steps:
```powershell
.\.venv\Scripts\python.exe -m analysis.pipelines.step07_tss_temperature_selection
.\.venv\Scripts\python.exe -m analysis.pipelines.step08_threshold_transfer
.\.venv\Scripts\python.exe -m analysis.pipelines.step09_near_ood_hardness
.\.venv\Scripts\python.exe -m analysis.pipelines.step10_ranking_change_explanation
.\.venv\Scripts\python.exe -m analysis.pipelines.step99_export_thesis_artifacts
```

## Fast Warm-Cache Reruns

After step 00 has built the model cache for a dataset, step 01 and downstream
analyses reuse those arrays directly. Step 01 writes
`results/ranking_stability/raw_metrics.parquet`; later steps read that fixed
table and write fixed per-dataset output folders.

