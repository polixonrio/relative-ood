# OOD Thesis Analysis

This repository contains the thesis analysis code built around OpenOOD. The
analysis implementation lives under `analysis/`. Upstream OpenOOD source is
treated as restorable vendor code rather than project-owned code.

Restore the OpenOOD source tree before running the analysis:

```bash
python -m analysis.tools.bootstrap_openood
```

Use `--force` when replacing an existing local OpenOOD checkout:

```bash
python -m analysis.tools.bootstrap_openood --force
```

The bootstrap command clones `Jingkang50/OpenOOD`, applies any patches from
`analysis/openood_patches/`, and copies the OpenOOD package plus the supporting
paths needed by the thesis setup and analysis scripts.

The active pipeline documentation is in `analysis/README.md`.
