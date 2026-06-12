# OpenOOD Patches

No OpenOOD patches are currently required.

If a local OpenOOD source change becomes necessary, keep the change as a patch
file in this directory instead of committing the generated OpenOOD tree. The
bootstrap command applies `*.patch` files before copying OpenOOD into the
workspace root.

One way to create a patch after editing a clean OpenOOD checkout is:

```bash
git diff > analysis/openood_patches/descriptive_name.patch
```

Then restore the source tree with:

```bash
python -m analysis.tools.bootstrap_openood --force
```
