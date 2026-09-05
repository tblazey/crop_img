# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Install (editable, with dev deps):
```
pip install -e ".[dev]"
```

Run all tests:
```
pytest
```

Run a single test:
```
pytest tests/test_core.py::TestFslMatrices::test_pure_crop_is_a_translation_in_fsl_space
```

Run the CLI (installed as a console script via `pip install -e .`):
```
crop_img func4d.nii.gz brainmask.nii.gz --pad 5
```
or without installing:
```
python -m crop_img.cli func4d.nii.gz brainmask.nii.gz --pad 5
```

`in_path`/`mask_path` are positional. Outputs are always named
`<out>_cropped.nii.gz`, `<out>_crop2full.mat`, `<out>_full2crop.mat`, and
`<out>_args.json` (a record of the parsed CLI arguments, for provenance),
where `<out>` defaults to `in_path` but can be overridden with `--out` (a
single base path/prefix applied to all outputs) — e.g. `crop_img
func4d.nii.gz brainmask.nii.gz --out results/sub01`. The underlying
`crop_to_mask()` function accepts the same prefix via `out_prefix`, plus
`out_path`/`out_crop2full`/`out_full2crop` to override individual output
paths directly for programmatic use; only the CLI wrapper omits the
per-file overrides. `<out>_args.json` is written by `cli.py` itself (not
`crop_to_mask()`), using the same `_splitext`-based prefix logic as
`derive_output_paths()` in `core.py`.

`tests/test_flirt_roundtrip.py` requires a real FSL install (`flirt` on `PATH`) and is
auto-skipped otherwise — no mocking is used for that test, since it exists specifically
to catch numerical regressions that only manifest when going through FSL's actual
interpolation code.

## Architecture

Package layout is flat `crop_img/` (no `src/` dir, matching `mat_resolve`'s and
`cmr_opt`'s layout): `core.py` holds all the math/IO logic; `cli.py` is a thin
argparse wrapper around `core.crop_to_mask()`.

The core operation has two parts that must stay in sync:

1. **Cropping** (`crop_image`): slices the array to the padded bounding box of the
   mask, and rewrites the affine's translation (`new_affine[:3,3]`) so world
   coordinates of shared voxels are unchanged — the rotation/scale part of the
   affine is left untouched. This is what makes the crop a pure translation with
   no resampling.

2. **FSL transform generation** (`fsl_matrices`): FSL's FLIRT does *not* use
   NIfTI world coordinates directly. It uses its own "scaled voxel" coordinate
   convention (voxel index × voxel size), and internally flips the x-axis
   whenever the affine's linear part has a positive determinant
   ("neurological" storage) — this is the `_vox2fsl` flip logic. The general
   conversion formula used here (`crop2full = V_full @ inv(A_full) @ A_crop @ inv(V_crop)`)
   is the standard way to build a FLIRT matrix from two images whose world
   coordinates already correspond (i.e. no resampling actually happened —
   exactly the crop/pad case).

   **Known gotcha, already handled/documented in code**: floating-point error
   from the chained matrix inversions (~1e-8 voxels) can push the outermost
   boundary voxel just past the source volume's valid index range. FSL's
   `flirt` then zero-fills that single edge voxel instead of sampling it,
   *regardless of interpolation method* (nearest-neighbour and trilinear both
   affected identically) — it's a source-boundary clamp, not an interpolation
   artifact. The fix is always `-paddingsize 1` when applying these matrices
   with `flirt`; this was verified empirically to produce zero difference
   end-to-end and is asserted by `tests/test_flirt_roundtrip.py`. Don't remove
   `-paddingsize 1` from that test or from usage docs without re-validating
   against real FSL.

Tests use a shared fixture (`tests/conftest.py::synthetic_images`) built with a
deliberately non-axis-aligned, anisotropic affine (rotation + different voxel
sizes per axis) rather than identity, so tests exercise the general case rather
than accidentally passing only because rotation/scale is trivial.
