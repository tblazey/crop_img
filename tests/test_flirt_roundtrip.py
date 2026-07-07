"""End-to-end validation against a real FSL install.

Confirms the FLIRT matrices produced by fsl_matrices() are correct by actually
invoking flirt: applying crop2full to the cropped volume should reproduce the
original data exactly within the cropped region.
"""

import shutil
import subprocess

import nibabel as nib
import numpy as np
import pytest

from crop_img.core import crop_image, fsl_matrices

pytestmark = pytest.mark.skipif(shutil.which("flirt") is None, reason="FSL (flirt) not installed")


def test_crop2full_roundtrips_through_flirt(tmp_path, synthetic_images):
    img, mask_img = synthetic_images
    cropped_img, starts, stops = crop_image(img, mask_img, pad=2)
    crop2full, _ = fsl_matrices(img, cropped_img)

    full_vol0 = nib.Nifti1Image(img.get_fdata()[..., 0].astype(np.float32), img.affine)
    crop_vol0 = nib.Nifti1Image(cropped_img.get_fdata()[..., 0].astype(np.float32), cropped_img.affine)

    full_path = tmp_path / "full_vol0.nii.gz"
    crop_path = tmp_path / "crop_vol0.nii.gz"
    mat_path = tmp_path / "crop2full.mat"
    out_path = tmp_path / "check_full.nii.gz"

    nib.save(full_vol0, full_path)
    nib.save(crop_vol0, crop_path)
    np.savetxt(mat_path, crop2full, fmt="%.10g")

    subprocess.run(
        [
            "flirt",
            "-in", str(crop_path),
            "-ref", str(full_path),
            "-applyxfm", "-init", str(mat_path),
            "-interp", "nearestneighbour",
            "-paddingsize", "1",
            "-out", str(out_path),
        ],
        check=True,
        capture_output=True,
    )

    full_data = full_vol0.get_fdata()
    check_data = nib.load(str(out_path)).get_fdata()
    sl = tuple(slice(s, e) for s, e in zip(starts, stops))

    np.testing.assert_allclose(full_data[sl], check_data[sl], atol=1e-4)
