import nibabel as nib
import numpy as np
import pytest


def _rotated_affine():
    """A non-trivial affine: rotation + anisotropic voxel sizes + translation.

    Deliberately not axis-aligned so tests exercise the general case, not just
    the identity-affine special case.
    """
    theta = 0.15
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    zooms = np.array([2.0, 2.5, 3.0])
    affine = np.eye(4)
    affine[:3, :3] = R @ np.diag(zooms)
    affine[:3, 3] = [10, -20, 5]
    return affine


@pytest.fixture
def synthetic_images():
    """A synthetic 4D image + 3D mask sharing a rotated, anisotropic affine."""
    affine = _rotated_affine()
    shape4d = (20, 24, 18, 5)

    rng = np.random.default_rng(0)
    data = rng.random(shape4d).astype(np.float32)
    img = nib.Nifti1Image(data, affine)
    img.header.set_zooms((2.0, 2.5, 3.0, 1.0))

    mask = np.zeros(shape4d[:3], dtype=np.float32)
    mask[5:15, 8:18, 4:14] = 1
    mask_img = nib.Nifti1Image(mask, affine)

    return img, mask_img


@pytest.fixture
def synthetic_paths(tmp_path, synthetic_images):
    """Synthetic image/mask written to disk, as (in_path, mask_path)."""
    img, mask_img = synthetic_images
    in_path = tmp_path / "full4d.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(img, in_path)
    nib.save(mask_img, mask_path)
    return str(in_path), str(mask_path)
