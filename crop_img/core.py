"""Crop a 4D NIfTI image to the bounding box of a 3D mask, with FSL-compatible transforms."""

import os

import nibabel as nib
import numpy as np


def compute_bbox(mask_data, pad, shape):
    """Bounding box (inclusive start, exclusive stop) per axis of nonzero mask voxels, padded and clipped."""
    if np.isscalar(pad):
        pad = (pad, pad, pad)

    nonzero = np.argwhere(mask_data > 0)
    if nonzero.size == 0:
        raise ValueError("Mask is empty (no nonzero voxels)")

    mins = nonzero.min(axis=0)
    maxs = nonzero.max(axis=0) + 1  # exclusive

    starts = [max(0, mins[ax] - pad[ax]) for ax in range(3)]
    stops = [min(shape[ax], maxs[ax] + pad[ax]) for ax in range(3)]
    return starts, stops


def crop_image(img, mask_img, pad=0):
    """Crop a 4D (or 3D) image to the padded bounding box of a 3D mask.

    Returns (cropped_img, starts, stops).
    """
    data = img.get_fdata(dtype=np.float32)
    mask_data = mask_img.get_fdata(dtype=np.float32)

    if img.shape[:3] != mask_img.shape[:3]:
        raise ValueError(f"Image spatial shape {img.shape[:3]} != mask shape {mask_img.shape[:3]}")
    if not np.allclose(img.affine, mask_img.affine, atol=1e-4):
        raise ValueError("Image and mask affines do not match")

    starts, stops = compute_bbox(mask_data, pad, img.shape[:3])
    slices = tuple(slice(s, e) for s, e in zip(starts, stops))

    cropped_data = data[slices] if data.ndim == 3 else data[slices + (slice(None),)]
    cropped_data = cropped_data.astype(data.dtype, copy=False)

    new_affine = img.affine.copy()
    new_affine[:3, 3] = nib.affines.apply_affine(img.affine, starts)

    new_header = img.header.copy()
    new_header.set_data_shape(cropped_data.shape)

    cropped_img = nib.Nifti1Image(cropped_data, new_affine, header=new_header)
    sform_code = int(img.header["sform_code"]) or 1
    qform_code = int(img.header["qform_code"]) or 1
    cropped_img.set_sform(new_affine, code=sform_code)
    cropped_img.set_qform(new_affine, code=qform_code)

    return cropped_img, starts, stops


def _vox2fsl(affine, shape, zooms):
    """4x4 matrix mapping voxel indices -> FSL's internal 'scaled voxel' coordinates for this image.

    FSL flips the x axis when the affine's linear part has positive determinant
    (i.e. the image is stored "neurological"), since FLIRT's internal convention
    assumes a "radiological" scaled-voxel frame. This mirrors FSL/nibabel's
    documented FLIRT coordinate convention.
    """
    V = np.diag([zooms[0], zooms[1], zooms[2], 1.0])
    if np.linalg.det(affine[:3, :3]) > 0:
        flip = np.eye(4)
        flip[0, 0] = -1
        flip[0, 3] = (shape[0] - 1) * zooms[0]
        V = flip @ V
    return V


def fsl_matrices(full_img, cropped_img):
    """Return (crop2full, full2crop) 4x4 FLIRT-convention matrices.

    crop2full is usable as: flirt -in <cropped> -ref <full> -applyxfm -init crop2full.mat -out ...
    full2crop is its inverse: flirt -in <full> -ref <cropped> -applyxfm -init full2crop.mat -out ...

    Note: pass `-paddingsize 1` to flirt when applying these. Without it, floating-point
    noise (~1e-8 voxels) from the chained matrix inversions can occasionally push the
    outermost boundary voxel just past the source volume's valid index range, and FSL's
    default zero-padding will blank that single edge voxel instead of sampling it. This
    was verified empirically: with -paddingsize 1, round-tripping cropped data through
    crop2full reproduces the original volume with zero difference.
    """
    A_full, A_crop = full_img.affine, cropped_img.affine
    z_full = full_img.header.get_zooms()[:3]
    z_crop = cropped_img.header.get_zooms()[:3]

    V_full = _vox2fsl(A_full, full_img.shape[:3], z_full)
    V_crop = _vox2fsl(A_crop, cropped_img.shape[:3], z_crop)

    crop2full = V_full @ np.linalg.inv(A_full) @ A_crop @ np.linalg.inv(V_crop)
    full2crop = np.linalg.inv(crop2full)
    return crop2full, full2crop


def derive_output_paths(prefix):
    """Derive (out_path, out_crop2full, out_full2crop) from a base path/prefix.

    `prefix` is stripped of a trailing .nii/.nii.gz extension (if any) before
    the standard suffixes are appended; if it has no NIfTI extension, .nii.gz
    is assumed for the cropped image.
    """
    base, ext = _splitext(prefix)
    ext = ext or ".nii.gz"
    return f"{base}_cropped{ext}", f"{base}_crop2full.mat", f"{base}_full2crop.mat"


def crop_to_mask(in_path, mask_path, pad=0, out_path=None, out_crop2full=None, out_full2crop=None, out_prefix=None):
    """Crop `in_path` to the padded bounding box of `mask_path` and write outputs.

    pad: int or (px, py, pz) tuple, in voxels.
    out_prefix: base path used to derive all three output paths (see
        `derive_output_paths`); defaults to `in_path`. Ignored for any output
        whose path is given explicitly via out_path/out_crop2full/out_full2crop.
    Returns (cropped_img_path, crop2full_path, full2crop_path).
    """
    img = nib.load(in_path)
    mask_img = nib.load(mask_path)

    cropped_img, starts, stops = crop_image(img, mask_img, pad=pad)
    crop2full, full2crop = fsl_matrices(img, cropped_img)

    default_path, default_crop2full, default_full2crop = derive_output_paths(out_prefix or in_path)
    out_path = out_path or default_path
    out_crop2full = out_crop2full or default_crop2full
    out_full2crop = out_full2crop or default_full2crop

    nib.save(cropped_img, out_path)
    np.savetxt(out_crop2full, crop2full, fmt="%.10g")
    np.savetxt(out_full2crop, full2crop, fmt="%.10g")

    print(f"bbox voxel start: {starts}, stop: {stops}")
    print(f"cropped image:    {out_path}  shape={cropped_img.shape}")
    print(f"crop -> full mat: {out_crop2full}")
    print(f"full -> crop mat: {out_full2crop}")
    print("note: pass -paddingsize 1 to flirt when applying these matrices (see fsl_matrices() docstring)")

    return out_path, out_crop2full, out_full2crop


def _splitext(path):
    if path.endswith(".nii.gz"):
        return path[: -len(".nii.gz")], ".nii.gz"
    return os.path.splitext(path)
