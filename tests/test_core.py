import nibabel as nib
import numpy as np
import pytest

from crop_img.core import compute_bbox, crop_image, crop_to_mask, fsl_matrices


class TestComputeBbox:
    def test_basic_bbox_no_pad(self, synthetic_images):
        _, mask_img = synthetic_images
        mask_data = mask_img.get_fdata()
        starts, stops = compute_bbox(mask_data, pad=0, shape=mask_img.shape)
        assert starts == [5, 8, 4]
        assert stops == [15, 18, 14]

    def test_pad_expands_bbox(self, synthetic_images):
        _, mask_img = synthetic_images
        mask_data = mask_img.get_fdata()
        starts, stops = compute_bbox(mask_data, pad=2, shape=mask_img.shape)
        assert starts == [3, 6, 2]
        assert stops == [17, 20, 16]

    def test_per_axis_pad(self, synthetic_images):
        _, mask_img = synthetic_images
        mask_data = mask_img.get_fdata()
        starts, stops = compute_bbox(mask_data, pad=(1, 2, 3), shape=mask_img.shape)
        assert starts == [4, 6, 1]
        assert stops == [16, 20, 17]

    def test_pad_clips_to_volume_bounds(self, synthetic_images):
        _, mask_img = synthetic_images
        mask_data = mask_img.get_fdata()
        starts, stops = compute_bbox(mask_data, pad=1000, shape=mask_img.shape)
        assert starts == [0, 0, 0]
        assert stops == list(mask_img.shape)

    def test_empty_mask_raises(self):
        mask_data = np.zeros((10, 10, 10))
        with pytest.raises(ValueError, match="empty"):
            compute_bbox(mask_data, pad=0, shape=mask_data.shape)


class TestCropImage:
    def test_shape_matches_bbox(self, synthetic_images):
        img, mask_img = synthetic_images
        cropped_img, starts, stops = crop_image(img, mask_img, pad=2)
        expected_shape = tuple(e - s for s, e in zip(starts, stops)) + (img.shape[3],)
        assert cropped_img.shape == expected_shape

    def test_data_matches_source_region(self, synthetic_images):
        img, mask_img = synthetic_images
        cropped_img, starts, stops = crop_image(img, mask_img, pad=2)
        sl = tuple(slice(s, e) for s, e in zip(starts, stops))
        np.testing.assert_array_equal(cropped_img.get_fdata(), img.get_fdata()[sl])

    def test_affine_maps_to_same_world_coords(self, synthetic_images):
        """Voxel (0,0,0) in the crop must land at the same world coordinate as
        voxel `starts` in the full image -- this is the whole point of adjusting
        the affine's translation rather than just slicing the array."""
        img, mask_img = synthetic_images
        cropped_img, starts, stops = crop_image(img, mask_img, pad=2)
        world_full = nib.affines.apply_affine(img.affine, starts)
        world_crop = nib.affines.apply_affine(cropped_img.affine, [0, 0, 0])
        np.testing.assert_allclose(world_full, world_crop, atol=1e-6)

    def test_rotation_part_of_affine_is_unchanged(self, synthetic_images):
        img, mask_img = synthetic_images
        cropped_img, _, _ = crop_image(img, mask_img, pad=2)
        np.testing.assert_allclose(cropped_img.affine[:3, :3], img.affine[:3, :3])

    def test_spatial_shape_mismatch_raises(self, synthetic_images):
        img, mask_img = synthetic_images
        bad_mask = nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), mask_img.affine)
        with pytest.raises(ValueError, match="shape"):
            crop_image(img, bad_mask)

    def test_affine_mismatch_raises(self, synthetic_images):
        img, mask_img = synthetic_images
        bad_affine = mask_img.affine.copy()
        bad_affine[0, 3] += 100
        bad_mask = nib.Nifti1Image(mask_img.get_fdata().astype(np.float32), bad_affine)
        with pytest.raises(ValueError, match="affine"):
            crop_image(img, bad_mask)


class TestFslMatrices:
    def test_crop2full_and_full2crop_are_inverses(self, synthetic_images):
        img, mask_img = synthetic_images
        cropped_img, _, _ = crop_image(img, mask_img, pad=2)
        crop2full, full2crop = fsl_matrices(img, cropped_img)
        np.testing.assert_allclose(crop2full @ full2crop, np.eye(4), atol=1e-6)

    def test_pure_crop_is_a_translation_in_fsl_space(self, synthetic_images):
        """Cropping doesn't rotate or resample -- only the origin moves -- so the
        FLIRT matrix between crop and full space should have an identity 3x3
        (no rotation/scale), only a translation component."""
        img, mask_img = synthetic_images
        cropped_img, _, _ = crop_image(img, mask_img, pad=2)
        crop2full, _ = fsl_matrices(img, cropped_img)
        np.testing.assert_allclose(crop2full[:3, :3], np.eye(3), atol=1e-6)

    def test_matches_known_voxel_offset(self, synthetic_images):
        """Applying crop2full (via the FSL scaled-voxel convention) to a crop
        voxel index should recover the corresponding full-image voxel index."""
        img, mask_img = synthetic_images
        cropped_img, starts, _ = crop_image(img, mask_img, pad=2)
        crop2full, _ = fsl_matrices(img, cropped_img)

        from crop_img.core import _vox2fsl

        z_full = img.header.get_zooms()[:3]
        z_crop = cropped_img.header.get_zooms()[:3]
        V_full = _vox2fsl(img.affine, img.shape[:3], z_full)
        V_crop = _vox2fsl(cropped_img.affine, cropped_img.shape[:3], z_crop)

        for crop_vox in ([0, 0, 0], [3, 5, 2], [13, 13, 13]):
            crop_fsl = V_crop @ np.array([*crop_vox, 1.0])
            full_fsl = crop2full @ crop_fsl
            full_vox = np.linalg.inv(V_full) @ full_fsl
            expected = np.array(starts) + np.array(crop_vox)
            np.testing.assert_allclose(full_vox[:3], expected, atol=1e-6)


class TestCropToMask:
    def test_end_to_end_writes_expected_files(self, tmp_path, synthetic_paths):
        in_path, mask_path = synthetic_paths
        out_path = tmp_path / "cropped.nii.gz"
        out_c2f = tmp_path / "c2f.mat"
        out_f2c = tmp_path / "f2c.mat"

        crop_to_mask(
            in_path,
            mask_path,
            pad=2,
            out_path=str(out_path),
            out_crop2full=str(out_c2f),
            out_full2crop=str(out_f2c),
        )

        assert out_path.exists()
        assert out_c2f.exists()
        assert out_f2c.exists()

        cropped_img = nib.load(str(out_path))
        assert cropped_img.shape == (14, 14, 14, 5)

        mat = np.loadtxt(out_c2f)
        assert mat.shape == (4, 4)

    def test_default_output_paths_derived_from_input(self, tmp_path, synthetic_paths):
        in_path, mask_path = synthetic_paths
        out_path, out_c2f, out_f2c = crop_to_mask(in_path, mask_path, pad=1)
        try:
            assert out_path == in_path.replace(".nii.gz", "_cropped.nii.gz")
            assert out_c2f == in_path.replace(".nii.gz", "_crop2full.mat")
            assert out_f2c == in_path.replace(".nii.gz", "_full2crop.mat")
        finally:
            for p in (out_path, out_c2f, out_f2c):
                import os

                if os.path.exists(p):
                    os.remove(p)
