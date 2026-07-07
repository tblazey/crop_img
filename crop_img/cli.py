"""Command-line interface for crop_img."""

import argparse

from crop_img.core import crop_to_mask


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Crop a 4D NIfTI image to a 3D mask's bounding box. Always writes the "
            "cropped image ('<in_path>_cropped.nii.gz') plus crop2full/full2crop "
            "FLIRT .mat transforms ('<in_path>_crop2full.mat' / '<in_path>_full2crop.mat')."
        )
    )
    parser.add_argument("in_path", help="Input 4D (or 3D) NIfTI image")
    parser.add_argument("mask_path", help="3D brain mask NIfTI, same space as input")
    parser.add_argument(
        "--pad",
        type=int,
        nargs="+",
        default=[0],
        help="Padding in voxels: one value for all axes, or three values for x y z (default: 0)",
    )
    args = parser.parse_args()

    if len(args.pad) == 1:
        pad = args.pad[0]
    elif len(args.pad) == 3:
        pad = tuple(args.pad)
    else:
        parser.error("--pad takes either 1 or 3 values")

    crop_to_mask(args.in_path, args.mask_path, pad=pad)


if __name__ == "__main__":
    main()
