"""Command-line interface for crop_img."""

import argparse

from crop_img.core import crop_to_mask


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Crop a 4D NIfTI image to a 3D mask's bounding box. Writes the "
            "cropped image plus crop2full/full2crop FLIRT .mat transforms, named "
            "'<out>_cropped.nii.gz', '<out>_crop2full.mat', and '<out>_full2crop.mat', "
            "where <out> defaults to <in_path> but can be overridden with --out."
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
    parser.add_argument(
        "--out",
        dest="out_prefix",
        help=(
            "Base path used for all outputs (default: <in_path>). Writes "
            "'<out>_cropped.nii.gz', '<out>_crop2full.mat', '<out>_full2crop.mat'."
        ),
    )
    args = parser.parse_args()

    if len(args.pad) == 1:
        pad = args.pad[0]
    elif len(args.pad) == 3:
        pad = tuple(args.pad)
    else:
        parser.error("--pad takes either 1 or 3 values")

    crop_to_mask(args.in_path, args.mask_path, pad=pad, out_prefix=args.out_prefix)


if __name__ == "__main__":
    main()
