"""Tests for crop_img.cli."""

import json
from unittest.mock import patch

from crop_img.cli import main


def test_main_writes_args_json_default_prefix(synthetic_paths):
    """With no --out, args.json should be named after in_path's stem."""
    in_path, mask_path = synthetic_paths
    with patch("sys.argv", ["crop_img", in_path, mask_path]):
        main()

    base = in_path[: -len(".nii.gz")]
    saved = json.loads(open(f"{base}_args.json", encoding="utf-8").read())
    assert saved["in_path"] == in_path
    assert saved["mask_path"] == mask_path
    assert saved["pad"] == [0]


def test_main_writes_args_json_with_out_prefix(tmp_path, synthetic_paths):
    """--out should redirect args.json's name too, like the other outputs."""
    in_path, mask_path = synthetic_paths
    out_prefix = str(tmp_path / "sub01")
    with patch(
        "sys.argv",
        ["crop_img", in_path, mask_path, "--out", out_prefix, "--pad", "3"],
    ):
        main()

    saved = json.loads(open(f"{out_prefix}_args.json", encoding="utf-8").read())
    assert saved["out_prefix"] == out_prefix
    assert saved["pad"] == [3]
