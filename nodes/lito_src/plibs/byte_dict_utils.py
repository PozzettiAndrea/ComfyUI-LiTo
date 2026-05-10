#
# Copyright (C) 2025 Apple Inc. All rights reserved.
#
# The file implements util functions to use byte dict from webdataset.

import io
import json
import os
from pathlib import Path
import typing as T

import numpy as np
import qoi

import torch

from plibs import exr_utils, img_utils


def convert_to_flat_structure_filename(filename: str, replacement: str = "-") -> str:
    """
    convert folder "/" to "-".
    """
    filename_p = Path(filename)
    flat_name = replacement.join(filename_p.parts)
    return flat_name


def load_file_from_byte_dict(
    byte_dict: T.Dict[str, T.Any],
    filename: str,
    start_path: str = None,
):
    """
    Load from the byte dict returned by webdataset.

    Args:
        byte_dict:
            key:
                the filename relative to index.json, folder structure ("/") is replaced by "-"
                e.g., "xyz_w-xyz_w_0.npy", 'rgbd_sphere-000000-rgb_000192.png'
            value:
                the byte content of the file
        filename:
            non-flat filename
    """
    if start_path is not None:
        filename = os.path.relpath(filename, start=start_path)
    key = convert_to_flat_structure_filename(filename)
    if filename.endswith(".json"):
        out = json.loads(byte_dict[key].decode("utf-8"))
    elif filename.endswith(".npy"):
        out = np.load(io.BytesIO(byte_dict[key]), allow_pickle=False)
    elif filename.endswith(".npz"):
        out = dict()
        with np.load(io.BytesIO(byte_dict[key]), allow_pickle=True) as loader:
            for _key in loader.files:
                out[_key] = loader[_key]
    else:
        raise NotImplementedError

    return out


def load_single_rgbd_file_from_byte_dict(
    byte_dict: T.Dict[str, T.Any],
    filename: str,
    attr_name: str,
    start_path: str = None,
    convert_to_flat_structure: bool = True,
) -> torch.Tensor:
    r"""
    Read a single file (rgb, depth, hit_map, or normal map) of
    an rgbd_image from byte dict

    Args:
        filename:
            filename of the file to be read. I.e, the key in byte_dict to read.
        attr_name:
            what kind of file is it:
            'rgb', 'depth', 'normal_w', 'hit_map'
        start_path:
            The prefix part of filename will be removed to form the actual key
        convert_to_flat_structure:
            Whether to replace all file structure (e.g, "/") to "-"

    Returns:
        (h, w) or (h, w, d) torch.Tensor
    """

    if start_path is not None:
        filename = os.path.relpath(filename, start=start_path)
    if convert_to_flat_structure:
        key = convert_to_flat_structure_filename(filename)

    assert key in byte_dict, f"{key} not in byte_dict"

    if filename.endswith(".npy"):
        arr = np.load(io.BytesIO(byte_dict[key]), allow_pickle=False)  # (h, w) or (h, w, 3)
        arr = torch.from_numpy(arr)
    elif filename.endswith(".exr"):
        arr = exr_utils.read_exr(io.BytesIO(byte_dict[key]))  # (h, w, c)
        arr = torch.from_numpy(arr)
    elif filename.endswith(".png") or filename.endswith(".qoi"):
        # Copy the data to make it writable for torch.frombuffer
        data_copy = bytearray(byte_dict[key])
        arr = img_utils.imread(
            filename=torch.frombuffer(data_copy, dtype=torch.uint8),
            mode="scaled",
        )  # (c, h, w) float32 [0, 1]
        if arr.size(0) == 1:
            arr = arr.squeeze(0)  # (h, w)
        else:
            arr = arr.permute(1, 2, 0)  # (h, w, c)

        if attr_name == "normal_w":
            assert len(arr.shape) == 3
            assert arr.shape[2] == 3 or arr.shape[2] == 4
            # when saved as qoi, normal_w's alpha channel is hit_map
            arr[..., :3] = arr[..., :3] * 2 - 1
            arr[..., :3] = torch.nn.functional.normalize(arr[..., :3], dim=-1)

        elif attr_name == "depth":
            filename2 = f"{os.path.splitext(filename)[0]}.pnginfo"
            key2 = convert_to_flat_structure_filename(filename2)
            assert key2 in byte_dict, f"{key2} not in byte_dict"
            lines = byte_dict[key2].decode("utf-8").splitlines()
            assert len(lines) == 2
            min_arr = float(lines[0].strip())  # Convert the first line to an integer
            max_arr = float(lines[1].strip())  # Convert the second line to an integer
            arr = arr * (max_arr - min_arr) + min_arr
        elif attr_name == "hit_map":
            arr = arr > 0.5  # make sure it is bool

    else:
        raise NotImplementedError(filename)
    return arr
