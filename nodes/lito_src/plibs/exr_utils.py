#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements functions to read openexr files.

import os
import tempfile
import typing as T
import zipfile

import Imath
import numpy as np
import OpenEXR

import torch


def read_exr(filename: str, zipfile_obj: zipfile.ZipFile | None = None) -> np.ndarray:
    """
    Read an openexr file that contains 1 (V), 3 (RGB), or 4 (RGBA) channels.
    Can read from filesystem or from within a zip archive.

    Args:
        filename: filename of the exr file (or path within zip)
        zip_path: path to zip file containing the exr (optional)

    Returns:
        (h, w, c) float32
    """

    if zipfile_obj is not None:
        # Extract file from zip to temporary location
        with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as tmp_file:
            tmp_file.write(zipfile_obj.read(filename))
            temp_filename = tmp_file.name

        try:
            # Read the temporary file
            result = _read_exr_file(temp_filename)
        finally:
            # Clean up temporary file
            os.unlink(temp_filename)

        return result
    else:
        # Read directly from filesystem
        return _read_exr_file(filename)


def _read_exr_file(filename: str) -> np.ndarray:
    """Helper function containing the original EXR reading logic"""
    """
    Read an openexr file that contains 1 (V), 3 (RGB), or 4 (RGBA) channels.

    Args:
        filename:
            filename of the exr file

    Returns:
        (h, w, c) float32
    """

    # Open the EXR file
    exr_file = OpenEXR.InputFile(filename)

    try:
        # Get the header to determine the size of the image and available channels
        header = exr_file.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        # Determine the channels available in the file
        channels = header["channels"].keys()

        # Define the channel type (32-bit float)
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

        # Read the channels
        data = {}
        for channel in channels:
            data[channel] = exr_file.channel(channel, FLOAT)

        # Convert the raw string data to numpy arrays and reshape them
        for key in data:
            data[key] = np.frombuffer(data[key], dtype=np.float32).reshape((height, width))

        # Handle different numbers of channels
        if len(channels) == 1:
            # Single channel (e.g., grayscale)
            img = data["V"] if "V" in data else data[list(data.keys())[0]]
            img = np.reshape(img, (height, width, 1))  # (h, w, 1)
        elif len(channels) == 3:
            # Three channels (RGB)
            img = np.stack([data["R"], data["G"], data["B"]], axis=-1)  # (h, w, 3)
        elif len(channels) == 4:
            # Four channels (RGBA)
            img = np.stack([data["R"], data["G"], data["B"], data["A"]], axis=-1)  # (h, w, 4)
        else:
            cs = []
            for ic in range(len(channels)):
                name = f"C{ic}"
                assert name in channels
                cs.append(data[name])
            img = np.stack(cs, axis=-1)  # (h, w, c)

        return img
    finally:
        exr_file.close()


def read_rgb_exr(filename: str) -> np.ndarray:
    """
    Read an openexr file that contains 3 channels (RGB).

    Args:
        filename:
            filename of the exr file

    Returns:
        (h, w, 3rgb) float32
    """

    # Open the EXR file
    exr_file = OpenEXR.InputFile(filename)

    # Get the header to determine the size of the image
    header = exr_file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Define the channel type (32-bit float)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # Read the RGBA channels
    red_str = exr_file.channel("R", FLOAT)
    green_str = exr_file.channel("G", FLOAT)
    blue_str = exr_file.channel("B", FLOAT)

    # Convert the raw string data to numpy arrays
    red = np.frombuffer(red_str, dtype=np.float32).reshape((height, width))  # (h, w)
    green = np.frombuffer(green_str, dtype=np.float32).reshape((height, width))  # (h, w)
    blue = np.frombuffer(blue_str, dtype=np.float32).reshape((height, width))  # (h, w)

    # Stack the channels into a single numpy array
    rgb = np.stack([red, green, blue], axis=-1)  # (h, w, 3)
    return rgb


def read_rgba_exr(filename: str) -> np.ndarray:
    """
    Read an openexr file that contains 4 channels (RGBA).

    Args:
        filename:
            filename of the exr file

    Returns:
        (h, w, 4rgba) float32
    """

    # Open the EXR file
    exr_file = OpenEXR.InputFile(filename)

    # Get the header to determine the size of the image
    header = exr_file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Define the channel type (32-bit float)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # Read the RGBA channels
    red_str = exr_file.channel("R", FLOAT)
    green_str = exr_file.channel("G", FLOAT)
    blue_str = exr_file.channel("B", FLOAT)
    alpha_str = exr_file.channel("A", FLOAT)

    # Convert the raw string data to numpy arrays
    red = np.frombuffer(red_str, dtype=np.float32).reshape((height, width))  # (h, w)
    green = np.frombuffer(green_str, dtype=np.float32).reshape((height, width))  # (h, w)
    blue = np.frombuffer(blue_str, dtype=np.float32).reshape((height, width))  # (h, w)
    alpha = np.frombuffer(alpha_str, dtype=np.float32).reshape((height, width))  # (h, w)

    # Stack the channels into a single numpy array
    rgba = np.stack([red, green, blue, alpha], axis=-1)  # (h, w, 4)
    return rgba


def read_bw_exr(filename: str):
    """
    Read an openexr file that contains 1 channel (V).

    Args:
        filename:
            filename of the exr file

    Returns:
        (h, w, 1)  float32
    """

    # Open the EXR file
    exr_file = OpenEXR.InputFile(filename)

    # Get the header to determine the size of the image
    header = exr_file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Define the channel type
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # Read the depth channel (assuming single channel depth)
    depth_str = exr_file.channel("V", FLOAT)

    # Convert the raw string data to a numpy array
    depth = np.frombuffer(depth_str, dtype=np.float32)
    depth.shape = (height, width, 1)  # Reshape the array to match the image dimensions

    return depth  # (h, w, 1)


def write_exr(filename: str, arr: T.Union[np.ndarray, torch.Tensor]):
    """
    Saves a depth map or RGBA image to an OpenEXR file (float32).

    Args:
        filename (str):
            The output file path for the EXR file.
        arr (np.ndarray):
            (h, w), (h, w, 1), (h, w, 3), (h, w, 4) The image data as a NumPy array.
            if (h, w) or (h, w, 1), the exr file will have 1 channel
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()

    if arr.ndim == 2:
        # Single-channel depth map
        height, width = arr.shape
        channels = {"V": arr}
    elif arr.ndim == 3:
        height, width, c = arr.shape
        if c == 4:
            # RGBA image
            channels = {
                "R": arr[:, :, 0],
                "G": arr[:, :, 1],
                "B": arr[:, :, 2],
                "A": arr[:, :, 3],
            }
        elif c == 3:
            # RGB image
            channels = {
                "R": arr[:, :, 0],
                "G": arr[:, :, 1],
                "B": arr[:, :, 2],
            }
        elif c == 1:
            # depth image
            channels = {"V": arr}
        else:
            # theoretically, we can support saving more channel
            channels = dict()
            for ic in range(c):
                name = f"C{ic}"
                channels[name] = arr[:, :, ic]
    else:
        raise ValueError("Unsupported image format.")

    # Set up the EXR header
    header = OpenEXR.Header(width, height)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    header["channels"] = {c: Imath.Channel(pixel_type) for c in channels}

    # Create an OpenEXR output file
    exr_file = OpenEXR.OutputFile(filename, header)

    # Convert channels to bytes and write to the file
    exr_file.writePixels({c: channels[c].astype(np.float32).tobytes() for c in channels})

    # Close the EXR file
    exr_file.close()
