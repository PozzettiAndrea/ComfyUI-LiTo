#
# Copyright (C) 2025 Apple Inc. All rights reserved.
#
# The file implements utils for loading images.

import io
import typing as T

from packaging import version

import torch
import torchvision
import torchvision.transforms.v2


def imread(
    filename: T.Union[str, io.BytesIO],
    mode: str,
) -> torch.Tensor:
    """
    Load an image "UNCHANGED", ie, if uint8, the output will be uint8 (0-255),
    if uint16, the output will be uint16 (0-65535), if float32, the output will be float32, etc.

    Args:
        filename:
            filename of the image or raw byte (uint8) tensor
        mode:
            'unchanged':
                return the raw image in original dtype (not the best idea) as
                pytorch does not support uint16 well (ie, no torch.max()).
            'converted':
                convert the image to torch.float32, keep the raw values
            'scaled':
                convert the image to torch.float32, and scale the max dtype value to 1.
                E.g, in the case of uint16, 65535 is mapped to 1.

    Returns:
        (c, h, w). When the image is grayscale, c = 1.
    """

    img = imread_unchanged(filename)  # (c, h, w)
    if mode == "unchanged":
        return img
    elif mode == "converted":
        return torchvision.transforms.v2.functional.to_dtype(
            img,
            dtype=torch.float32,
            scale=False,
        )
    elif mode == "scaled":
        if version.parse(torchvision.__version__) >= version.parse("0.20.0"):
            return torchvision.transforms.v2.functional.to_dtype(
                img,
                dtype=torch.float32,
                scale=True,
            )
        else:
            # there is bug in torchvision that it does not scale uint16 properly
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            elif img.dtype == torch.uint16:
                img = img.float() / 65535.0
            elif img.is_floating_point():
                pass
            else:
                raise NotImplementedError(f"img.dtype: {img.dtype}")
            return img
    else:
        raise NotImplementedError


def imread_unchanged(
    filename: T.Union[str, io.BytesIO],
) -> torch.Tensor:
    """
    Load an image "UNCHANGED", ie, if uint8, the output will be uint8 (0-255),
    if uint16, the output will be uint16 (0-65535), if float32, the output will be float32, etc.

    Args:
        filename:
            filename of the image, or unit8 raw byte tensor

    Returns:
        (c, h, w). When the image is grayscale, c = 1.

    Notes:
        only torchvision.version >= 0.20.0 (ie pytorch 2.5)
    """

    if is_qoi(filename):
        import qoi

        if hasattr(filename, "read") or isinstance(filename, (bytes, bytearray, torch.Tensor)):
            if hasattr(filename, "read"):
                data = filename.read()
            elif isinstance(filename, torch.Tensor):
                data = filename.detach().cpu().contiguous().numpy().tobytes()
            else:
                data = filename
            arr = qoi.decode(data)  # (h, w, 3/4) uint8
        else:
            arr = qoi.read(filename)
        img = torch.from_numpy(arr).permute(2, 0, 1)  # (c, h, w) uint8
        return img

    if version.parse(torchvision.__version__) >= version.parse("0.20.0"):
        # support 8-bit and 16-bit
        if hasattr(filename, "read") or isinstance(filename, (bytes, bytearray)):
            # Handle file-like objects or bytes data (e.g., from zip)
            if hasattr(filename, "read"):
                data = filename.read()
            else:
                data = filename
            # Copy the data to make it writable for torch.frombuffer
            data_copy = bytearray(data)
            img = torchvision.io.decode_image(
                torch.frombuffer(data_copy, dtype=torch.uint8),
                mode=torchvision.io.ImageReadMode.UNCHANGED,
            )  # (c, h, w), c=1 if grayscale

        else:
            # Handle regular file paths
            img = torchvision.io.decode_image(
                filename,  # need torchvision version >= 0.20 to take str
                mode=torchvision.io.ImageReadMode.UNCHANGED,
            )  # (c, h, w), c=1 if grayscale
    else:
        import cv2

        if isinstance(filename, str):
            arr = cv2.imread(filename, cv2.IMREAD_UNCHANGED)  # (h, w, c) or (h, w)
        elif isinstance(filename, torch.Tensor):
            filename = filename.cpu().to(torch.uint8).flatten().numpy()
            arr = cv2.imdecode(filename, cv2.IMREAD_UNCHANGED)
        else:
            raise NotImplementedError(f"filename: {filename}")

        if arr.ndim == 2:
            arr = arr[:, :, None]  # (h, w, 1)

        # opencv reads as bgr
        if arr.shape[-1] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif arr.shape[-1] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)

        img = torch.from_numpy(arr).permute(2, 0, 1)  # (c, h, w)

    return img


def _looks_like_qoi_header(head: bytes) -> bool:
    if len(head) < 14:
        return False
    if head[:4] != b"qoif":
        return False
    channels = head[12]
    colorspace = head[13]
    if channels not in (3, 4):
        return False
    if colorspace not in (0, 1):
        return False
    return True


def is_qoi(filename: T.Union[str, io.BytesIO, bytes, bytearray, torch.Tensor]) -> bool:
    QOI_MIN_LEN = 14 + 8
    QOI_END_MARKER = b"\x00" * 7 + b"\x01"

    if isinstance(filename, torch.Tensor):
        filename = filename.detach().cpu().contiguous().numpy().tobytes()

    # Case 1: path string -> check extension only (cheap).
    # If you want to verify actual bytes for paths too, see note below.
    if isinstance(filename, str):
        return filename.lower().endswith(".qoi")

    # Case 2: raw bytes-like
    if isinstance(filename, (bytes, bytearray)):
        data = bytes(filename)
        if len(data) < QOI_MIN_LEN:
            return False
        if not _looks_like_qoi_header(data[:14]):
            return False
        # End marker is part of the spec; this is still a "simple" strong check.
        return data[-8:] == QOI_END_MARKER

    # Case 3: file-like (io.BytesIO)
    if hasattr(filename, "read"):
        f = filename

        # Try to read without consuming (seek/tell)
        try:
            pos = f.tell()
            head = f.read(14)
            if not _looks_like_qoi_header(head):
                f.seek(pos)
                return False

            # If we can also check end marker cheaply:
            f.seek(0, io.SEEK_END)
            end_pos = f.tell()
            if end_pos < QOI_MIN_LEN:
                f.seek(pos)
                return False
            f.seek(end_pos - 8)
            tail = f.read(8)
            f.seek(pos)
            return tail == QOI_END_MARKER
        except Exception:
            # Non-seekable stream: fall back to reading all bytes (consumes it).
            data = f.read()
            if len(data) < QOI_MIN_LEN:
                return False
            if not _looks_like_qoi_header(data[:14]):
                return False
            return data[-8:] == QOI_END_MARKER

    return False
