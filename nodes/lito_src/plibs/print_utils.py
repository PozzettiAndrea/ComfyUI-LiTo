#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import json
import typing as T

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import torch


def imagesc(
    arr: np.ndarray,
    xs: T.Sequence[float] = None,
    ys: T.Sequence[float] = None,
    fig=None,
    axes=None,
    dpi=150,
    colorbar=True,
    vmin: float = None,
    vmax: float = None,
):
    """
    Mimic matlab's imagesc using matplotlib.

    Args:
        arr:
            2D matrix
        xs:
            coordinate of columns (None: use 0~N-1)
        ys:
            coordinate of rows (None: use 0~M-1)

    Returns:
        fig, axes
    """

    if fig is None:
        fig, axes = plt.subplots(dpi=dpi)
    elif axes is None:
        fig.add_axes([0, 0, 1, 1])

    if xs is None:
        xs = np.arange(arr.shape[1])
    if ys is None:
        ys = np.arange(arr.shape[0])

    def extents(ts):
        if len(ts) == 1:
            delta = 1
        else:
            delta = ts[1] - ts[0]
        return [ts[0] - delta / 2, ts[-1] + delta / 2]

    if vmin is None:
        vmin = arr.min()
    if vmax is None:
        vmax = arr.max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.cm.get_cmap("viridis")

    axes.imshow(
        arr,
        aspect="auto",
        interpolation="none",
        extent=extents(xs) + extents(ys[::-1]),
        origin="upper",
        norm=norm,
        cmap=cmap,
    )
    if colorbar:
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
        )

    return fig, axes


def format_number(size) -> str:
    if size >= 1e9:
        size_str = f"{size / 1e9:.2f} G"
    elif size >= 1e6:
        size_str = f"{size / 1e6:.2f} M"
    elif size >= 1e3:
        size_str = f"{size / 1e3:.2f} K"
    else:
        size_str = f"{size}"
    return size_str


def print_shape_and_memory(**kwargs):
    for key, val in kwargs.items():
        print(f"{key}: {val.shape} ({format_number(val.numel())})")


class StatisticsCollector:
    """
    Compute the average and standard deviation of a dictionary
    of values.
    """

    def __init__(self, convert_to_float: bool = True):
        """
        Args:
            convert_to_float:
                whether to convert input values (from Tensor, ndarray) to float
        """
        self.convert_to_float = convert_to_float
        self.x_mean_dict = dict()
        self.x2_mean_dict = dict()
        self.x_count_dict = dict()

    def record(
        self,
        val_dict: T.Dict[str, float],
    ):
        """
        Record the values in val_dict

        Args:
            val_dict:
                a dictionary containing the floats to compute statistics.
        """

        for key, val in val_dict.items():
            if isinstance(val, torch.Tensor) and val.numel() > 1:
                continue

            if self.convert_to_float:
                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu().item()
                elif isinstance(val, np.ndarray):
                    val = val.item()
                elif isinstance(val, int):
                    val = float(val)
                elif isinstance(val, float):
                    pass
                else:
                    raise NotImplementedError(f"{type(val)}")

            if key not in self.x_mean_dict:
                self.x_mean_dict[key] = val
                self.x2_mean_dict[key] = val**2
                self.x_count_dict[key] = 1
            else:
                self.x_mean_dict[key] = (self.x_mean_dict[key] * self.x_count_dict[key] + val) / (
                    self.x_count_dict[key] + 1
                )
                self.x2_mean_dict[key] = (self.x2_mean_dict[key] * self.x_count_dict[key] + val**2) / (
                    self.x_count_dict[key] + 1
                )
                self.x_count_dict[key] = self.x_count_dict[key] + 1

    def compute_statistics(self) -> T.Dict[str, T.Dict[str, float]]:
        """Compute the statistics.

        Returns:
            a dictionary containing the current statistics
            `"mean"`
            `"std"`
            `"variance"`
            `"second_moment"`
            `"count"`
        """

        mean_dict = dict()
        second_moment_dict = dict()
        variance_dict = dict()
        std_dict = dict()
        count_dict = dict()

        for key in self.x_mean_dict:
            mean = self.x_mean_dict[key]
            second_moment = self.x2_mean_dict[key]
            variance = second_moment - mean**2
            std = variance**0.5

            mean_dict[key] = mean
            second_moment_dict[key] = second_moment
            variance_dict[key] = variance
            std_dict[key] = std
            count_dict[key] = self.x_count_dict[key]

        return dict(
            mean=mean_dict,
            std=std_dict,
            variance=variance_dict,
            second_moment=second_moment_dict,
            count=count_dict,
        )
