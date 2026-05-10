#
# Copyright (C) 2021 Apple Inc. All rights reserved.
# Author: Rick Chang
#
# This file implements the util function for layers.py.

from abc import ABC
from collections.abc import MutableMapping
import enum
from enum import Enum
import typing as T

import numpy as np

import torch


def init_weight(
    weight: torch.Tensor,
    w_init_gain: str = "linear",
    init_method: str = "xavier_normal",
    lrelu_nslope: float = 0.01,
    kaiming_fan_mode: str = "fan_in",
):
    """
    A helper function to initialize the weights of a linear/convolutional layer.

    Args:
        weight:
            (*), an n-dimensional torch.Tensor to be initialized
        w_init_gain:
            The nonlinearity after the linear layer. Can be chosen from the functions supported by
            torch.nn.init.calculate_gain.
            This includes:
            'linear', 'relu', 'silu', 'leaky_relu', 'relu', 'tanh', 'sigmoid'.
        init_method:
            The initialization method. Can be chosen from:
                'normal':
                    randomly sampeld from a Gaussian distribution
                'uniform':
                    randomly sampeld from a uniform distribution
                'xavier_uniform':
                    Check :py:func:`torch.nn.init.xavier_uniform_`.
                'xavier_normal'
                    Check :py:func:`torch.nn.init.xavier_normal_`.
                'xavier':
                    same as 'xavier_normal'
                'kaiming_uniform':
                    Check :py:func:`torch.nn.init.kaiming_uniform_`.
                'kaiming_normal'
                    Check :py:func:`torch.nn.init.kaiming_normal_`.
                'kaiming':
                    same as 'kaiming_normal'
                'orthogonal':
                    Check :py:func:`torch.nn.init.orthogonal_`.
        lrelu_nslope:
            Negative slope used in the leaky-relu.
        kaiming_fan_mode:
            Fan mode used by kaiming_* init_methods.
    Returns:
        Does not return. The function directly modifies the content of weight.

    Note that the function contains torch.no_grad, so there is no need to wrap it with one.
    """

    # handle silu/swish
    if w_init_gain in {"silu", "swish", "gelu"}:
        # since silu and relu has similar shape, use the gain for relu
        w_init_gain = "relu"

    # calculate gain
    if w_init_gain == "leaky_relu":
        gain = torch.nn.init.calculate_gain(w_init_gain, lrelu_nslope)
        kaiming_a = lrelu_nslope
    else:
        gain = torch.nn.init.calculate_gain(w_init_gain)
        kaiming_a = 0

    if init_method in {
        "kaiming_uniform",
        "kaiming_normal",
        "kaiming",
    } and w_init_gain not in {"relu", "leaky_relu"}:
        print("using kaiming init method on %s, not recommended" % (w_init_gain))

    if init_method == "normal":
        torch.nn.init.normal_(weight, 0.0, gain)
    if init_method == "uniform":
        torch.nn.init.uniform_(weight, -gain, gain)
    elif init_method == "xavier_uniform":
        torch.nn.init.xavier_uniform_(weight, gain=gain)
    elif init_method == "xavier_normal" or init_method == "xavier":
        torch.nn.init.xavier_normal_(weight, gain=gain)
    elif init_method == "kaiming_uniform":
        torch.nn.init.kaiming_uniform_(weight, a=kaiming_a, mode=kaiming_fan_mode, nonlinearity=w_init_gain)
    elif init_method == "kaiming_normal" or init_method == "kaiming":
        torch.nn.init.kaiming_normal_(weight, a=kaiming_a, mode=kaiming_fan_mode, nonlinearity=w_init_gain)
    elif init_method == "orthogonal":
        torch.nn.init.orthogonal_(weight, gain=gain)
    else:
        raise NotImplementedError("initialization method [%s] is not implemented" % init_method)


def detach(x: T.Union[torch.Tensor, T.Dict[str, T.Any], T.Sequence[torch.Tensor]]):
    """
    Detach each element in x, regardless if it is a tensor or nested list of tensors.
    """
    if isinstance(x, torch.Tensor):
        return x.detach()
    elif isinstance(x, dict):
        for key, val in x.items():
            x[key] = detach(val)
        return x
    elif isinstance(x, T.Sequence):
        return [detach(xi) for xi in x]
    else:
        raise NotImplementedError


def randn_like(x: T.Union[torch.Tensor, T.Sequence[torch.Tensor]]):
    """
    Create a new tensor or nested list of tensors that has the same shape as x.
    Each of the tensor is filled with iid samples from a standard normal distribution.
    """
    if isinstance(x, torch.Tensor):
        return torch.randn_like(x)
    elif isinstance(x, T.Sequence):
        return [randn_like(xi) for xi in x]
    else:
        raise NotImplementedError


def get_valid_mask(
    valid_lens: T.Union[T.Sequence[int], torch.LongTensor, np.ndarray],
    max_len: int = None,
    device=torch.device("cpu"),
    invalid=False,
) -> torch.BoolTensor:
    """
    Returns a BoolTensor B, where B[i,j] = True if j < valid_lens[i].

    Args:
        valid_lens: (batch,)
        max_len: int, max length of the mask. If None, use max(valid_len)
        device: the device of the output mask.
        invalid: invert the result

    Returns:
         boolTensor (batch, max_len)
    """
    if max_len is None:
        max_len = torch.max(valid_lens)

    if isinstance(valid_lens, torch.Tensor):
        pass
    elif isinstance(valid_lens, (list, tuple)):
        valid_lens = torch.tensor(valid_lens, dtype=torch.long, device=device)
    elif isinstance(valid_lens, np.ndarray):
        valid_lens = torch.from_numpy(valid_lens).to(dtype=torch.long, device=device)
    else:
        raise NotImplementedError
    idxs = torch.arange(0, max_len, device=valid_lens.device)  # (max_len,)
    if not invalid:
        mask = idxs < valid_lens.unsqueeze(1)
    else:
        mask = idxs >= valid_lens.unsqueeze(1)
    return mask.to(device=device)


def construct_onehot_vectors(labels: torch.Tensor, total_classes: int) -> torch.Tensor:
    """
    Given labels
    :param labels: (seq_len, batch_size,), (batch_size,), or (*) int, [0, total_classes-1]
    :param total_classes: total number of classes
    :return: onehot embedding in float (*, total_classes) and same device as labels
    """
    ori_label_shape = labels.shape
    onehots = torch.zeros(labels.numel(), total_classes, device=labels.device)
    onehots.scatter_(1, labels.view(-1, 1), 1)
    return onehots.view(*ori_label_shape, total_classes)


def pad_till_sequence_length(
    x: torch.Tensor,
    min_seq_len: int,
    pad_val: float = 0.0,
    batch_first: bool = False,
):
    """
    Pad x in the sequence dimension so that x has
    sequence length at least min_seq_len

    Args:
        x:
            (seq_len, b, dim) if not batch_first
            (b, dim, seq_len) otherwise
        min_seq_len:
            min seq_len of the padded x
        pad_val:
            value to pad x with
        batch_first:
            whether x is (seq_len, b, dim) if not batch_first, or
            (b, dim, seq_len) otherwise

    Returns:
        (min_seq_len, b, dim) or (b, dim, min_seq_len) if `batch_first` is True

    """
    # pad x so that x is long enough to be downsampled
    if batch_first:
        b, c, seq_len = x.shape
        if seq_len < min_seq_len:
            tmp = x
            x = torch.ones(b, c, min_seq_len, dtype=x.dtype, device=x.device) * pad_val
            x[..., :seq_len] = tmp
    else:
        seq_len, b, c = x.shape
        if seq_len < min_seq_len:
            tmp = x
            x = torch.ones(min_seq_len, b, c, dtype=x.dtype, device=x.device) * pad_val
            x[:seq_len] = tmp
    return x
