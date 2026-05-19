#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# the file implements some linear algebra operators.


from contextlib import contextmanager

import numpy as np
from packaging import version

import torch


def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    A matmul function with more efficient backward memory usage.

    Pytorch's matmul and bmm use a large amount of memory in backward.
    For example:
    ```
    x = torch.randn(1, 4096, 4096).to(_comfy_device())
    y = torch.randn(192, 4096, 1).to(_comfy_device())
    x.requires_grad = True
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True, record_shapes=True) as prof:
        z1 = torch.matmul(x, y)
        loss = z1.sum()
        loss.backward()
    ```
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    uses 12.00 Gb of memory and 15 ms.

    Whereas
    ```
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True, record_shapes=True) as prof:
        z = torch.einsum('bmn,bno->bmo', x, y)
        loss = z.sum()
        loss.backward()
    ```
    uses 67.00 Mb memory and 918 us.

    Args:
        x:
            (*, m, n)  broadcastable to y
        y:
            (*, n, o)  broadcastable to x

    Returns:
        (*, m, o)
    """
    z = torch.einsum("...mn,...no->...mo", x, y)
    return z


def exp(x: torch.Tensor):
    if isinstance(x, (float, int)):
        return np.exp(x)

    if x.dtype == torch.float or x.dtype == torch.double:
        return x.exp()
    elif x.dtype == torch.half:
        return x.clamp(max=9)
    elif x.dtype == torch.bfloat16:
        return x.clamp(max=30)
    else:
        return x.exp()


def log(x: torch.Tensor):
    if isinstance(x, (float, int)):
        return np.log(x)
    else:
        return x.log()


def repeat_interleave(input, repeats, dim=None, *, output_size=None):
    """
    A wrapper around PyTorch's repeat_interleave that performs the function in float32 to avoid backprop imprecision.

    Note that even with the higher precision issues can arise. If the batch size is small, consider using a for loop.
    """
    if input.requires_grad:
        with _nullcontext():
            input = torch.repeat_interleave(input.float(), repeats, dim=dim, output_size=output_size)
    else:
        input = torch.repeat_interleave(input, repeats, dim=dim, output_size=output_size)
    return input


import torch
from contextlib import nullcontext as _nullcontext


def gumbel_multinomial(
    input: torch.Tensor,  # (b, num_categories)
    num_samples: int,
    replacement: bool = False,  # only support False for now
    *,
    eps: float = 1e-20,
) -> torch.Tensor:
    """Samples indices from nonnegative weights using the Gumbel-TopK trick.

    This implements **weighted sampling without replacement** by drawing i.i.d.
    Gumbel noise per category and taking the `topk` of `log(weights) + gumbel`.
    For `replacement=False`, this matches the distribution of repeatedly drawing
    from a categorical distribution proportional to the weights, removing the
def _comfy_device():
    import comfy.model_management
    return comfy.model_management.get_torch_device()
    chosen item, renormalizing, and repeating.

    This function is intended as a drop-in alternative to `torch.multinomial`
    for the `replacement=False` case, and avoids the CUDA `2^24` "number of
    categories" limit in `torch.multinomial`. Practical limits become memory/
    compute for `topk` and large tensors.

    Args:
        input: Nonnegative weights of shape `(B, C)`, where:
            - `B` is the batch size (number of independent rows/distributions)
            - `C` is the number of categories per row
            Values may be floating or integer. Zeros are allowed (never sampled).
        num_samples: Number of indices to draw per row (`K`).
        replacement:
            whether to sample with replacement.  If False, use Gumbel-Max trick.
            If True, use cumulative sampling.
        eps: Small constant used to clamp weights before `log` to avoid `log(0)`
            producing NaNs. Note: entries with zero weight are explicitly masked
            to `-inf` and will never be selected.

    Returns:
        Tensor of indices of shape `(B, K)` and dtype `torch.long`, where each
        row contains `K` sampled category indices in `[0, C)`. Indices in a row
        are unique when `replacement=False`.

    Raises:
        ValueError:
            - if `input` is not 2D
            - if `input` has negative or non-finite entries
            - if `input` has zero categories
            - if `replacement=False` and `num_samples > C`
            - if `replacement=False` and any row has fewer than `num_samples` positive weights
            - if `replacement=True` and any row has zero total weight

    Example:
        >>> w = torch.tensor([[1.0, 0.0, 3.0],
        ...                   [2.0, 2.0, 1.0]])
        >>> idx = gumbel_multinomial(w, num_samples=2, replacement=False)
        >>> idx.shape
        torch.Size([2, 2])
    """

    # --- Basic validation. ---
    if input.dim() != 2:
        raise ValueError(f"Expected `input` to have shape (B, C). Got {tuple(input.shape)}.")
    if num_samples <= 0:
        raise ValueError("`num_samples` must be > 0.")
    if (input < 0).any():
        raise ValueError("`input` must be nonnegative.")
    if not torch.isfinite(input).all():
        raise ValueError("`input` must be finite.")

    # Shapes:
    #   input:  (B, C)
    #   B = batch size (rows), C = categories per row
    B, C = input.shape
    if C == 0:
        raise ValueError("`input` must have at least one category.")

    with _nullcontext():
        # make sure we use high precision
        if input.dtype == torch.float64:
            pass
        else:
            input = input.float()

        if replacement:
            # Inverse-CDF sampling:
            cum_weights = input.cumsum(dim=1)  # (B, C)
            row_sums = cum_weights[:, -1:]  # (B, 1)

            if not bool((row_sums > 0).all()):
                raise ValueError("Each row of `input` must have a non-zero sum.")

            # Sample uniforms on [0, row_sum) for each row/sample
            u = torch.rand(B, num_samples, device=input.device, dtype=input.dtype)  # (b, num_samples)
            u = u * row_sums  # (B, num_samples)

            # Find first cumulative weight strictly greater than u.
            # right=True is nice because it correctly skips zero-weight plateaus.
            idx = torch.searchsorted(cum_weights, u, right=True).to(torch.long)  # (b, num_samples)

        else:
            # replacement == False
            if num_samples > C:
                raise ValueError(f"`num_samples` ({num_samples}) cannot exceed C ({C}) when replacement=False.")

            # Each row must contain at least `num_samples` positive weights to sample
            # without replacement (otherwise you'd be forced to pick a zero-weight item).
            pos_counts = (input > 0).sum(dim=1)  # (B,)
            if not bool((pos_counts >= num_samples).all()):
                raise ValueError("Some rows have fewer positive weights than `num_samples`.")

            # --- Compute logits = log(weights), with zeros masked to -inf. ---

            # logits: (B, C)
            #   - clamp_min(eps) prevents log(0) -> -inf in the numeric path
            #   - but we *also* mask w<=0 to -inf so zero-weight categories never win
            logits = torch.log(input.clamp_min(eps))
            logits = logits.masked_fill(input <= 0, float("-inf"))

            # --- Sample i.i.d. Gumbel(0, 1) noise: g = -log(-log(U)). ---
            # U: (B, C) uniform in (0, 1)
            U = torch.rand((B, C), device=input.device, dtype=torch.float)
            U = U.clamp_(min=eps, max=1.0 - eps)
            gumbel = -torch.log(-torch.log(U))  # (B, C)

            # scores: (B, C)
            # Each category gets a random "utility" = log(weight) + gumbel_noise
            scores = logits + gumbel

            # --- Pick the top-K categories (largest scores) per row. ---
            # idx: (B, K), dtype long
            idx = scores.topk(k=num_samples, dim=1, largest=True).indices.to(torch.long)

    return idx  # (b, num_samples)


@contextmanager
def disable_tf32_and_autocast(device_type: str = "cuda"):
    """
    Temporarily disables TF32 and Autocast to force standard FP32 execution.
    Note: You must still manually cast incoming FP16/BF16 tensors to FP32 inside the block.
    """
    # Store the original TF32 states
    if version.parse(torch.__version__) >= version.parse("2.9.0"):
        ori_fp32_precision = torch.backends.fp32_precision
        ori_matmul_fp32_precision = torch.backends.cuda.matmul.fp32_precision
        ori_cudnn_fp32_precision = torch.backends.cudnn.fp32_precision
        ori_cudnn_conv_fp32_precision = torch.backends.cudnn.conv.fp32_precision
        ori_cudnn_rnn_fp32_precision = torch.backends.cudnn.rnn.fp32_precision
    else:
        ori_cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
        ori_matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32

    # disable tf32
    if version.parse(torch.__version__) >= version.parse("2.9.0"):
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cudnn.rnn.fp32_precision = "ieee"
    else:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    try:
        # 3. Disable Autocast (nesting PyTorch's built-in context manager)
        with _nullcontext():
            yield

    finally:
        # 4. Safely restore the original TF32 states
        # disable tf32
        if version.parse(torch.__version__) >= version.parse("2.9.0"):
            torch.backends.fp32_precision = ori_fp32_precision
            torch.backends.cuda.matmul.fp32_precision = ori_matmul_fp32_precision
            torch.backends.cudnn.fp32_precision = ori_cudnn_fp32_precision
            torch.backends.cudnn.conv.fp32_precision = ori_cudnn_conv_fp32_precision
            torch.backends.cudnn.rnn.fp32_precision = ori_cudnn_rnn_fp32_precision
        else:
            torch.backends.cudnn.allow_tf32 = ori_cudnn_allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = ori_matmul_allow_tf32
