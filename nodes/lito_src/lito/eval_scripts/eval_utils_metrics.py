import math
import typing as T

import torch
import torch.nn.functional as F

try:
    import fused_ssim
except ImportError:
    fused_ssim = None
    print("Please install fused-ssim (https://github.com/rahul-goel/fused-ssim)")


def gaussian(window_size, sigma):
    # https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/utils/loss_utils.py#L46
    gauss = torch.Tensor([math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    # https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/utils/loss_utils.py#L50
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, reduction="mean", size_average=True):
    # NOTE: this function assumes the shape to be NCHW
    assert (img1.ndim == 4) and (img1.shape[-3] == 3), f"{img1.shape=}"
    assert (img2.ndim == 4) and (img2.shape[-3] == 3), f"{img2.shape=}"

    # https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/utils/loss_utils.py#L66
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)  # (N, C, H, W)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)  # (N, C, H, W)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq  # (N, C, H, W)
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq  # (N, C, H, W)
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2  # (N, C, H, W)

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )  # (N, C, H, W)

    if reduction is None:
        return ssim_map
    elif reduction == "mean":
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    else:
        raise NotImplementedError(f"{reduction=}")


def ssim(img1, img2, window_size=11, reduction="mean", size_average=True):
    # https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/utils/loss_utils.py#L56
    #
    # NOTE: this function assumes the shape to be NCHW
    assert (img1.ndim == 4) and (img1.shape[-3] <= 4), f"{img1.shape=}"
    assert (img2.ndim == 4) and (img2.shape[-3] <= 4), f"{img2.shape=}"

    img1 = img1[:, :3, ...]  # (N, 3, H, W)
    img2 = img2[:, :3, ...]  # (N, 3, H, W)

    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(
        img1=img1,
        img2=img2,
        window=window,
        window_size=window_size,
        channel=channel,
        size_average=size_average,
        reduction=reduction,
    )


def fast_ssim(
    img1: torch.Tensor,  # (b, c, h, w)  [0, 1]
    img2: torch.Tensor,  # (b, c, h, w)  [0, 1]
    reduction: T.Union[str, None] = "mean",
    padding: str = "same",  # "same", "valid"
    train: bool = True,
):
    """
    Compute ssim using fused kernel.

    Args:
        img1:
            (b, c, h, w) [0, 1]
        img2:
            (b, c, h, w) [0, 1]
        reduction:
            "mean", "none", None
        padding:
            "same", "valid"
        train:
            is no_grad, set it to True will be faster.

    Returns:
        (,) or (b, h', w')
    """
    assert fused_ssim is not None
    assert padding in fused_ssim.allowed_padding, f"{padding=}"

    if reduction is None:
        reduction = "none"

    C1 = 0.01**2
    C2 = 0.03**2
    out = fused_ssim.FusedSSIMMap.apply(
        C1,  # C1=C1,
        C2,  # C2=C2,
        img1.contiguous(),  # img1=img1.contiguous(),
        img2,  # img2=img2,
        padding,  # padding=padding,
        train,  # train=train,
        2,  # spatial_dims=2,
    )  # (b, h', w') h', w' depends on the padding mode

    if reduction == "mean":
        return out.mean()
    elif reduction == "none":
        return out
    else:
        raise NotImplementedError(f"{reduction=}")


def psnr(img1, img2):
    # https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/utils/image_utils.py#L17
    #
    # NOTE: this function assumes
    # - the shape to be NCHW
    # - the value range is [0, 1] as PSNR = 20 \cdot log_10 (MAX / sqrt(MSE))
    assert (img1.ndim == 4) and (img1.shape[-3] <= 4), f"{img1.shape=}"
    assert (img2.ndim == 4) and (img2.shape[-3] <= 4), f"{img2.shape=}"

    img1 = img1[:, :3, ...]
    img2 = img2[:, :3, ...]

    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
