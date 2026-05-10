#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements the flow match trajectory.
import abc
import math
import typing as T

import torch


def unsqueeze_to(t: torch.Tensor, x: torch.Tensor):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


class BasePath(abc.ABC):
    r"""
    Base class for flow matching path

    We follow the notation of
    SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers

    xt = sigma_t * x0 + alpha_t * x1,

    where x0 is the noise, and x1 is the data
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def compute_alpha_t(self, t: torch.Tensor) -> T.Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute alpha_t and d_alpha_t_dt

        Args:
            t:
                (*,)

        Returns:
            alpha_t:
                (*,)
            d_alpha_t_dt:
                (*,)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_sigma_t(self, t):
        """
        Compute sigma_t and d_sigma_t_dt

        Args:
            t:
                (*,)

        Returns:
            sigma_t:
                (*,)
            d_sigma_t_dt:
                (*,)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_t(self, sigma_t):
        """
        Compute t that would result in the given sigma_t

        Args:
            sigma_t:
                (*,)

        Returns:
            t:
                (*,)

        Notes:
            the function is only needed if wanting to convert sigma_t
            back to t (assuming sigma_t and t are one-to-one, onto)
        """
        raise NotImplementedError

    def compute_xt_ut(self, t, x0, x1) -> T.Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample xt and ut from time-dependent density p_t

        Args:
            t:
                (b,) or (,)
            x0:
                (b, *)
            x1:
                (b, *)
        Returns:
            xt:
                (b, *)
            ut:
                (b, *)
        """
        t = unsqueeze_to(t=t, x=x0)  # (b, *1)
        alpha_t, d_alpht_t_dt = self.compute_alpha_t(t)
        sigma_t, d_sigma_t_dt = self.compute_sigma_t(t)
        xt = alpha_t * x1 + sigma_t * x0
        ut = d_alpht_t_dt * x1 + d_sigma_t_dt * x0
        return xt, ut


class LinearPath(BasePath):
    """Linear Coupling Plan"""

    def __init__(self):
        super().__init__()

    def compute_alpha_t(self, t):
        """
        Compute alpha_t and d_alpha_t_dt

        Args:
            t:
                (*,)

        Returns:
            alpha_t:
                (*,)
            d_alpha_t_dt:
                (*,)
        """
        if isinstance(t, torch.Tensor):
            return t, torch.ones_like(t)
        else:
            return t, 1

    def compute_sigma_t(self, t):
        """
        Compute sigma_t and d_sigma_t_dt

        Args:
            t:
                (*,)

        Returns:
            sigma_t:
                (*,)
            d_sigma_t_dt:
                (*,)
        """
        if isinstance(t, torch.Tensor):
            return 1 - t, -torch.ones_like(t)
        else:
            return 1 - t, -1

    def compute_t(self, sigma_t):
        """
        Compute t that would result in the given sigma_t

        Args:
            sigma_t:
                (*,)

        Returns:
            t:
                (*,)
        """
        return 1 - sigma_t


class SinusoidalPath(BasePath):
    """
    Sinusoidal Coupling Plan or generalized variance preserving
    https://arxiv.org/pdf/2209.15571
    https://arxiv.org/pdf/2401.08740

    xt = sigma_t * x0 + alpha_t * x1,  (x0 is noise, x1 is data)

    alpha_t = sin(0.5 * pi * t)
    sigma_t = cos(0.5 * pi * t)
    """

    def __init__(self):
        super().__init__()

    def compute_alpha_t(self, t):
        """
        Compute alpha_t and d_alpha_t_dt

        Args:
            t:
                (*,)

        Returns:
            alpha_t:
                (*,)
            d_alpha_t_dt:
                (*,)
        """

        if isinstance(t, torch.Tensor):
            half_pi = 0.5 * torch.pi
            half_pi_t = half_pi * t  # (*,)
            return torch.sin(half_pi_t), half_pi * torch.cos(half_pi_t)
        else:
            half_pi = 0.5 * math.pi
            half_pi_t = half_pi * t  # (*,)
            return math.sin(half_pi_t), half_pi * math.cos(half_pi_t)

    def compute_sigma_t(self, t):
        """
        Compute sigma_t and d_sigma_t_dt

        Args:
            t:
                (*,)

        Returns:
            sigma_t:
                (*,)
            d_sigma_t_dt:
                (*,)
        """
        if isinstance(t, torch.Tensor):
            half_pi = 0.5 * torch.pi
            half_pi_t = half_pi * t  # (*,)
            return torch.cos(half_pi_t), -half_pi * torch.sin(half_pi_t)
        else:
            half_pi = 0.5 * math.pi
            half_pi_t = half_pi * t  # (*,)
            return math.cos(half_pi_t), -half_pi * math.sin(half_pi_t)

    def compute_t(self, sigma_t):
        """
        Compute t that would result in the given sigma_t

        Args:
            sigma_t:
                (*,)

        Returns:
            t:
                (*,)
        """
        half_pi = 0.5 * torch.pi
        t = torch.acos(sigma_t) / half_pi
        return t
