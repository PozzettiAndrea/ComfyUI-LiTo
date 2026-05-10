#
# Copyright (C) 2026 Apple Inc. All rights reserved.
#
# MLX implementation of flow matching paths.
# Inference only — mirrors src/lito/flow/path.py

import typing as T

import mlx.core as mx


class LinearPath:
    """Linear coupling plan for flow matching.

    Defines the interpolation: xt = sigma_t * x0 + alpha_t * x1,
    where x0 is noise, x1 is data, alpha_t = t, and sigma_t = 1 - t.
    """

    def compute_alpha_t(self, t: mx.array) -> T.Tuple[mx.array, mx.array]:
        """Compute alpha_t and its time derivative.

        For the linear path, alpha_t = t and d_alpha_t/dt = 1.

        Args:
            t: Time values. (*,)

        Returns:
            alpha_t: Interpolation coefficient for data. (*,)
            d_alpha_t_dt: Time derivative of alpha_t. (*,)
        """
        return t, mx.ones_like(t)

    def compute_sigma_t(
        self, t: T.Union[mx.array, float, int]
    ) -> T.Tuple[T.Union[mx.array, float, int], T.Union[mx.array, float, int]]:
        """Compute sigma_t and its time derivative.

        For the linear path, sigma_t = 1 - t and d_sigma_t/dt = -1.

        Args:
            t: Time values, either an mx.array (*,) or a Python scalar.

        Returns:
            sigma_t: Interpolation coefficient for noise. (*,)
            d_sigma_t_dt: Time derivative of sigma_t. (*,)
        """
        if isinstance(t, mx.array):
            return 1 - t, -mx.ones_like(t)
        else:
            return 1 - t, -1

    def compute_t(self, sigma_t: T.Union[mx.array, float, int]) -> T.Union[mx.array, float, int]:
        """Compute t that would produce the given sigma_t.

        For the linear path, t = 1 - sigma_t.

        Args:
            sigma_t: Noise coefficient value. (*,)

        Returns:
            t: Corresponding time value. (*,)
        """
        return 1 - sigma_t
