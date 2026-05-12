#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements various ODE solvers.


import typing as T

from tqdm import tqdm

import torch
from contextlib import nullcontext as _nullcontext


def odeint(
    func: T.Callable,
    x0: torch.Tensor,
    ts: torch.Tensor,
    method: str,
    printout: bool = False,
    keep_freq: int = None,
    **kwargs,
) -> T.Union[torch.Tensor, T.Tuple[torch.Tensor, T.List[torch.Tensor]]]:
    """
    Numerically integrating the ODE from ts[0] to ts[-1]:

    dx/dt = func(t, x),  initial condition x at ts[0] = x0.


    Args:
        func:
            any callable implementing the ordinary differential equation f(t, x),
            where t has shape (b,) and x has shape (b, *d).
        x0:
            (b, *d)
        ts:
            (b, num_steps+1) or (num_steps+1,).
            Note that if using torchdiffeq, it only supports (num_steps+1,).
        method:
            'euler', 'heun', others in torchdiffeq
        keep_freq:
            if not None, we will additinoally save and return intermediate x every keep_freq iterations.
            I.e., we have ts[::keep_freq].
            Saved result will be on cpu
        kwargs:
            only for torchdiffeq

    Returns:
        The final x (b, *d).
    """

    if method == "euler":
        # construct uniform ts
        x_final, xs_intermediate = odeint_euler(
            func=func,
            x0=x0,
            ts=ts,
            keep_freq=keep_freq,
            printout=printout,
        )  # (b, num_points, d)

    elif method.startswith("heun"):
        # construct uniform ts
        x_final, xs_intermediate = odeint_heun(
            func=func,
            x0=x0,
            ts=ts,
            keep_freq=keep_freq,
            printout=printout,
        )  # (b, num_points, d)

    else:
        # use torchdiffeq
        ADAPTIVE_SOLVER = ["dopri5", "dopri8", "adaptive_heun", "bosh3"]
        FIXER_SOLVER = ["euler", "rk4", "midpoint", "stochastic"]

        if method in ADAPTIVE_SOLVER:
            options = dict(
                dtype=torch.float64,
            )
        else:
            options = None

        default_kwargs = dict(
            rtol=1e-3,
            atol=1e-4,
            options=options,
            adjoint_params=(),
        )
        default_kwargs.update(**kwargs)

        xs = odeint(
            func=func,
            y0=x0,
            t=ts,  # (num_steps,)
            method=method,
            **default_kwargs,
        )  # (num_time_steps, b, num_points, d)
        x_final = xs[-1]  # (b, num_points, d)
        xs_intermediate = xs[0::keep_freq]
        xs_intermediate = torch.chunk(xs_intermediate.detach().cpu(), chunks=xs_intermediate.size(0), dim=0)

    if keep_freq is not None:
        return x_final, xs_intermediate
    else:
        return x_final


def odeint_euler(
    func: T.Callable,
    x0: torch.Tensor,
    ts: torch.Tensor,
    keep_freq: int = None,
    printout: bool = False,
) -> T.Tuple[torch.Tensor, T.List[torch.Tensor]]:
    """
    Numerically integrating the ODE with euler first order approximation.

    dx/dt = func(t, x),  initial condition x at ts[0] = x0.

    We approximate x_final = \int_{ts[0]}^{ts[-1]} func(t, x) dt
    with x_{t+1} = x_t + (ts[t+1] - ts[t]) * func(ts[t], x_t).

    In other words, we integrate from ts[0] to ts[-1].

    Args:
        func:
            any callable implementing the ordinary differential equation f(t, x),
            where t has shape (b,) and x has shape (b, *d).
        x0:
            (b, *d)
        ts:
            (b, num_steps+1) or (num_steps+1,)
        keep_freq:
            if not None, we will additionally save and return intermediate x every keep_freq iterations.
            I.e., we have ts[::keep_freq].
            Saved result will be on cpu.

    Returns:
        The final x (b, *d).
    """
    b, *d_shape = x0.shape
    if ts.ndim == 1:
        ts = ts.expand(b, -1)  # (b, num_steps+1)

    num_steps = ts.shape[-1] - 1

    x = x0.float()  # (b, *d)
    xs_intermediate = []
    for i in tqdm(range(num_steps), disable=not printout):
        if keep_freq is not None and i % keep_freq == 0:
            xs_intermediate.append(x.detach().cpu())  # (b, *d)

        dxdt = func(ts[:, i], x)  # (b, *d)
        hi = (ts[:, i + 1] - ts[:, i]).reshape(-1, *([1] * len(d_shape)))  # (b, *d)
        with _nullcontext():
            x = x + hi.float() * dxdt.float()

    return x, xs_intermediate


def odeint_heun(
    func: T.Callable,
    x0: torch.Tensor,
    ts: torch.Tensor,
    keep_freq: int = None,
    printout: bool = False,
) -> T.Tuple[torch.Tensor, T.List[torch.Tensor]]:
    """
    Numerically integrating the ODE with heun's second order approximation.
    See https://arxiv.org/pdf/2206.00364 (Alg 3)

    dx/dt = func(t, x),  initial condition x at ts[0] = x0.

    We approximate x_final = \int_{ts[0]}^{ts[-1]} func(t, x) dt
    by integrating from ts[0] to ts[-1].

    Args:
        func:
            any callable implementing the ordinary differential equation f(t, x),
            where t has shape (b,) and x has shape (b, *d).
        x0:
            (b, *d)
        ts:
            (b, num_steps+1) or (num_steps+1,)
        keep_freq:
            if not None, we will additionally save and return intermediate x every keep_freq iterations.
            I.e., we have ts[::keep_freq].
            Saved result will be on cpu.

    Returns:
        The final x (b, *d).
    """
    b, *d_shape = x0.shape
    if ts.ndim == 1:
        ts = ts.expand(b, -1)  # (b, num_steps+1)

    num_steps = ts.shape[-1] - 1

    x = x0.float()  # (b, *d)
    xs_intermediate = []
    for i in tqdm(range(num_steps), disable=not printout):
        if keep_freq is not None and i % keep_freq == 0:
            xs_intermediate.append(x.detach().cpu())  # (b, *d)

        dxdt = func(ts[:, i], x).float()  # (b, *d)
        hi = (ts[:, i + 1] - ts[:, i]).reshape(-1, *([1] * len(d_shape)))  # (b, *d)
        x_new = x + hi.float() * dxdt.float()
        t_new = ts[:, i + 1]
        dxdt_new = func(t_new, x_new)  # (b, *d)
        with _nullcontext():
            x = x + (hi.float() * 0.5) * (dxdt.float() + dxdt_new.float())

    return x, xs_intermediate
