#
# Copyright (C) 2026 Apple Inc. All rights reserved.
#
# MLX implementations of ODE solvers for flow matching sampling.
# Inference only — mirrors src/lito/odelibs/ode_solvers.py

import typing as T

import mlx.core as mx
from tqdm import tqdm


def odeint(
    func: T.Callable,
    x0: mx.array,
    ts: mx.array,
    method: str,
    printout: bool = False,
    **kwargs,
) -> mx.array:
    """Numerically integrate the ODE from ts[0] to ts[-1].

    Solves dx/dt = func(t, x), with initial condition x(ts[0]) = x0.

    Accumulation is always done in float32 for numerical stability,
    matching the PyTorch ODE solver behavior. The returned result
    preserves the float32 accumulation dtype.

    Args:
        func: Callable implementing f(t, x), where t has shape (b,) and x
            has shape (b, *d).
        x0: Initial state. (b, *d)
        ts: Timestep schedule. (num_steps+1,) or (b, num_steps+1)
        method: Integration method, one of "euler" or "heun".
        printout: If True, show a tqdm progress bar.

    Returns:
        The final integrated state x in float32. (b, *d)
    """
    if method == "euler":
        x_final = odeint_euler(func, x0, ts, printout=printout)
    elif method.startswith("heun"):
        x_final = odeint_heun(func, x0, ts, printout=printout)
    else:
        raise ValueError(f"Unsupported method: {method}")
    return x_final


def odeint_euler(
    func: T.Callable,
    x0: mx.array,
    ts: mx.array,
    printout: bool = False,
) -> mx.array:
    """Numerically integrate the ODE with Euler first-order approximation.

    Solves dx/dt = func(t, x) from ts[0] to ts[-1] using the update rule:
        x_{t+1} = x_t + (ts[t+1] - ts[t]) * func(ts[t], x_t)

    Accumulation is done in float32 for numerical stability. The model
    (func) may run in any precision; outputs are cast to float32 before
    the step update.

    Args:
        func: Callable implementing f(t, x), where t has shape (b,) and x
            has shape (b, *d).
        x0: Initial state. (b, *d)
        ts: Timestep schedule. (num_steps+1,) or (b, num_steps+1)
        printout: If True, show a tqdm progress bar.

    Returns:
        The final integrated state x in float32. (b, *d)
    """
    b = x0.shape[0]
    d_shape = x0.shape[1:]

    if ts.ndim == 1:
        ts = mx.broadcast_to(ts[None], (b, ts.shape[0]))  # (b, num_steps+1)

    num_steps = ts.shape[-1] - 1

    x = x0.astype(mx.float32)  # (b, *d) accumulate in float32
    for i in tqdm(range(num_steps), disable=not printout):
        dxdt = func(ts[:, i], x).astype(mx.float32)  # (b, *d)
        hi = (ts[:, i + 1] - ts[:, i]).astype(mx.float32).reshape(-1, *([1] * len(d_shape)))  # (b, 1, 1, ...)
        x = x + hi * dxdt
        mx.eval(x)  # Materialize to prevent lazy computation graph explosion

    return x


def odeint_heun(
    func: T.Callable,
    x0: mx.array,
    ts: mx.array,
    printout: bool = False,
) -> mx.array:
    """Numerically integrate the ODE with Heun's second-order approximation.

    Uses a predictor-corrector scheme (see https://arxiv.org/pdf/2206.00364, Alg 3):
        x_pred = x_t + h * f(t, x_t)
        x_{t+1} = x_t + 0.5 * h * (f(t, x_t) + f(t+h, x_pred))

    Accumulation is done in float32 for numerical stability. The model
    (func) may run in any precision; outputs are cast to float32 before
    the step update.

    Args:
        func: Callable implementing f(t, x), where t has shape (b,) and x
            has shape (b, *d).
        x0: Initial state. (b, *d)
        ts: Timestep schedule. (num_steps+1,) or (b, num_steps+1)
        printout: If True, show a tqdm progress bar.

    Returns:
        The final integrated state x in float32. (b, *d)
    """
    b = x0.shape[0]
    d_shape = x0.shape[1:]

    if ts.ndim == 1:
        ts = mx.broadcast_to(ts[None], (b, ts.shape[0]))  # (b, num_steps+1)

    num_steps = ts.shape[-1] - 1

    x = x0.astype(mx.float32)  # (b, *d) accumulate in float32
    for i in tqdm(range(num_steps), disable=not printout):
        dxdt = func(ts[:, i], x).astype(mx.float32)  # (b, *d)
        hi = (ts[:, i + 1] - ts[:, i]).astype(mx.float32).reshape(-1, *([1] * len(d_shape)))  # (b, 1, 1, ...)
        x_new = x + hi * dxdt  # (b, *d)
        t_new = ts[:, i + 1]  # (b,)
        dxdt_new = func(t_new, x_new).astype(mx.float32)  # (b, *d)
        x = x + (hi * 0.5) * (dxdt + dxdt_new)
        mx.eval(x)  # Materialize to prevent lazy computation graph explosion

    return x
