"""Minimal SciPy.optimize replacement for environments without SciPy."""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


class OptimizeResult(dict):
    """Dictionary subclass mimicking :class:`scipy.optimize.OptimizeResult`."""

    def __getattr__(self, name: str):  # pragma: no cover - passthrough
        return self[name]


def _num_jac(fun: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    f0 = fun(x)
    m = f0.size
    n = x.size
    jac = np.zeros((m, n), dtype=float)
    for i in range(n):
        step = eps * max(1.0, abs(x[i]))
        x_step = x.copy()
        x_step[i] += step
        jac[:, i] = (fun(x_step) - f0) / step
    return jac


def least_squares(
    fun: Callable[[np.ndarray], np.ndarray],
    x0: Sequence[float],
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    xtol: float = 1e-8,
    ftol: float = 1e-8,
    gtol: float = 1e-8,
    max_nfev: int = 100,
):
    lower, upper = None, None
    if bounds is not None:
        lower = np.asarray(bounds[0], dtype=float)
        upper = np.asarray(bounds[1], dtype=float)

    x = np.asarray(x0, dtype=float)

    def project(vec: np.ndarray) -> np.ndarray:
        if lower is None:
            return vec
        return np.minimum(np.maximum(vec, lower), upper)

    x = project(x)
    for _ in range(max_nfev):
        f_val = fun(x)
        norm_f = np.linalg.norm(f_val)
        if norm_f < ftol:
            break
        jac = _num_jac(fun, x)
        try:
            step, *_ = np.linalg.lstsq(jac, -f_val, rcond=None)
        except np.linalg.LinAlgError:
            break
        damping = 1.0
        while damping > 1e-4:
            x_trial = project(x + damping * step)
            f_trial = fun(x_trial)
            if np.linalg.norm(f_trial) < norm_f:
                x = x_trial
                break
            damping *= 0.5
        else:
            # Could not find an improvement; stop
            break

        if np.linalg.norm(step) < xtol:
            break

    f_final = fun(x)
    success = np.linalg.norm(f_final) < ftol
    return OptimizeResult(
        success=success,
        x=x,
        fun=f_final,
        cost=0.5 * np.dot(f_final, f_final),
        message="converged" if success else "maximum iterations reached",
    )


class Bounds:
    def __init__(self, lb: Iterable[float], ub: Iterable[float]):
        self.lb = np.asarray(tuple(lb), dtype=float)
        self.ub = np.asarray(tuple(ub), dtype=float)


def minimize(
    fun: Callable[[np.ndarray], float],
    x0: Sequence[float],
    method: str | None = None,
    bounds: Bounds | None = None,
    constraints: Sequence[Mapping[str, Callable[[np.ndarray], float]]] = (),
    options: Mapping[str, float] | None = None,
):
    max_iter = int(options.get("maxiter", 100)) if options else 100
    tol = float(options.get("ftol", 1e-6)) if options else 1e-6
    x = np.asarray(x0, dtype=float)

    if bounds is not None:
        def project(vec: np.ndarray) -> np.ndarray:
            return np.minimum(np.maximum(vec, bounds.lb), bounds.ub)
    else:
        def project(vec: np.ndarray) -> np.ndarray:  # type: ignore[misc]
            return vec

    def penalty(vec: np.ndarray) -> float:
        penalty_val = 0.0
        for constraint in constraints:
            func = constraint["fun"]
            value = func(vec)
            if constraint.get("type", "eq") == "eq":
                penalty_val += value * value
            else:
                penalty_val += max(0.0, -value) ** 2
        return penalty_val

    step_size = 0.1
    for _ in range(max_iter):
        f_val = fun(x) + penalty(x)
        grad = np.zeros_like(x)
        for i in range(x.size):
            step = 1e-6 * max(1.0, abs(x[i]))
            x_step = x.copy()
            x_step[i] += step
            grad[i] = (fun(x_step) + penalty(x_step) - f_val) / step
        x_new = project(x - step_size * grad)
        f_new = fun(x_new) + penalty(x_new)
        if abs(f_new - f_val) < tol:
            x = x_new
            break
        if f_new > f_val:
            step_size *= 0.5
        else:
            x = x_new

    final_val = fun(x)
    return OptimizeResult(success=True, x=x, fun=final_val, nit=max_iter, message="completed")
