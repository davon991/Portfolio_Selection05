from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable
import numpy as np

from .simplex import project_to_simplex


@dataclass
class SolverResult:
    weights: np.ndarray
    converged: bool
    iterations_pg: int
    iterations_newton: int
    kkt_residual: float
    fallback_used: bool
    active_free_dim: int
    status: str
    objective_value: float

    def to_dict(self) -> dict:
        out = asdict(self)
        out['weights'] = self.weights.tolist()
        return out


def numerical_grad(func: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = eps
        g[i] = (func(x + e) - func(x - e)) / (2 * eps)
    return g


def numerical_hessian(func: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    n = len(x)
    h = np.zeros((n, n), dtype=float)
    fx = func(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = eps
        h[i, i] = (func(x + ei) - 2 * fx + func(x - ei)) / (eps ** 2)
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = eps
            val = (func(x + ei + ej) - func(x + ei - ej) - func(x - ei + ej) + func(x - ei - ej)) / (4 * eps**2)
            h[i, j] = val
            h[j, i] = val
    return h


def kkt_residual_simplex(func: Callable[[np.ndarray], float], x: np.ndarray) -> float:
    g = numerical_grad(func, x)
    g_centered = g - g.mean()
    inactive = x > 1e-8
    if inactive.any():
        return float(np.linalg.norm(g_centered[inactive], ord=2))
    return float(np.linalg.norm(g_centered, ord=2))


def projected_gradient_then_newton(
    func: Callable[[np.ndarray], float],
    x0: np.ndarray,
    pg_max_iter: int = 300,
    pg_step: float = 0.1,
    pg_tol: float = 1e-7,
    newton_max_iter: int = 20,
    interior_eps: float = 1e-6,
) -> SolverResult:
    x = project_to_simplex(np.asarray(x0, dtype=float))
    f_prev = func(x)
    converged = False
    fallback_used = False

    # Stage I: projected gradient
    pg_iter = 0
    for pg_iter in range(1, pg_max_iter + 1):
        g = numerical_grad(func, x)
        step = pg_step
        improved = False
        for _ in range(20):
            x_new = project_to_simplex(x - step * g)
            f_new = func(x_new)
            if f_new <= f_prev + 1e-12:
                improved = True
                break
            step *= 0.5
        if not improved:
            fallback_used = True
            break
        if np.linalg.norm(x_new - x) < pg_tol:
            x = x_new
            f_prev = f_new
            converged = True
            break
        x = x_new
        f_prev = f_new

    # Stage II: damped Newton on interior/free coordinates
    newton_iter = 0
    for newton_iter in range(1, newton_max_iter + 1):
        free = np.where(x > interior_eps)[0]
        if len(free) <= 1:
            break
        # reduced coordinates: last free variable determined by simplex sum
        base = free[:-1]
        last = free[-1]

        def reduced_func(z: np.ndarray) -> float:
            xr = x.copy()
            xr[base] = z
            xr[last] = 1.0 - xr.sum() + xr[last]
            if (xr < 0).any():
                return 1e12
            return func(xr)

        z0 = x[base].copy()
        gz = numerical_grad(reduced_func, z0)
        if np.linalg.norm(gz) < 1e-6:
            converged = True
            break
        hz = numerical_hessian(reduced_func, z0)
        hz = hz + 1e-6 * np.eye(hz.shape[0])
        try:
            dz = -np.linalg.solve(hz, gz)
        except np.linalg.LinAlgError:
            fallback_used = True
            break
        step = 1.0
        improved = False
        f_curr = func(x)
        for _ in range(20):
            z_new = z0 + step * dz
            xr = x.copy()
            xr[base] = z_new
            xr[last] = 1.0 - xr.sum() + xr[last]
            if (xr >= -1e-12).all():
                xr = project_to_simplex(xr)
                f_new = func(xr)
                if f_new <= f_curr + 1e-12:
                    x = xr
                    improved = True
                    break
            step *= 0.5
        if not improved:
            fallback_used = True
            break
        if np.linalg.norm(step * dz) < 1e-7:
            converged = True
            break

    resid = kkt_residual_simplex(func, x)
    return SolverResult(
        weights=x,
        converged=converged or resid < 1e-5,
        iterations_pg=pg_iter,
        iterations_newton=newton_iter,
        kkt_residual=resid,
        fallback_used=fallback_used,
        active_free_dim=int((x > interior_eps).sum()),
        status='success' if (converged or resid < 1e-5) else 'partial',
        objective_value=float(func(x)),
    )
