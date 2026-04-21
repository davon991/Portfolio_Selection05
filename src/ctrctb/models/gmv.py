
from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy.optimize import minimize

from ..solvers.core import SolverResult, kkt_residual_simplex


def solve_gmv_long_only_slsqp(sigma: np.ndarray, x0: np.ndarray | None = None, max_iter: int = 500, ftol: float = 1e-12) -> Tuple[np.ndarray, SolverResult]:
    """Dedicated long-only GMV solver, decoupled from the generic PG-Newton solver.

    Problem:
        minimize x' Σ x
        s.t. x >= 0, 1'x = 1

    Uses SLSQP because GMV is a smooth convex quadratic problem with simplex constraints.
    """
    sigma = np.asarray(sigma, dtype=float)
    n = sigma.shape[0]
    if x0 is None:
        x0 = np.ones(n) / n
    x0 = np.asarray(x0, dtype=float)
    x0 = np.clip(x0, 0.0, None)
    s = x0.sum()
    x0 = x0 / s if s > 0 else np.ones(n) / n

    def obj(x: np.ndarray) -> float:
        return float(x @ sigma @ x)

    def jac(x: np.ndarray) -> np.ndarray:
        return 2.0 * (sigma @ x)

    constraints = ({'type': 'eq', 'fun': lambda x: float(x.sum() - 1.0), 'jac': lambda x: np.ones_like(x)},)
    bounds = [(0.0, 1.0) for _ in range(n)]

    opt = minimize(
        obj,
        x0,
        method='SLSQP',
        jac=jac,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': int(max_iter), 'ftol': float(ftol), 'disp': False},
    )

    x = np.asarray(opt.x, dtype=float)
    x = np.clip(x, 0.0, None)
    s = x.sum()
    x = x / s if s > 0 else np.ones(n) / n
    resid = kkt_residual_simplex(obj, x)
    ok = bool(opt.success) or resid < 1e-7
    status = 'success' if ok else 'partial'
    return x, SolverResult(
        weights=x,
        converged=ok,
        iterations_pg=0,
        iterations_newton=int(getattr(opt, 'nit', 0) or 0),
        kkt_residual=float(resid),
        fallback_used=False,
        active_free_dim=int((x > 1e-8).sum()),
        status=status,
        objective_value=float(obj(x)),
    )
