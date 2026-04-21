
from __future__ import annotations

from typing import Tuple
import numpy as np

from .objectives import objective_ctb_only, objective_ctro_only, objective_main, objective_mdp
from .gmv import solve_gmv_long_only_slsqp
from ..solvers.core import SolverResult, projected_gradient_then_newton
from ..solvers.simplex import project_to_simplex


DEFAULT_SOLVER = {
    'pg_max_iter': 300,
    'pg_step': 0.1,
    'pg_tol': 1e-7,
    'newton_max_iter': 20,
}


def equal_weight(n: int) -> Tuple[np.ndarray, SolverResult]:
    x = np.ones(n) / n
    res = SolverResult(x, True, 0, 0, 0.0, False, n, 'success', 0.0)
    return x, res


def solve_gmv(sigma: np.ndarray, solver_cfg: dict | None = None) -> Tuple[np.ndarray, SolverResult]:
    cfg = solver_cfg or {}
    return solve_gmv_long_only_slsqp(
        sigma,
        x0=np.ones(sigma.shape[0]) / sigma.shape[0],
        max_iter=int(cfg.get('gmv_max_iter', 500)),
        ftol=float(cfg.get('gmv_ftol', 1e-12)),
    )


def solve_mdp(sigma: np.ndarray, gamma: float = 0.0, solver_cfg: dict | None = None) -> Tuple[np.ndarray, SolverResult]:
    n = sigma.shape[0]
    cfg = {**DEFAULT_SOLVER, **(solver_cfg or {})}
    x0 = np.ones(n) / n
    func = lambda x: objective_mdp(x, sigma, gamma=gamma).obj_total
    res = projected_gradient_then_newton(func, x0, **{k: cfg[k] for k in ['pg_max_iter','pg_step','pg_tol','newton_max_iter']})
    return res.weights, res


def solve_ctr_only(sigma: np.ndarray, b: np.ndarray, x_prev: np.ndarray | None, eta: float, gamma: float, solver_cfg: dict | None = None) -> Tuple[np.ndarray, SolverResult]:
    n = sigma.shape[0]
    cfg = {**DEFAULT_SOLVER, **(solver_cfg or {})}
    x0 = project_to_simplex(x_prev if x_prev is not None else np.ones(n) / n)
    func = lambda x: objective_ctro_only(x, sigma, b, x_prev, eta, gamma).obj_total
    res = projected_gradient_then_newton(func, x0, **{k: cfg[k] for k in ['pg_max_iter','pg_step','pg_tol','newton_max_iter']})
    return res.weights, res


def solve_ctb_only(sigma: np.ndarray, x_prev: np.ndarray | None, eta: float, gamma: float, solver_cfg: dict | None = None) -> Tuple[np.ndarray, SolverResult]:
    n = sigma.shape[0]
    cfg = {**DEFAULT_SOLVER, **(solver_cfg or {})}
    x0 = project_to_simplex(x_prev if x_prev is not None else np.ones(n) / n)
    func = lambda x: objective_ctb_only(x, sigma, x_prev, eta, gamma).obj_total
    res = projected_gradient_then_newton(func, x0, **{k: cfg[k] for k in ['pg_max_iter','pg_step','pg_tol','newton_max_iter']})
    return res.weights, res


def solve_main(sigma: np.ndarray, b: np.ndarray, x_prev: np.ndarray | None, eta: float, gamma: float, delta: float, rho: float, solver_cfg: dict | None = None) -> Tuple[np.ndarray, SolverResult]:
    n = sigma.shape[0]
    cfg = {**DEFAULT_SOLVER, **(solver_cfg or {})}
    x0 = project_to_simplex(x_prev if x_prev is not None else np.ones(n) / n)
    func = lambda x: objective_main(x, sigma, b, x_prev, eta, gamma, delta, rho).obj_total
    res = projected_gradient_then_newton(func, x0, **{k: cfg[k] for k in ['pg_max_iter','pg_step','pg_tol','newton_max_iter']})
    return res.weights, res
