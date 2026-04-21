from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .metrics import d_b, d_r, diversification_ratio, portfolio_variance


@dataclass
class ObjectiveBreakdown:
    obj_total: float
    dr_term: float
    db_term: float
    smooth_term: float
    l2_term: float
    band_penalty: float


def objective_ctro_only(x: np.ndarray, sigma: np.ndarray, b: np.ndarray, x_prev: np.ndarray | None, eta: float, gamma: float) -> ObjectiveBreakdown:
    dr_term = d_r(x, sigma, b)
    smooth_term = float(eta * np.sum((x - x_prev)**2)) if x_prev is not None else 0.0
    l2_term = float(gamma * np.sum(x**2))
    total = dr_term + smooth_term + l2_term
    return ObjectiveBreakdown(total, dr_term, 0.0, smooth_term, l2_term, 0.0)


def objective_ctb_only(x: np.ndarray, sigma: np.ndarray, x_prev: np.ndarray | None, eta: float, gamma: float) -> ObjectiveBreakdown:
    db_term = d_b(x, sigma)
    smooth_term = float(eta * np.sum((x - x_prev)**2)) if x_prev is not None else 0.0
    l2_term = float(gamma * np.sum(x**2))
    total = db_term + smooth_term + l2_term
    return ObjectiveBreakdown(total, 0.0, db_term, smooth_term, l2_term, 0.0)


def objective_main(x: np.ndarray, sigma: np.ndarray, b: np.ndarray, x_prev: np.ndarray | None, eta: float, gamma: float, delta: float, rho: float) -> ObjectiveBreakdown:
    dr_term = d_r(x, sigma, b)
    db_term = d_b(x, sigma)
    smooth_term = float(eta * np.sum((x - x_prev)**2)) if x_prev is not None else 0.0
    l2_term = float(gamma * np.sum(x**2))
    violation = max(db_term - delta, 0.0)
    band_penalty = float(0.5 * rho * violation**2)
    total = dr_term + smooth_term + l2_term + band_penalty
    return ObjectiveBreakdown(total, dr_term, db_term, smooth_term, l2_term, band_penalty)


def objective_gmv(x: np.ndarray, sigma: np.ndarray) -> ObjectiveBreakdown:
    total = portfolio_variance(x, sigma)
    return ObjectiveBreakdown(total, 0.0, 0.0, 0.0, 0.0, 0.0)


def objective_mdp(x: np.ndarray, sigma: np.ndarray, gamma: float = 0.0) -> ObjectiveBreakdown:
    total = -diversification_ratio(x, sigma) + gamma * float(np.sum(x**2))
    return ObjectiveBreakdown(total, 0.0, 0.0, 0.0, gamma * float(np.sum(x**2)), 0.0)
