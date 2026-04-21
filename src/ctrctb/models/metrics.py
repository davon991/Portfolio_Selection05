from __future__ import annotations

import numpy as np


def portfolio_variance(x: np.ndarray, sigma: np.ndarray) -> float:
    return float(x @ sigma @ x)


def portfolio_volatility(x: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.sqrt(max(portfolio_variance(x, sigma), 1e-16)))


def ctr_values(x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    vol = portfolio_volatility(x, sigma)
    return x * (sigma @ x) / vol


def ctr_shares(x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    var = max(portfolio_variance(x, sigma), 1e-16)
    return x * (sigma @ x) / var


def ctb_values(x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sig_i = np.sqrt(np.clip(np.diag(sigma), 1e-16, None))
    sig_p = portfolio_volatility(x, sigma)
    return (sigma @ x) / (sig_i * sig_p)


def d_r(x: np.ndarray, sigma: np.ndarray, b: np.ndarray) -> float:
    shares = ctr_shares(x, sigma)
    gap = shares - b
    return float(0.5 * np.sum(gap**2))


def d_b(x: np.ndarray, sigma: np.ndarray) -> float:
    vals = ctb_values(x, sigma)
    gap = vals - vals.mean()
    return float(0.5 * np.sum(gap**2))


def diversification_ratio(x: np.ndarray, sigma: np.ndarray) -> float:
    sig_i = np.sqrt(np.clip(np.diag(sigma), 1e-16, None))
    num = float(x @ sig_i)
    den = portfolio_volatility(x, sigma)
    return num / max(den, 1e-16)
