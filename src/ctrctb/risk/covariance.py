from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def annualize_cov(cov: np.ndarray, periods_per_year: int = 252) -> np.ndarray:
    return cov * periods_per_year


def ledoit_wolf_cov(returns: pd.DataFrame, periods_per_year: int = 252) -> np.ndarray:
    if returns.shape[0] < 2:
        raise ValueError('Need at least 2 rows of returns to estimate covariance.')
    lw = LedoitWolf().fit(returns.to_numpy())
    return annualize_cov(lw.covariance_, periods_per_year=periods_per_year)


def sample_cov(returns: pd.DataFrame, periods_per_year: int = 252) -> np.ndarray:
    return annualize_cov(returns.cov().to_numpy(), periods_per_year=periods_per_year)


def estimate_covariance(returns: pd.DataFrame, method: str = 'ledoit_wolf_252d') -> np.ndarray:
    if method == 'ledoit_wolf_252d':
        return ledoit_wolf_cov(returns, periods_per_year=252)
    if method == 'sample_cov_252d':
        return sample_cov(returns, periods_per_year=252)
    raise ValueError(f'Unsupported covariance method: {method}')
