from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd

from ..data.prepare import compute_daily_returns, month_end_dates, price_panel_from_long, save_processed_data
from ..data.yahoo import load_or_download_adjusted_close
from ..exports.results import export_run_outputs
from ..models.metrics import ctb_values, ctr_values, d_b, d_r
from ..models.objectives import objective_ctb_only, objective_ctro_only, objective_main
from ..models.strategies import equal_weight, solve_ctb_only, solve_ctr_only, solve_gmv, solve_main, solve_mdp
from ..risk.covariance import estimate_covariance
from ..utils.io import ensure_dir, timestamp_utc, write_json
from ..utils.registry import append_registry_row


@dataclass
class RunArtifacts:
    run_id: str
    result_dir: Path


def _strategy_order() -> list[str]:
    return ['EW', 'GMV', 'CtR-only', 'MDP', 'CtB-only', 'Main']


def _calc_turnover(x_new: np.ndarray, x_prev: np.ndarray | None) -> float:
    if x_prev is None:
        return float(np.abs(x_new).sum())
    return float(np.abs(x_new - x_prev).sum())


def _monthly_held_return(daily_returns: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp, weights: np.ndarray) -> float:
    window = daily_returns[(daily_returns.index > start_date) & (daily_returns.index <= end_date)]
    if window.empty:
        return 0.0
    path = (1.0 + window.to_numpy() @ weights)
    return float(np.prod(path) - 1.0)


def run_real_backtest(config: dict[str, Any], root: str | Path) -> RunArtifacts:
    root = Path(root)
    data_cfg = config['data']
    run_cfg = config['run']
    model_cfg = config['model']
    solver_cfg = config.get('solver', {})

    raw_long = load_or_download_adjusted_close(
        tickers=data_cfg['assets'],
        start=data_cfg['raw_start'],
        end=data_cfg['raw_end'],
        raw_dir=root / 'data' / 'raw',
        force_download=data_cfg.get('force_download', False),
    )
    raw_long['date'] = pd.to_datetime(raw_long['date'])
    prices = price_panel_from_long(raw_long)
    returns = compute_daily_returns(prices)
    save_processed_data(prices, returns, root / 'data' / 'processed')

    eval_start = pd.Timestamp(data_cfg['evaluation_start'])
    eval_end = pd.Timestamp(data_cfg['evaluation_end'])
    month_ends = month_end_dates(returns.index)
    month_ends = month_ends[(month_ends >= eval_start) & (month_ends <= eval_end)]
    if len(month_ends) < 3:
        raise RuntimeError('Not enough monthly rebalance dates to run backtest.')

    strategies = _strategy_order()
    n = len(data_cfg['assets'])
    b = np.ones(n) / n
    prev_weights = {s: None for s in strategies}

    weights_rows, ctr_rows, ctb_rows, drdb_rows, obj_rows, solver_rows, monthly_rows, turnover_rows = ([] for _ in range(8))

    window = int(data_cfg.get('cov_window_days', 252))
    delta = float(model_cfg['delta'])
    eta = float(model_cfg['eta'])
    gamma = float(model_cfg['gamma'])
    rho = float(model_cfg['rho'])

    for idx in range(len(month_ends) - 1):
        reb_date = month_ends[idx]
        next_date = month_ends[idx + 1]
        hist = returns[returns.index <= reb_date].tail(window)
        if len(hist) < window:
            continue
        sigma = estimate_covariance(hist, method=data_cfg['cov_estimator'])

        solved = {}
        diagnostics = {}

        x, res = equal_weight(n)
        solved['EW'] = x
        diagnostics['EW'] = res

        x, res = solve_gmv(sigma, solver_cfg)
        solved['GMV'] = x
        diagnostics['GMV'] = res

        x, res = solve_ctr_only(sigma, b, prev_weights['CtR-only'], eta, gamma, solver_cfg)
        solved['CtR-only'] = x
        diagnostics['CtR-only'] = res

        x, res = solve_mdp(sigma, gamma=gamma, solver_cfg=solver_cfg)
        solved['MDP'] = x
        diagnostics['MDP'] = res

        x, res = solve_ctb_only(sigma, prev_weights['CtB-only'], eta, gamma, solver_cfg)
        solved['CtB-only'] = x
        diagnostics['CtB-only'] = res

        x, res = solve_main(sigma, b, prev_weights['Main'], eta, gamma, delta, rho, solver_cfg)
        solved['Main'] = x
        diagnostics['Main'] = res

        for strategy in strategies:
            x = solved[strategy]
            ctr_vals = ctr_values(x, sigma)
            ctb_vals = ctb_values(x, sigma)
            dr = d_r(x, sigma, b)
            db = d_b(x, sigma)
            turnover = _calc_turnover(x, prev_weights[strategy])
            monthly_ret = _monthly_held_return(returns, reb_date, next_date, x)
            prev = prev_weights[strategy]
            if strategy == 'CtR-only':
                obj = objective_ctro_only(x, sigma, b, prev, eta, gamma)
            elif strategy == 'CtB-only':
                obj = objective_ctb_only(x, sigma, prev, eta, gamma)
            elif strategy == 'Main':
                obj = objective_main(x, sigma, b, prev, eta, gamma, delta, rho)
            else:
                obj = None

            for asset, weight, ctr_val, ctb_val in zip(data_cfg['assets'], x, ctr_vals, ctb_vals):
                weights_rows.append({'date': reb_date.date().isoformat(), 'strategy': strategy, 'asset': asset, 'weight': float(weight)})
                ctr_rows.append({'date': reb_date.date().isoformat(), 'strategy': strategy, 'asset': asset, 'ctr': float(ctr_val)})
                ctb_rows.append({'date': reb_date.date().isoformat(), 'strategy': strategy, 'asset': asset, 'ctb': float(ctb_val)})

            drdb_rows.append({
                'date': reb_date.date().isoformat(), 'strategy': strategy,
                'D_R': float(dr), 'D_B': float(db),
                'band_active': int(db > delta if strategy == 'Main' else 0),
                'band_violation': float(max(db - delta, 0.0) if strategy == 'Main' else 0.0),
            })
            turnover_rows.append({'date': reb_date.date().isoformat(), 'strategy': strategy, 'turnover': turnover})
            monthly_rows.append({'date': next_date.date().isoformat(), 'strategy': strategy, 'period_return': monthly_ret})
            if obj is not None:
                obj_rows.append({
                    'date': reb_date.date().isoformat(), 'strategy': strategy,
                    'obj_total': obj.obj_total, 'dr_term': obj.dr_term, 'db_term': obj.db_term,
                    'smooth_term': obj.smooth_term, 'l2_term': obj.l2_term, 'band_penalty': obj.band_penalty,
                })
            res = diagnostics[strategy]
            solver_rows.append({
                'date': reb_date.date().isoformat(), 'strategy': strategy,
                'converged': int(res.converged), 'iterations_pg': res.iterations_pg,
                'iterations_newton': res.iterations_newton, 'kkt_residual': res.kkt_residual,
                'fallback_used': int(res.fallback_used), 'active_free_dim': res.active_free_dim,
                'status': res.status, 'objective_value': res.objective_value,
            })
            prev_weights[strategy] = x.copy()

    outputs = export_run_outputs(
        config=config,
        root=root,
        weights_df=pd.DataFrame(weights_rows),
        ctr_df=pd.DataFrame(ctr_rows),
        ctb_df=pd.DataFrame(ctb_rows),
        drdb_df=pd.DataFrame(drdb_rows),
        objective_df=pd.DataFrame(obj_rows),
        monthly_returns_df=pd.DataFrame(monthly_rows),
        turnover_df=pd.DataFrame(turnover_rows),
        solver_df=pd.DataFrame(solver_rows),
    )
    return outputs


def run_toy_experiment(config: dict[str, Any], root: str | Path) -> RunArtifacts:
    root = Path(root)
    run_cfg = config['run']
    toy_cfg = config['toy']
    solver_cfg = config.get('solver', {})
    model_cfg = config['model']

    assets = toy_cfg['assets']
    n = len(assets)
    b = np.array(toy_cfg.get('budget', [1.0 / n] * n), dtype=float)
    sigma = np.array(toy_cfg['covariance'], dtype=float)
    x_prev = np.array(toy_cfg['x_prev'], dtype=float) if toy_cfg.get('x_prev') is not None else None
    delta = float(model_cfg['delta'])
    eta = float(model_cfg['eta'])
    gamma = float(model_cfg['gamma'])
    rho = float(model_cfg['rho'])

    strategies = _strategy_order()
    solved = {}
    diagnostics = {}
    x, res = equal_weight(n); solved['EW'] = x; diagnostics['EW'] = res
    x, res = solve_gmv(sigma, solver_cfg); solved['GMV'] = x; diagnostics['GMV'] = res
    x, res = solve_ctr_only(sigma, b, x_prev, eta, gamma, solver_cfg); solved['CtR-only'] = x; diagnostics['CtR-only'] = res
    x, res = solve_mdp(sigma, gamma=gamma, solver_cfg=solver_cfg); solved['MDP'] = x; diagnostics['MDP'] = res
    x, res = solve_ctb_only(sigma, x_prev, eta, gamma, solver_cfg); solved['CtB-only'] = x; diagnostics['CtB-only'] = res
    x, res = solve_main(sigma, b, x_prev, eta, gamma, delta, rho, solver_cfg); solved['Main'] = x; diagnostics['Main'] = res

    fake_date = pd.Timestamp(run_cfg['run_date'])
    weights_rows, ctr_rows, ctb_rows, drdb_rows, obj_rows, solver_rows, monthly_rows, turnover_rows = ([] for _ in range(8))
    for strategy in strategies:
        x = solved[strategy]
        ctr_vals = ctr_values(x, sigma)
        ctb_vals = ctb_values(x, sigma)
        dr = d_r(x, sigma, b)
        db = d_b(x, sigma)
        turnover = _calc_turnover(x, x_prev)
        monthly_rows.append({'date': fake_date.date().isoformat(), 'strategy': strategy, 'period_return': 0.0})
        for asset, weight, ctr_val, ctb_val in zip(assets, x, ctr_vals, ctb_vals):
            weights_rows.append({'date': fake_date.date().isoformat(), 'strategy': strategy, 'asset': asset, 'weight': float(weight)})
            ctr_rows.append({'date': fake_date.date().isoformat(), 'strategy': strategy, 'asset': asset, 'ctr': float(ctr_val)})
            ctb_rows.append({'date': fake_date.date().isoformat(), 'strategy': strategy, 'asset': asset, 'ctb': float(ctb_val)})
        drdb_rows.append({
            'date': fake_date.date().isoformat(), 'strategy': strategy,
            'D_R': float(dr), 'D_B': float(db), 'band_active': int(db > delta if strategy == 'Main' else 0),
            'band_violation': float(max(db - delta, 0.0) if strategy == 'Main' else 0.0),
        })
        turnover_rows.append({'date': fake_date.date().isoformat(), 'strategy': strategy, 'turnover': turnover})
        if strategy == 'CtR-only':
            obj = objective_ctro_only(x, sigma, b, x_prev, eta, gamma)
        elif strategy == 'CtB-only':
            obj = objective_ctb_only(x, sigma, x_prev, eta, gamma)
        elif strategy == 'Main':
            obj = objective_main(x, sigma, b, x_prev, eta, gamma, delta, rho)
        else:
            obj = None
        if obj is not None:
            obj_rows.append({
                'date': fake_date.date().isoformat(), 'strategy': strategy,
                'obj_total': obj.obj_total, 'dr_term': obj.dr_term, 'db_term': obj.db_term,
                'smooth_term': obj.smooth_term, 'l2_term': obj.l2_term, 'band_penalty': obj.band_penalty,
            })
        res = diagnostics[strategy]
        solver_rows.append({
            'date': fake_date.date().isoformat(), 'strategy': strategy,
            'converged': int(res.converged), 'iterations_pg': res.iterations_pg,
            'iterations_newton': res.iterations_newton, 'kkt_residual': res.kkt_residual,
            'fallback_used': int(res.fallback_used), 'active_free_dim': res.active_free_dim,
            'status': res.status, 'objective_value': res.objective_value,
        })

    outputs = export_run_outputs(
        config=config,
        root=root,
        weights_df=pd.DataFrame(weights_rows),
        ctr_df=pd.DataFrame(ctr_rows),
        ctb_df=pd.DataFrame(ctb_rows),
        drdb_df=pd.DataFrame(drdb_rows),
        objective_df=pd.DataFrame(obj_rows),
        monthly_returns_df=pd.DataFrame(monthly_rows),
        turnover_df=pd.DataFrame(turnover_rows),
        solver_df=pd.DataFrame(solver_rows),
    )
    return outputs
