from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd

from ..utils.io import ensure_dir, timestamp_utc, write_json
from ..utils.registry import append_registry_row


@dataclass
class RunArtifacts:
    run_id: str
    result_dir: Path


def _write_latex_table_safe(df: pd.DataFrame, path: Path) -> None:
    """Write a LaTeX table, but do not fail the run if optional styling deps are missing."""
    try:
        df.to_latex(path, index=False, float_format='%.4f')
        return
    except Exception:
        pass

    cols = list(df.columns)
    lines = []
    lines.append('\begin{tabular}{' + 'l' * len(cols) + '}')
    lines.append('\hline')
    lines.append(' & '.join(str(c) for c in cols) + r' \\')
    lines.append('\hline')
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f'{v:.4f}')
            else:
                vals.append(str(v))
        lines.append(' & '.join(vals) + r' \\')
    lines.append('\hline')
    lines.append('\end{tabular}')
    path.write_text('\n'.join(lines), encoding='utf-8')


def _compute_summary(monthly_returns_df: pd.DataFrame, drdb_df: pd.DataFrame, turnover_df: pd.DataFrame, solver_df: pd.DataFrame) -> pd.DataFrame:
    required = {'strategy', 'period_return'}
    if monthly_returns_df.empty or not required.issubset(monthly_returns_df.columns):
        return pd.DataFrame(columns=[
            'strategy', 'ann_return', 'ann_vol', 'sharpe', 'max_drawdown',
            'turnover_mean', 'turnover_p95', 'dr_mean', 'db_mean', 'active_rate', 'failure_rate',
        ])
    rows = []
    for strategy, grp in monthly_returns_df.groupby('strategy'):
        r = grp.sort_values('date')['period_return'].astype(float)
        if len(r) == 0:
            continue
        nav = (1 + r).cumprod()
        peak = nav.cummax()
        dd = nav / peak - 1.0
        ann_return = float(nav.iloc[-1] ** (12 / max(len(r), 1)) - 1) if len(r) > 0 else 0.0
        ann_vol = float(r.std(ddof=1) * np.sqrt(12)) if len(r) > 1 else 0.0
        sharpe = ann_return / ann_vol if ann_vol > 1e-12 else np.nan
        dr_mean = float(drdb_df.loc[drdb_df['strategy'] == strategy, 'D_R'].mean())
        db_mean = float(drdb_df.loc[drdb_df['strategy'] == strategy, 'D_B'].mean())
        active_rate = float(drdb_df.loc[drdb_df['strategy'] == strategy, 'band_active'].mean()) if 'band_active' in drdb_df.columns else 0.0
        turn = turnover_df.loc[turnover_df['strategy'] == strategy, 'turnover'].astype(float)
        fail = 1.0 - float(solver_df.loc[solver_df['strategy'] == strategy, 'converged'].mean()) if 'converged' in solver_df.columns else 0.0
        rows.append({
            'strategy': strategy,
            'ann_return': ann_return,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'max_drawdown': float(dd.min()) if len(dd) else 0.0,
            'turnover_mean': float(turn.mean()) if len(turn) else 0.0,
            'turnover_p95': float(turn.quantile(0.95)) if len(turn) else 0.0,
            'dr_mean': dr_mean,
            'db_mean': db_mean,
            'active_rate': active_rate,
            'failure_rate': fail,
        })
    return pd.DataFrame(rows).sort_values('strategy')


def _analysis_pack(summary_df: pd.DataFrame) -> dict[str, Any]:
    if summary_df.empty:
        return {'best_strategy': None, 'key_findings': [], 'metric_deltas_vs_erc': {}, 'recommended_figures': [], 'warnings': ['No summary rows generated.']}
    ranked = summary_df.sort_values('sharpe', ascending=False)
    best = ranked.iloc[0]['strategy']
    main = summary_df[summary_df['strategy'] == 'Main']
    ctr = summary_df[summary_df['strategy'] == 'CtR-only']
    deltas = {}
    if not main.empty and not ctr.empty:
        deltas = {
            'dr_mean_delta_vs_ctr': float(main.iloc[0]['dr_mean'] - ctr.iloc[0]['dr_mean']),
            'db_mean_delta_vs_ctr': float(main.iloc[0]['db_mean'] - ctr.iloc[0]['db_mean']),
            'turnover_mean_delta_vs_ctr': float(main.iloc[0]['turnover_mean'] - ctr.iloc[0]['turnover_mean']),
        }
    return {
        'best_strategy': best,
        'key_findings': [
            'Interpret sharpe only after checking D_R / D_B / convergence / turnover.',
            'Main should first be judged by mechanism and constraint behavior, not only by return.'
        ],
        'metric_deltas_vs_erc': deltas,
        'recommended_figures': [
            'fig_weights_latest', 'fig_dr_db_timeseries', 'fig_summary_metrics'
        ],
        'warnings': []
    }


def _write_basic_figures(summary_df: pd.DataFrame, weights_df: pd.DataFrame, drdb_df: pd.DataFrame, fig_dir: Path, figdata_dir: Path) -> None:
    import matplotlib.pyplot as plt

    # summary metrics
    summary_figdata = figdata_dir / 'figdata_summary_metrics.csv'
    summary_df.to_csv(summary_figdata, index=False)
    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(summary_df['strategy'], summary_df['sharpe'])
        ax.set_title('Sharpe by strategy')
        ax.set_ylabel('Sharpe')
        ax.tick_params(axis='x', rotation=30)
        fig.tight_layout()
        fig.savefig(fig_dir / 'fig_summary_metrics.png', dpi=150)
        fig.savefig(fig_dir / 'fig_summary_metrics.pdf')
        plt.close(fig)

    # latest weights
    latest = weights_df.sort_values('date').groupby(['strategy', 'asset'], as_index=False).last()
    latest.to_csv(figdata_dir / 'figdata_weights_latest.csv', index=False)
    if not latest.empty:
        pivot = latest.pivot(index='strategy', columns='asset', values='weight').fillna(0)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        bottom = np.zeros(len(pivot))
        for col in pivot.columns:
            vals = pivot[col].to_numpy()
            ax.bar(pivot.index, vals, bottom=bottom, label=col)
            bottom += vals
        ax.set_title('Latest weights by strategy')
        ax.set_ylabel('Weight')
        ax.tick_params(axis='x', rotation=30)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        fig.tight_layout()
        fig.savefig(fig_dir / 'fig_weights_latest.png', dpi=150)
        fig.savefig(fig_dir / 'fig_weights_latest.pdf')
        plt.close(fig)

    # dr db time series
    drdb_df.to_csv(figdata_dir / 'figdata_dr_db_timeseries.csv', index=False)
    if not drdb_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for strategy, grp in drdb_df.groupby('strategy'):
            ax.plot(pd.to_datetime(grp['date']), grp['D_B'], label=f'{strategy} D_B')
        ax.set_title('D_B over time')
        ax.set_ylabel('D_B')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        fig.tight_layout()
        fig.savefig(fig_dir / 'fig_dr_db_timeseries.png', dpi=150)
        fig.savefig(fig_dir / 'fig_dr_db_timeseries.pdf')
        plt.close(fig)


def export_run_outputs(
    config: dict[str, Any],
    root: str | Path,
    weights_df: pd.DataFrame,
    ctr_df: pd.DataFrame,
    ctb_df: pd.DataFrame,
    drdb_df: pd.DataFrame,
    objective_df: pd.DataFrame,
    monthly_returns_df: pd.DataFrame,
    turnover_df: pd.DataFrame,
    solver_df: pd.DataFrame,
) -> RunArtifacts:
    root = Path(root)
    run_cfg = config['run']
    run_id = run_cfg['run_id']
    result_dir = root / 'results' / 'runs' / run_id
    analysis_dir = ensure_dir(result_dir / 'analysis')
    diagnostics_dir = ensure_dir(result_dir / 'diagnostics')
    figures_dir = ensure_dir(result_dir / 'figures')
    figdata_dir = ensure_dir(result_dir / 'figure_data')
    tables_dir = ensure_dir(result_dir / 'tables')
    manifest_dir = ensure_dir(result_dir / 'manifest')

    weights_df.to_csv(analysis_dir / 'weights.csv', index=False)
    ctr_df.to_csv(analysis_dir / 'ctr_long.csv', index=False)
    ctb_df.to_csv(analysis_dir / 'ctb_long.csv', index=False)
    drdb_df.to_csv(analysis_dir / 'dr_db_timeseries.csv', index=False)
    objective_df.to_csv(analysis_dir / 'objective_terms.csv', index=False)
    turnover_df.to_csv(analysis_dir / 'turnover_timeseries.csv', index=False)
    monthly_returns_df.to_csv(analysis_dir / 'monthly_returns.csv', index=False)
    solver_df.to_csv(diagnostics_dir / 'solver_diagnostics.csv', index=False)

    summary_df = _compute_summary(monthly_returns_df, drdb_df, turnover_df, solver_df)
    summary_df.to_csv(analysis_dir / 'summary_metrics.csv', index=False)

    run_diagnostics = {
        'run_id': run_id,
        'timestamp_utc': timestamp_utc(),
        'n_rows_weights': int(len(weights_df)),
        'n_rows_drdb': int(len(drdb_df)),
        'n_rows_solver': int(len(solver_df)),
    }
    analysis_pack = _analysis_pack(summary_df)
    write_json(analysis_dir / 'analysis_pack.json', analysis_pack)
    write_json(diagnostics_dir / 'run_diagnostics.json', run_diagnostics)
    _write_basic_figures(summary_df, weights_df, drdb_df, figures_dir, figdata_dir)

    # tables
    summary_df.to_csv(tables_dir / 'table_summary_metrics.csv', index=False)
    _write_latex_table_safe(summary_df, tables_dir / 'table_summary_metrics.tex')

    manifest = {
        'run_id': run_id,
        'run_type': run_cfg['run_type'],
        'timestamp_utc': timestamp_utc(),
        'universe_name': config.get('data', {}).get('universe_name', config.get('toy', {}).get('scenario_name', 'toy')),
        'assets': config.get('data', {}).get('assets', config.get('toy', {}).get('assets', [])),
        'date_start': config.get('data', {}).get('evaluation_start', run_cfg.get('run_date', '')),
        'date_end': config.get('data', {}).get('evaluation_end', run_cfg.get('run_date', '')),
        'rebalance_frequency': config.get('data', {}).get('rebalance_frequency', 'static'),
        'cov_estimator': config.get('data', {}).get('cov_estimator', 'toy_fixed_sigma'),
        'strategies': ['EW', 'GMV', 'CtR-only', 'MDP', 'CtB-only', 'Main'],
        'main_params': {
            'delta': config['model']['delta'],
            'eta': config['model']['eta'],
            'gamma': config['model']['gamma'],
            'rho': config['model']['rho'],
        },
        'status': 'success',
    }
    write_json(manifest_dir / 'run_manifest.json', manifest)

    registry_row = {
        'run_id': run_id,
        'run_type': run_cfg['run_type'],
        'status': 'success',
        'scope': run_cfg.get('scope', 'main'),
        'objective_role': run_cfg.get('objective_role', 'development'),
        'date_registered': timestamp_utc()[:10],
        'data_window': run_cfg.get('data_window', ''),
        'universe_name': manifest['universe_name'],
        'cov_estimator': manifest['cov_estimator'],
        'strategies': ','.join(manifest['strategies']),
        'main_params': json.dumps(manifest['main_params']),
        'solver_profile': run_cfg.get('solver_profile', 'default_pg_newton'),
        'result_path': str(result_dir),
        'eligible_for_main_text': str(run_cfg.get('eligible_for_main_text', False)),
        'notes': run_cfg.get('notes', ''),
    }
    append_registry_row(root, registry_row)
    return RunArtifacts(run_id=run_id, result_dir=result_dir)
