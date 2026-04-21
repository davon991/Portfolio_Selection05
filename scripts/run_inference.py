
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_run_dir(root: Path, run_id: str) -> Path:
    run_dir = root / 'results' / 'runs' / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f'Run directory not found: {run_dir}')
    return run_dir


def _moving_block_bootstrap(arr: np.ndarray, block_size: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == 0:
        return np.array([])
    block_size = max(1, min(block_size, n))
    starts = np.arange(0, n - block_size + 1)
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = []
        while len(sample) < n:
            s = int(rng.choice(starts))
            sample.extend(arr[s:s + block_size])
        sample = np.asarray(sample[:n], dtype=float)
        out[b] = float(sample.mean())
    return out


def _paired_diff_series(df: pd.DataFrame, metric_col: str, main: str, benchmark: str) -> pd.Series:
    pivot = df.pivot(index='date', columns='strategy', values=metric_col).sort_index()
    need = [main, benchmark]
    miss = [c for c in need if c not in pivot.columns]
    if miss:
        raise KeyError(f'Missing strategies for metric={metric_col}: {miss}')
    diff = (pivot[main] - pivot[benchmark]).dropna()
    diff.name = f'{main}_minus_{benchmark}'
    return diff


def _summarize_metric(diff: pd.Series, preferred_sign: int, block_size: int, n_boot: int, seed: int) -> dict:
    obs = float(diff.mean()) if len(diff) else float('nan')
    boots = _moving_block_bootstrap(diff.to_numpy(), block_size=block_size, n_boot=n_boot, seed=seed) if len(diff) else np.array([])
    if len(boots):
        ci_low = float(np.quantile(boots, 0.025))
        ci_high = float(np.quantile(boots, 0.975))
        if preferred_sign < 0:
            p_one_sided = float((boots >= 0.0).mean())
        else:
            p_one_sided = float((boots <= 0.0).mean())
        p_two_sided = float(2 * min((boots <= 0.0).mean(), (boots >= 0.0).mean()))
    else:
        ci_low = ci_high = p_one_sided = p_two_sided = float('nan')
    return {
        'n_obs': int(len(diff)),
        'mean_diff': obs,
        'ci_95_low': ci_low,
        'ci_95_high': ci_high,
        'p_one_sided': p_one_sided,
        'p_two_sided': p_two_sided,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--benchmarks', nargs='*', default=['CtR-only', 'CtB-only', 'EW', 'GMV'])
    parser.add_argument('--main-strategy', default='Main')
    parser.add_argument('--block-size', type=int, default=6)
    parser.add_argument('--n-bootstrap', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_dir = _load_run_dir(ROOT, args.run_id)
    analysis_dir = run_dir / 'analysis'
    out_dir = analysis_dir / 'inference'
    out_dir.mkdir(parents=True, exist_ok=True)

    monthly = pd.read_csv(analysis_dir / 'monthly_returns.csv')
    drdb = pd.read_csv(analysis_dir / 'dr_db_timeseries.csv')
    turnover = pd.read_csv(analysis_dir / 'turnover_timeseries.csv')

    metric_sources = {
        'period_return': (monthly, +1),
        'D_R': (drdb, -1),
        'D_B': (drdb, -1),
        'band_active': (drdb, -1),
        'band_violation': (drdb, -1),
        'turnover': (turnover, -1),
    }

    rows = []
    boot_rows = []
    test_rows = []

    for bench in args.benchmarks:
        for metric, (df, preferred_sign) in metric_sources.items():
            diff = _paired_diff_series(df, metric, args.main_strategy, bench)
            summary = _summarize_metric(diff, preferred_sign, args.block_size, args.n_bootstrap, args.seed)
            row = {
                'run_id': args.run_id,
                'main_strategy': args.main_strategy,
                'benchmark_strategy': bench,
                'metric': metric,
                'preferred_sign': preferred_sign,
                **summary,
            }
            rows.append(row)
            test_rows.append({
                'run_id': args.run_id,
                'main_strategy': args.main_strategy,
                'benchmark_strategy': bench,
                'metric': metric,
                'null_hypothesis': 'mean_diff = 0',
                'alternative': 'mean_diff < 0' if preferred_sign < 0 else 'mean_diff > 0',
                'p_one_sided': summary['p_one_sided'],
                'p_two_sided': summary['p_two_sided'],
                'reject_at_5pct': bool(pd.notna(summary['p_one_sided']) and summary['p_one_sided'] < 0.05),
            })
            boots = _moving_block_bootstrap(diff.to_numpy(), block_size=args.block_size, n_boot=args.n_bootstrap, seed=args.seed)
            if len(boots):
                boot_rows.append(pd.DataFrame({
                    'run_id': args.run_id,
                    'main_strategy': args.main_strategy,
                    'benchmark_strategy': bench,
                    'metric': metric,
                    'bootstrap_mean_diff': boots,
                }))

    summary_df = pd.DataFrame(rows)
    tests_df = pd.DataFrame(test_rows)
    boots_df = pd.concat(boot_rows, ignore_index=True) if boot_rows else pd.DataFrame(columns=['run_id','main_strategy','benchmark_strategy','metric','bootstrap_mean_diff'])

    summary_df.to_csv(out_dir / 'inference_summary.csv', index=False)
    tests_df.to_csv(out_dir / 'hypothesis_tests.csv', index=False)
    boots_df.to_csv(out_dir / 'bootstrap_metric_deltas.csv', index=False)

    notes = f"""# Inference notes

Run: `{args.run_id}`

- Bootstrap type: moving block bootstrap
- Block size: {args.block_size} months
- Bootstrap replications: {args.n_bootstrap}
- Main strategy: {args.main_strategy}
- Benchmarks: {', '.join(args.benchmarks)}

Interpretation rule:
- For `D_R`, `D_B`, `band_active`, `band_violation`, and `turnover`, negative mean differences are favorable for Main.
- For `period_return`, positive mean differences are favorable for Main.
- Use `p_one_sided` for directional claims and `ci_95_*` for uncertainty reporting.
"""
    (out_dir / 'inference_notes.md').write_text(notes, encoding='utf-8')
    print(f'INFERENCE_DIR={out_dir}')


if __name__ == '__main__':
    main()
