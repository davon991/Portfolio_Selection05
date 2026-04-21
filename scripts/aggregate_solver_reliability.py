
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_manifest(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    runs_root = root / 'results' / 'runs'
    out_dir = root / 'results' / 'solver_reliability'
    out_dir.mkdir(parents=True, exist_ok=True)

    detailed = []
    failures = []

    for run_dir in sorted(runs_root.glob('*')):
        solver_path = run_dir / 'diagnostics' / 'solver_diagnostics.csv'
        manifest_path = run_dir / 'manifest' / 'run_manifest.json'
        if not solver_path.exists() or not manifest_path.exists():
            continue
        manifest = _load_manifest(manifest_path)
        solver = pd.read_csv(solver_path)
        if solver.empty:
            continue
        solver['run_id'] = manifest['run_id']
        solver['run_type'] = manifest.get('run_type', 'unknown')
        solver['scope'] = manifest.get('scope', '')
        solver['date_start'] = manifest.get('date_start', '')
        solver['date_end'] = manifest.get('date_end', '')
        detailed.append(solver)
        bad = solver[(solver['status'] != 'success') | (solver['fallback_used'] == 1) | (solver['converged'] == 0)]
        if not bad.empty:
            failures.append(bad.copy())

    if not detailed:
        raise RuntimeError('No solver_diagnostics.csv files found under results/runs.')

    detail_df = pd.concat(detailed, ignore_index=True)
    detail_df.to_csv(out_dir / 'solver_stage_breakdown.csv', index=False)

    def _rate(s: pd.Series, v) -> float:
        return float((s == v).mean()) if len(s) else float('nan')

    grouped = detail_df.groupby(['run_type', 'strategy'], dropna=False)
    summary = grouped.agg(
        n_rows=('status', 'size'),
        success_rate=('status', lambda s: _rate(s, 'success')),
        partial_rate=('status', lambda s: _rate(s, 'partial')),
        fallback_rate=('fallback_used', 'mean'),
        nonconverged_rate=('converged', lambda s: float((s == 0).mean())),
        mean_pg_iter=('iterations_pg', 'mean'),
        mean_newton_iter=('iterations_newton', 'mean'),
        median_kkt_residual=('kkt_residual', 'median'),
        max_kkt_residual=('kkt_residual', 'max'),
    ).reset_index()
    summary.to_csv(out_dir / 'solver_reliability_summary.csv', index=False)

    failure_df = pd.concat(failures, ignore_index=True) if failures else pd.DataFrame(columns=detail_df.columns)
    failure_df.to_csv(out_dir / 'solver_failure_catalog.csv', index=False)

    print(f'SOLVER_RELIABILITY_DIR={out_dir}')


if __name__ == '__main__':
    main()
