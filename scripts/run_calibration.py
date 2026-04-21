from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import itertools
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ctrctb.backtest.runner import run_real_backtest
from ctrctb.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_yaml(ROOT / args.config)
    delta_grid = cfg['calibration']['delta_grid']
    eta_grid = cfg['calibration']['eta_grid']
    base = cfg['base_run']
    records = []
    parent_run_id = cfg['run']['run_id']

    for i, (delta, eta) in enumerate(itertools.product(delta_grid, eta_grid), start=1):
        run_cfg = deepcopy(base)
        run_cfg['run']['run_id'] = f"{parent_run_id}__d{i:02d}"
        run_cfg['model']['delta'] = float(delta)
        run_cfg['model']['eta'] = float(eta)
        artifacts = run_real_backtest(run_cfg, ROOT)
        summary = pd.read_csv(artifacts.result_dir / 'analysis' / 'summary_metrics.csv')
        main_row = summary[summary['strategy'] == 'Main'].iloc[0].to_dict()
        main_row.update({'run_id': run_cfg['run']['run_id'], 'delta': delta, 'eta': eta})
        records.append(main_row)

    out_dir = ROOT / 'results' / 'runs' / parent_run_id / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(out_dir / 'calibration_log.csv', index=False)
    if not df.empty:
        accepted = df[(df['failure_rate'] <= cfg['calibration']['max_failure_rate']) & (df['turnover_mean'] <= cfg['calibration']['max_turnover_mean'])]
        accepted = accepted.sort_values(['db_mean', 'dr_mean', 'turnover_mean'])
        accepted.to_csv(out_dir / 'calibration_summary.csv', index=False)
    print(f'CALIBRATION_DIR={out_dir}')


if __name__ == '__main__':
    main()
