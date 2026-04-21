from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ctrctb.exports.results import _write_basic_figures
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()

    base = ROOT / 'results' / 'runs' / args.run_id
    summary = pd.read_csv(base / 'analysis' / 'summary_metrics.csv')
    weights = pd.read_csv(base / 'analysis' / 'weights.csv')
    drdb = pd.read_csv(base / 'analysis' / 'dr_db_timeseries.csv')
    _write_basic_figures(summary, weights, drdb, base / 'figures', base / 'figure_data')
    print(f'FIGURES_REBUILT={base / "figures"}')


if __name__ == '__main__':
    main()
