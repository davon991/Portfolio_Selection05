from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()

    base = ROOT / 'results' / 'runs' / args.run_id
    summary = pd.read_csv(base / 'analysis' / 'summary_metrics.csv')
    out_dir = base / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / 'table_summary_metrics.csv', index=False)
    summary.to_latex(out_dir / 'table_summary_metrics.tex', index=False, float_format='%.4f')
    print(f'TABLES_REBUILT={out_dir}')


if __name__ == '__main__':
    main()
