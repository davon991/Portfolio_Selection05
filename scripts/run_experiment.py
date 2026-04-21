from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ctrctb.backtest.runner import run_real_backtest, run_toy_experiment
from ctrctb.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_yaml(ROOT / args.config)
    run_type = cfg['run']['run_type']
    if run_type == 'toy':
        artifacts = run_toy_experiment(cfg, ROOT)
    elif run_type in {'minimal_real', 'full'}:
        artifacts = run_real_backtest(cfg, ROOT)
    else:
        raise ValueError(f'Unsupported run_type for run_experiment.py: {run_type}')

    print(f'RUN_ID={artifacts.run_id}')
    print(f'RESULT_DIR={artifacts.result_dir}')


if __name__ == '__main__':
    main()
