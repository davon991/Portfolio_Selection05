
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ctrctb.backtest.runner import run_real_backtest
from ctrctb.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--suffix', default='gmvfix')
    args = parser.parse_args()

    cfg = load_yaml(ROOT / args.config)
    run_cfg = deepcopy(cfg)
    run_cfg['run']['run_id'] = f"{cfg['run']['run_id']}__{args.suffix}"
    run_cfg['run']['scope'] = 'gmv_recheck'
    run_cfg['run']['objective_role'] = 'baseline_reliability'
    run_cfg['run']['eligible_for_main_text'] = False
    notes = run_cfg['run'].get('notes', '')
    run_cfg['run']['notes'] = f"{notes} GMV decoupled recheck using dedicated long-only SLSQP solver.".strip()

    artifacts = run_real_backtest(run_cfg, ROOT)
    print(f'RUN_ID={artifacts.run_id}')
    print(f'RESULT_DIR={artifacts.result_dir}')


if __name__ == '__main__':
    main()
