from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ctrctb.utils.registry import read_registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    args = parser.parse_args()
    rows = read_registry(Path(args.root))
    print(f'REGISTRY_ROWS={len(rows)}')
    if rows:
        print(f'LAST_RUN_ID={rows[-1]["run_id"]}')


if __name__ == '__main__':
    main()
