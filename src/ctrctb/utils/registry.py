from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable

REGISTRY_COLUMNS = [
    'run_id', 'run_type', 'status', 'scope', 'objective_role', 'date_registered',
    'data_window', 'universe_name', 'cov_estimator', 'strategies', 'main_params',
    'solver_profile', 'result_path', 'eligible_for_main_text', 'notes'
]


def append_registry_row(root: str | Path, row: Dict[str, str]) -> Path:
    root = Path(root)
    registry_path = root / 'results' / 'experiment_registry.csv'
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    exists = registry_path.exists()
    with open(registry_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in REGISTRY_COLUMNS})
    return registry_path


def read_registry(root: str | Path) -> list[dict[str, str]]:
    root = Path(root)
    registry_path = root / 'results' / 'experiment_registry.csv'
    if not registry_path.exists():
        return []
    with open(registry_path, 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))
