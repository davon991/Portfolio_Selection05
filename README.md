# CtR–CtB Portfolio Project Skeleton

This is a local, runnable project skeleton aligned with the frozen control files:

- `00_project_charter.md`
- `01_notation_master.md`
- `02_definition_formula_ledger.md`
- `03_thesis_skeleton.md`
- `03A_contribution_claims.md`
- `05_data_contract.md`
- `06_model_contract.md`
- `06A_baseline_ladder.md`
- `07_solver_contract.md`
- `08_result_contract.md`
- `08A_statistical_validation.md`
- `experiment_registry.md`
- `synthetic_toy_experiments.md`

## What this skeleton does

- Runs toy / synthetic experiments.
- Runs a minimal real ETF backtest using monthly rebalancing and rolling covariance estimates.
- Runs a basic calibration sweep on `delta` and `eta`.
- Exports result files aligned with the result contract.
- Updates a CSV-based experiment registry.

## What this skeleton is and is not

This is a **research-grade starter implementation**. It is designed to be:

- coherent with the frozen contracts;
- locally runnable on Windows PowerShell;
- easy to inspect and extend;
- strict about result files and registry updates.

It is **not** a claim that the implementation is already final-best or empirically validated. The intended process is:

```text
Toy code stage
→ Minimal real ETF run
→ Calibration
→ Full experiments
→ Feed back into chapters / defense / PPT
```

## Quick start

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts\run_experiment.py --config configs\toy_smoke.yaml
python scripts\run_experiment.py --config configs\minimal_real.yaml
python scripts\run_calibration.py --config configs\calibration.yaml
python scripts\run_experiment.py --config configs\full_test.yaml
python scripts\build_figures.py --run-id <full_run_id>
python scripts\build_tables.py --run-id <full_run_id>
python scripts\update_registry.py --root .
```

## Output locations

Each run writes to:

```text
results/runs/<run_id>/
```

Expected subdirectories include:

- `analysis/`
- `diagnostics/`
- `figures/`
- `figure_data/`
- `tables/`
- `manifest/`

## Minimal files to send back for evaluation

After each run, send back at least:

- `results/runs/<run_id>/analysis/summary_metrics.csv`
- `results/runs/<run_id>/analysis/weights.csv`
- `results/runs/<run_id>/analysis/ctr_long.csv`
- `results/runs/<run_id>/analysis/ctb_long.csv`
- `results/runs/<run_id>/analysis/dr_db_timeseries.csv`
- `results/runs/<run_id>/diagnostics/solver_diagnostics.csv`
- `results/runs/<run_id>/analysis/analysis_pack.json`
- `results/runs/<run_id>/manifest/run_manifest.json`

For calibration runs also send:

- `results/runs/<run_id>/analysis/calibration_log.csv`
- `results/runs/<run_id>/analysis/calibration_summary.csv`
