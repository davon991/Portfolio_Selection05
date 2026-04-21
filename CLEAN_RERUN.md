# Clean rerun package

This package is a fresh, consolidated code snapshot for rerunning the project from scratch in a **new folder**.

## Recommended directory
Create a new directory such as `C:\Portfolio05` and extract this package there.
Do **not** reuse the old project directory if you want to avoid contamination from previous results.

## Fresh run order
1. `toy_smoke.yaml`
2. `toy_a.yaml` to `toy_g.yaml`
3. `minimal_real.yaml`
4. `calibration.yaml`
5. `full_test.yaml`  (main spec = d08)
6. `full_test_d04.yaml` (robustness)
7. `run_inference.py`
8. `aggregate_solver_reliability.py`

## Final main spec in this package
- delta = 0.02
- eta = 0.05
- gamma = 0.001
- rho = 100.0

## Notes
- `full_test.yaml` has been updated to the final main spec.
- `full_test_d04.yaml` is included for the robustness run.
- GMV is decoupled from the unified PG/Newton solver.
- The package excludes old `results/` outputs to reduce lineage pollution.
