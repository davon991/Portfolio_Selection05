python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

python scriptsun_experiment.py --config configs	oy_smoke.yaml
python scriptsun_experiment.py --config configs	oy_a.yaml
python scriptsun_experiment.py --config configs	oy_b.yaml
python scriptsun_experiment.py --config configs	oy_c.yaml
python scriptsun_experiment.py --config configs	oy_d.yaml
python scriptsun_experiment.py --config configs	oy_e.yaml
python scriptsun_experiment.py --config configs	oy_f.yaml
python scriptsun_experiment.py --config configs	oy_g.yaml

python scriptsun_experiment.py --config configs\minimal_real.yaml
python scriptsun_calibration.py --config configs\calibration.yaml
python scriptsun_experiment.py --config configsull_test.yaml
python scriptsun_experiment.py --config configsull_test_d04.yaml
python scriptsun_gmv_recheck.py --config configs\minimal_real.yaml
python scriptsun_gmv_recheck.py --config configsull_test.yaml
python scriptsun_inference.py --run-id full__main__test__20260419__01
python scriptsggregate_solver_reliability.py
