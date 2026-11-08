.PHONY: install test simulation biology report v2-data v2-audit v2-main v2-sweep v2-ablation v2-baselines v2-report v2
install:
	python -m pip install -e .
test:
	python -m unittest discover -s tests -v
simulation:
	python -m kan_mlp.study --config configs/simulation.json --out results/simulation
biology:
	python -m kan_mlp.study --config configs/biology.json --out results/biology
report:
	python scripts/summarize.py results/simulation
	python scripts/summarize.py results/biology

# ---- v2 research study (NumPy engine; no torch needed) ----
WORKERS ?= 2
v2-data:
	python scripts/fetch_data_v2.py
v2-audit:
	python scripts/run_audit.py --out results/v2/audit.json --doc docs/structure_audit.md
v2-main:
	python -m kan_mlp.run_v2 --config configs/v2_main.json --out results/v2/main --workers $(WORKERS)
v2-sweep:
	python -m kan_mlp.run_v2 --config configs/v2_sample_size.json --out results/v2/sample_size --workers $(WORKERS)
v2-ablation:
	python -m kan_mlp.run_v2 --config configs/v2_ablation_period.json --out results/v2/ablation_period --workers $(WORKERS)
v2-baselines:
	python scripts/baselines_v2.py --config configs/v2_main.json --config configs/v2_ablation_period.json --out results/v2/baselines.csv
v2-report:
	python scripts/analyze_v2.py --runs results/v2/main --audit results/v2/audit.json --baselines results/v2/baselines.csv --sweep results/v2/sample_size
	python scripts/analyze_v2.py --runs results/v2/ablation_period --baselines results/v2/baselines.csv
v2: v2-audit v2-main v2-sweep v2-ablation v2-baselines v2-report
