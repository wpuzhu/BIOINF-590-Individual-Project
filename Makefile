.PHONY: install test simulation biology report
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

# ---- spatial SVG benchmark (branch spatial-svg-benchmark) ----
spatial-test:
	cd spatial_benchmark && python -m unittest discover -s tests
spatial-pilot:
	cd spatial_benchmark && python scripts/run_pilot.py --out results/pilot --workers 2 && python scripts/make_figures.py --pilot results/pilot
