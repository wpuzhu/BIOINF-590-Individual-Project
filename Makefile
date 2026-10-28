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
