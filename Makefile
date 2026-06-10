# Common developer tasks. Run `make help` for the list.
.PHONY: help install test lint demo export clean

help:
	@echo "install  - editable install with dev extras"
	@echo "test     - run the pytest suite"
	@echo "lint     - run ruff"
	@echo "demo     - run the n-gram vs neural-LM comparison"
	@echo "export   - retrain and export weights to docs/model.json"
	@echo "clean    - remove caches and build artifacts"

install:
	pip install -e ".[dev,viz]"

test:
	pytest

lint:
	ruff check src tests examples

demo:
	python examples/compare.py

export:
	python scripts/export_weights.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
