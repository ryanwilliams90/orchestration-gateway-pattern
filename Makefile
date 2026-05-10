.PHONY: install lint type test all clean

PY ?= python3.11
VENV := .venv

install:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e '.[dev]'

lint:
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/ruff format --check src tests

type:
	$(VENV)/bin/mypy src

test:
	PYTHONPATH=src $(VENV)/bin/pytest -q

all: lint type test

clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
