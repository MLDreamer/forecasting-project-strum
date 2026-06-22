# Windows-compatible Makefile (run via `make` from Git Bash or WSL)
PYTHON := .venv/Scripts/python
PYTEST := .venv/Scripts/pytest
RUFF   := .venv/Scripts/ruff
MYPY   := .venv/Scripts/mypy

.PHONY: install lint format typecheck test gate clean

install:
	pip install -e ".[dev]"

lint:
	$(RUFF) check src/ tests/

format:
	$(RUFF) format src/ tests/

format-check:
	$(RUFF) format --check src/ tests/

typecheck:
	$(MYPY) src/forecasting

test:
	$(PYTEST)

gate: lint format-check typecheck test
	@echo "=== Phase gate PASSED ==="

clean:
	rm -rf data/interim/* data/processed/* outputs/*
