.PHONY: install install-dev lint format typecheck test test-cov clean help init

# Default Python interpreter
PYTHON ?= python3

help:
	@echo "Brainstormer Development Commands"
	@echo "================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install package dependencies"
	@echo "  make install-dev   Install with development dependencies"
	@echo ""
	@echo "Quality:"
	@echo "  make lint          Run linting (ruff)"
	@echo "  make format        Format code (ruff)"
	@echo "  make typecheck     Run type checking (mypy)"
	@echo "  make check         Run all quality checks"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run tests"
	@echo "  make test-cov      Run tests with coverage"
	@echo ""
	@echo "Other:"
	@echo "  make clean         Remove build artifacts"
	@echo "  make init          Initialize a sample project"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

typecheck:
	$(PYTHON) -m mypy src/brainstormer

test:
	$(PYTHON) -m pytest tests/

test-cov:
	$(PYTHON) -m pytest tests/ --cov=src/brainstormer --cov-report=term-missing --cov-report=html

check: lint typecheck test
	@echo "All checks passed!"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

init:
	brainstormer init .
