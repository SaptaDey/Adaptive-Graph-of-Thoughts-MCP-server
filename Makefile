.PHONY: test install-dev lint format check-types

install-dev:
	poetry install --with dev

test:
	@echo "Running tests..."
	poetry run pytest

lint:
	@echo "Running linter (Ruff check)..."
	poetry run ruff check .

format:
	@echo "Formatting code (Ruff format)..."
	poetry run ruff format .

check-types:
	@echo "Running type checker (MyPy)..."
	poetry run mypy src/
	@echo "Running type checker (Pyright)..."
	poetry run pyright src/

all-checks: lint check-types test
	@echo "All checks passed!"

help:
	@echo "Available targets:"
	@echo "  install-dev   - Install project dependencies including dev tools."
	@echo "  test          - Run the test suite using pytest."
	@echo "  lint          - Run Ruff to check for linting errors."
	@echo "  format        - Format code using Ruff."
	@echo "  check-types   - Run MyPy and Pyright for static type checking."
	@echo "  all-checks    - Run lint, check-types, and test targets sequentially."
