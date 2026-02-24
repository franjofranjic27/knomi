.PHONY: lint format test test-all test-coverage dev dev-stop install

## Install all dependencies (including dev extras)
install:
	uv sync --all-extras

## Run linters and type checker
lint:
	ruff check .
	ruff format --check .
	mypy knomi

## Auto-fix formatting and lint issues
format:
	ruff format .
	ruff check --fix .

## Run unit tests only (no Docker required)
test:
	pytest tests/unit -v --tb=short

## Run all tests (requires Qdrant — run `make dev` first)
test-all:
	pytest -v --tb=short

## Run unit tests with coverage report
test-coverage:
	pytest tests/unit --cov=knomi --cov-report=term-missing --tb=short

## Start Qdrant in the background
dev:
	docker compose up qdrant -d

## Stop all compose services
dev-stop:
	docker compose down
