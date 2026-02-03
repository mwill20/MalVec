.PHONY: help install install-dev test test-security test-coverage lint format typecheck pre-commit clean docker-build docker-run

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install MalVec and runtime dependencies
	pip install -r requirements.txt
	pip install -e .

install-dev: install ## Install MalVec with development dependencies
	pip install -r requirements-dev.txt
	pre-commit install

test: ## Run all tests
	pytest tests/ -v

test-security: ## Run security tests only
	pytest tests/security/ -v

test-coverage: ## Run tests with coverage report
	pytest tests/ -v --cov=malvec --cov-report=html --cov-report=term-missing

lint: ## Run linter
	ruff check .

format: ## Format code
	ruff format .

typecheck: ## Run type checker
	mypy malvec/ --ignore-missing-imports

pre-commit: ## Run all pre-commit hooks
	pre-commit run --all-files

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .mypy_cache .pytest_cache .ruff_cache htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

docker-build: ## Build Docker image
	docker build -t malvec:latest .

docker-run: ## Run MalVec in Docker (usage: make docker-run ARGS="malvec.cli.classify --help")
	docker run --rm -v $(PWD)/data:/app/data malvec:latest $(ARGS)
