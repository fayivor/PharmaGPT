# Makefile for PharmaGPT

.PHONY: help install install-dev test lint format clean run demo setup

# Default target
help:
	@echo "PharmaGPT - Intelligent Drug & Clinical Trial Assistant"
	@echo ""
	@echo "Available commands:"
	@echo "  install      Install dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  setup        Setup environment and configuration"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  clean        Clean up temporary files"
	@echo "  run          Run Streamlit app"
	@echo "  demo         Run demo script"
	@echo "  build        Build the project"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest black flake8 mypy

# Setup
setup:
	@echo "Setting up PharmaGPT..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from template"; \
		echo "Please edit .env with your API keys"; \
	fi
	@mkdir -p data/vector_store
	@mkdir -p logs
	@echo "Setup complete!"

# Testing
test:
	python -m pytest tests/ -v

test-coverage:
	python -m pytest tests/ --cov=. --cov-report=html

# Code quality
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black . --line-length 100
	isort . --profile black

type-check:
	mypy . --ignore-missing-imports

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

# Running
run:
	streamlit run app/main.py

demo:
	python demo.py

# Building
build:
	python setup.py sdist bdist_wheel

# Development workflow
dev-setup: install-dev setup
	@echo "Development environment ready!"

check: lint type-check test
	@echo "All checks passed!"

# Docker commands (if you want to add Docker support later)
docker-build:
	docker build -t pharmagpt .

docker-run:
	docker run -p 8501:8501 pharmagpt

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "Documentation available in README.md"

# Quick start
quick-start: install setup
	@echo ""
	@echo "Quick Start Complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Edit .env file with your API keys"
	@echo "2. Run 'make demo' to test functionality"
	@echo "3. Run 'make run' to start the web interface"
	@echo ""
