.PHONY: install lint typecheck test check prepare-data verify-data smoke clean

install:
	uv sync --locked --extra dev

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest -W error::DeprecationWarning

check: lint typecheck test

prepare-data:
	uv run garbage prepare-data --set data.data_dir data/raw

verify-data:
	uv run garbage verify-data --set data.data_dir data/raw

smoke:
	uv run garbage train --config configs/learning_minimal.yaml \
		--set data.data_dir data/raw --set train.epochs 1 --set train.batch_size 16 \
		--set device cpu --set run_name smoke

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -not -path "./.venv/*" -exec rm -rf {} +
