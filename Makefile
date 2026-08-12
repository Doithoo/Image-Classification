.PHONY: install lint test prepare-data smoke

install:
	uv pip install -e ".[dev]"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

test:
	pytest tests/ -q

prepare-data:
	garbage prepare-data --set data.data_dir data/raw

smoke:  ## 1-epoch CPU smoke run on real data
	garbage train --config configs/resnet50.yaml \
		--set train.epochs 1 --set train.batch_size 16 \
		--set data.num_workers 0 --set device cpu --set run_name smoke
