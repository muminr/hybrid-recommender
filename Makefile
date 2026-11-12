.PHONY: install test lint data eval eval-synthetic train serve

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check . && ruff format --check .

data:
	python scripts/download_movielens.py

eval:
	python -m hybrec.evaluate --data data/ml-latest-small

eval-synthetic:
	python -m hybrec.evaluate --synthetic

train:
	python -m hybrec.train --data data/ml-latest-small --out artifacts/model.joblib

serve:
	MODEL_PATH=artifacts/model.joblib uvicorn hybrec.service:app --reload
