.PHONY: install test lint build demo clean

install:
	uv sync

test:
	uv run pytest -q -m "not live"

lint:
	uv run ruff check src tests kaggle-kernel

build:
	uv build

demo:
	uv run vision-check demo --max-identical-delta 0.17 --min-lossy-accuracy 0.8

clean:
	rm -rf dist build *.egg-info .pytest_cache .ruff_cache visioncheck-demo
