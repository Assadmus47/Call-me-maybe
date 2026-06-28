install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +

lint:
	uv run flake8 . --exclude=.venv,llm_sdk
	uv run mypy . --exclude=llm_sdk --ignore-missing-imports --warn-return-any --warn-unused-ignores --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 . --exclude=.venv,llm_sdk
	uv run mypy . --exclude=llm_sdk --strict --ignore-missing-imports

.PHONY: run debug install lint lint-strict clean%                                                                                                                                                                                                                                                                                                                         
