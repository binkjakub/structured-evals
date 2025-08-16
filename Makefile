all: quality
check_dirs := src tests examples

install:
	uv sync --locked --all-extras --dev

quality:
	uv run pre-commit run --all-files
	uv run mypy --install-types --non-interactive $(check_dirs)

fix:
	uv run pre-commit run --all-files

test:
	uv run coverage run -m pytest tests

build:
	uv build
