check_dirs := src tests examples

all: quality
quality: lint typecheck

install:
	uv sync --locked --all-extras --dev

lint:
	uv run pre-commit run --all-files

typecheck:
	uv run mypy --install-types --non-interactive $(check_dirs)

fix:
	uv run pre-commit run --all-files

test:
	uv run coverage run --branch -m pytest tests -vv

coverage:
	uv run coverage report --show-missing

coverage-xml:
	uv run coverage xml

coverage-html:
	uv run coverage html

build:
	uv build
