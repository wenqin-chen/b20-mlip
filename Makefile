.PHONY: setup lint fmt test test-cov bench audit schema clean

UV ?= uv
RUN := $(UV) run --no-sync

setup:            ## create .venv and lock (Python 3.11, CPU torch)
	$(UV) sync --extra dev
	# macOS: some tooling flags .venv "hidden" (chflags); CPython >= 3.11.15 then skips the
	# editable-install .pth file and `import b20mlip` fails. Clearing the flag is harmless.
	@if [ "$$(uname)" = "Darwin" ]; then chflags -R nohidden .venv 2>/dev/null || true; fi

lint:             ## ruff check + format check + mypy
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) mypy

fmt:              ## auto-format and fix imports
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

test:             ## offline test suite (network/slow/cluster markers excluded)
	$(RUN) pytest -q

test-cov:         ## offline tests with coverage of the package
	$(RUN) pytest -q --cov=b20mlip --cov-report=term-missing

bench:            ## Day-1 timing task (SPEC section 15); needs MACE-MPA-0 medium cached
	$(RUN) b20mlip bench --out runs/bench/

audit:            ## honesty gates on README + reports/numbers.json
	$(RUN) b20mlip report audit --strict

schema:           ## regenerate docs/manifest.schema.json from the Manifest model
	$(RUN) python -c "from b20mlip.provenance import write_manifest_schema as w; print(w('docs/manifest.schema.json'))"

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
