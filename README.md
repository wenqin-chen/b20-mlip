# b20-mlip

`b20-mlip` (package `b20mlip`, MIT): fine-tune MACE-MPA-0 (equivariant GNN) on in-house
spin-polarised Quantum ESPRESSO frames of B20 skyrmion hosts (FeSi, CoSi, MnSi, FeGe); measure
what it fixes (force/phonon softening) and costs (forgetting); deploy in ASE and LAMMPS MD;
umbrella-sample a vacancy hop; drive it with a provenance-checked tool-calling agent.

**Status: under construction.** Only the foundation (models, config, provenance, I/O, executors,
CLI skeleton, CI) exists. No results are reported yet; every number that will appear here is
regenerated from `reports/numbers.json`, which is itself built from run manifests and gated by
`b20mlip report audit --strict` (see `CONTRACTS.md` section 8).

## Development

```bash
make setup          # uv sync --extra dev  (Python 3.11, CPU torch)
make lint           # ruff check + ruff format --check + mypy
make test           # offline test suite: uv run pytest -q
make test-cov       # same, with coverage
uv run b20mlip --help
```

Tests never touch the network (`tests/test_no_network.py` blocks sockets for the whole session)
and never import `matbench_discovery`. Markers `network`, `slow` and `cluster` are excluded by
default; run them explicitly with `uv run pytest -m network`.

Design documents: `SPEC.md` (science and schedule), `CONTRACTS.md` (binding interfaces),
`docs/design/` (option studies and environment facts).
