# Contributing to b20-mlip

## How this repository is built

The code is developed with **Claude Code** (Anthropic's CLI coding agent) as the coding agent,
orchestrated by the author in tiers that follow the build order of `CONTRACTS.md` section 1:
each tier implements a set of modules against the binding contracts (names, fields,
signatures, CLI, config, manifest schema, tests) and hands the next tier a green CI.
Design decisions and environment facts live in `docs/design/`; the science and schedule in
`SPEC.md`. Human review happens at tier boundaries and before every tag.

## Ground rules

1. **Contracts are binding.** `CONTRACTS.md` names every model, signature and CLI command. If a
   contract turns out to be impossible, implement the nearest thing and record the deviation
   in the PR under "contract issues" — do not silently rename.
2. **No fabricated numbers.** README numbers are regenerated from `reports/numbers.json`,
   which is built from run manifests and checked by `b20mlip report audit --strict`
   (honesty gates A1–A11). Negative results ship.
3. **Every stage writes a manifest** (`runs/<stage>/<run_id>/manifest.json`) through
   `b20mlip.provenance.run_stage`, even when it fails. Stage functions have the signature
   `run(cfg: Settings, ctx: RunContext, **kw) -> StageResult` and never print to README.
4. **Tests are offline.** `pytest` blocks sockets for the whole session; anything needing the
   network, a cluster or more than a minute carries the `network`, `cluster` or `slow` marker.
   `matbench_discovery` is never imported or installed.
5. **Style:** `ruff check` + `ruff format` (line length 100), `mypy` on `src/`, Python 3.11.

## Workflow

```bash
make setup                          # uv sync --extra dev
make lint && make test              # must be green before a PR
uv run pytest -m slow               # opt-in: tiny MACE training etc.
```

Commit messages: imperative subject, body explaining *why*; PRs list contract issues, new
dependencies and anything the next tier must know.
