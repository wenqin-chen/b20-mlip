# Agent eval (`evals/`)

SPEC.md section 7: a budgeted, provenance-checked tool-calling agent scored on 12 tasks.
Everything here is offline-safe; the live backend needs `ANTHROPIC_API_KEY` and is never used in CI.

## Files

| path | what |
|---|---|
| `agent_tasks.jsonl` | the 12 tasks (one JSON object per line, see below) |
| `gold_<model_label>.json` | gold answers written by `b20mlip screen` for one MLIP (not committed; regenerate) |
| `refs/phonons_<compound>_<label>.json` | reference `PhononResult` files `compare_phonons` reads (not committed; see below) |
| `../tests/fixtures/traces/*.jsonl` | three **synthetic** recorded traces the mock backend replays in CI |
| `../runs/agent/traces/<task_id>.jsonl` | real recorded traces (`agent run --record`; redacted; gitignored with `runs/`) |

## Task format

Each line holds the `AgentTask` fields (`task_id`, `prompt`, `tools_allowed`, `budget`, `gold`,
`tolerance`, `injected_failure`; CONTRACTS.md section 2) plus the planner keys the scripted
backend uses:

- `kind` — the canonical tool sequence (`phonons_imaginary`, `relax_a0`, `softening_rank`,
  `force_mae`, `select_frames`, `dft_plan`, `md_drift`, `relax_report`, `inventory_relax_report`);
- `params` — compound(s), dataset/model labels, `n`, `T`, `ps`, `natoms`, `tier`, `reference`, ...;
- `gold_from: "screen"` — the gold values are **recomputed** by `b20mlip screen` with the model
  under test; the JSONL keeps `gold: null` because every value depends on the model, the
  configuration (`agent.phonon_supercell`, `eval.fmax`, the seed) and the references;
- `tolerance` — the graded keys with their absolute tolerances (`0` = exact; strings must match).

Four tasks carry an `injected_failure` (`tool_error`, `bad_structure`, `budget_exhausted`,
`sum_rule`); the runner injects it at run time and `recovery` scores whether the answer is still
right. Dataset labels are file stems under `data/frames/` (`labelled_r0` = `dft collect` output,
`candidates_r0_filtered` = the round-0 pool) or `--frames LABEL=PATH` registrations; model
labels come from `list_data("models")` (the `--model` file, `--committee` files, `models/**`).

## Workflow

```bash
# 1. references for compare_phonons (production: from the dft tier's QE force sets)
uv run python -c "
from b20mlip.phonons import harmonic
for c in ('FeSi', 'CoSi', 'MnSi'):
    harmonic.to_json(harmonic.from_force_sets(None, f'data/phonons/force_sets_{c}.json'),
                     f'evals/refs/phonons_{c}_qe.json')"

# 2. gold for the model under test (deterministic scripted plans, injected failures stripped)
uv run b20mlip screen --model models/foundation/mace-mpa-0-medium.model \
    --compounds FeSi,CoSi,MnSi --tasks evals/agent_tasks.jsonl --out evals/gold_B0.json

# 3a. CI: replay recorded traces (fixtures) and score them against the gold
uv run b20mlip agent eval --backend mock --trace-dir tests/fixtures/traces --gold evals/gold_B0.json \
    --model models/foundation/mace-mpa-0-medium.model --limit 2

# 3b. live: Claude Opus 5 through the SDK tool runner (needs ANTHROPIC_API_KEY; never in CI)
export ANTHROPIC_API_KEY=...            # from your shell or a gitignored .env; never committed
uv run b20mlip agent eval --backend anthropic --gold evals/gold_B0.json \
    --model models/foundation/mace-mpa-0-medium.model
uv run b20mlip agent run --backend anthropic --task evals/agent_tasks.jsonl:t01-phonons-imaginary-FeSi \
    --model models/foundation/mace-mpa-0-medium.model --record     # writes runs/agent/traces/<task_id>.jsonl

# 4. README numbers (agent.eval.*): only non-mock backends are rendered (gate A10)
uv run b20mlip report build --readme && uv run b20mlip report audit --strict
```

Every tool call is its own `agent.run` sub-stage (`runs/agent.run/<run_id>/`: manifest,
`result.json`, the stage outputs) and every agent run writes `trace.jsonl` (argument, result and
manifest hashes, redacted), `report.json` and `numbers.json`. A number is *grounded* only when
its citation resolves to an `ok` manifest whose `result.json` holds that value; `write_report`
rejects any other number. Tool sub-runs never publish stage numbers (`eval.errors.*` etc.): an
agent cannot write README numbers through a side door.

## Scores (`agent eval`)

`agent.eval.{accuracy, invalid_call_rate, dag_valid_rate, provenance_rate, recovery_rate,
tokens_per_task, usd_per_task, n_tasks}` — means over tasks with a seeded bootstrap CI over
tasks, plus the A10 meta (`backend`, `model_id`, `trace_path`). Accuracy compares the graded
keys with the gold within the per-key tolerance; `invalid_call_rate` counts guard violations and
tool errors, except the call that received the task's injected failure (the agent cannot foresee
it; `recovery_rate` scores the response); `dag_valid_rate` checks the tool order (`get_structure`/`list_data` before
`relax`/`phonons`/`run_md`, `compare_phonons` after its `phonons`, `select_frames` after
`evaluate_errors`, `write_report` last); `provenance_rate` is the grounded fraction;
`recovery_rate` is accuracy on the injected-failure tasks; cost uses the Claude Opus 5 list
price ($5 / $25 per million input / output tokens, `b20mlip.agent.backends.PRICES_USD_PER_MTOK`).

Live cost: a task takes 2-9 tool calls with ~8-29k input and ~0.4-1.2k output tokens in total;
the first live run (2026-09-22, Claude Opus 5, 12 tasks) used 164k input and 10k output tokens,
$1.08 at list price ($0.09 per task; no prompt caching is used).

## Task revisions

- 2026-09-22 (before the first live run): t03 ranks FeSi and CoSi only (the MnSi QE phonon
  reference was not computed: budget); t04/t05 evaluate `labelled_r0plus` on split `v2`
  (`data/splits/006f595381e8.json`, policy `group_hash_v2`) because the round-0 split
  `d195dc5f2104` let never-trained MnGe frames into T0 and val. Gold is regenerated with `screen`.
- 2026-09-22 (after the first live run, `runs/agent.eval/20260922T174732-26ca37-0`: accuracy
  0.889, invalid-call rate 0.090, DAG-valid 1.0, provenance 0.833, recovery 1.0, $1.08). Each
  miss was traced; three were defects of the eval, not of the agent, and are fixed before the
  second run, which is the one published:
  - t05: the scripted gold took the first three inventory models (`B0`, `b1v1_seed0`,
    `b1v1_seed1`) as the committee, mixing the zero-shot model into the uncertainty estimate
    (gold σ_f,max 0.527 eV/Å); the agent used the three committee members (0.048 eV/Å).
    `list_data("models")` now names the committee, the scripted plan uses it, and the prompt
    names the committee and the model to evaluate.
  - t06: the grader required `submitted`, which the prompt never asked for; the prompt now asks.
  - t03: the prompt did not say at which cell the model phonons are computed; the agent relaxed
    first (s = 0.961 for CoSi, correct compound) while the gold follows the protocol (DFT cell,
    s = 0.934). The prompt now says "at the DFT reference cell".
  - metric: the injected transient error (t09) and the first call blocked by the injected
    budget (t11) were counted as the agent's invalid calls; the injected call is now excluded
    (later calls that ignore the error still count).
  Not changed, because they are the agent's own misses: t04's two malformed calls (tool-call
  markup leaked into the `model` argument; recovered on the third call) and t07/t08 answers that
  carried uncited numbers (`natoms`, `ps`, `converged`), which the provenance gate rejects.
- 2026-09-22, second live run (`runs/agent.eval/20260922T181438-26ca37-0`, the published one):
  accuracy 1.0, invalid-call rate 0.028, DAG-valid 1.0, provenance 0.917, recovery 1.0, $1.03.
  The one invalid call is t04's markup leak again: it happens when the model tries to pass an
  empty string as the first argument, so `evaluate_errors` now documents explicit labels
  (`model="primary"`, `split="none"`; `""` stays accepted) for later runs. The provenance miss
  is t07 again: the answer echoes the task inputs `natoms` and `ps` without a citation.
