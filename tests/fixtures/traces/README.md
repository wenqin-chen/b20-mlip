# Synthetic agent traces (mock-backend fixtures)

These three files are **synthetic**: they were produced by running the `scripted` backend on
tasks of `evals/agent_tasks.jsonl` with the in-test tiny MACE model and then re-labelled as
if they were live `anthropic` runs (`backend: anthropic`, `model_id: claude-opus-5`, invented
`usage` events with plausible token counts, a synthetic clock starting at 2025-09-18T12:00Z).
No Claude API call was made to create them; every header carries a `synthetic` marker.

They are written in the exact format `b20mlip.agent.trace.Trace` produces (one JSON object per
line; the ten fixed fields `t, kind, tool, args_sha256, args, result_sha256, manifest_sha256,
run_id, tokens_in, tokens_out` on every event, plus per-kind extras), which is what
`MockBackend(trace_path)` replays in CI:

| file | task | calls |
|---|---|---|
| `t01-phonons-imaginary-FeSi.jsonl` | phonon imaginary-mode count of FeSi | `get_structure`, `phonons` |
| `t02-relax-a0-MnSi.jsonl` | relaxed lattice parameter of MnSi | `get_structure`, `relax` |
| `t09-relax-a0-FeSi-tool-error.jsonl` | the same with the injected `tool_error` (first `relax` fails, retried) | `get_structure`, `relax` x2 |

When replayed, the mock backend re-executes the recorded tool calls for real (fresh sub-runs;
recorded `frame_id`/`run_id` values in later arguments and in the final answer's citations are
re-mapped to the fresh ones), or returns the stored results verbatim with `replay_results=True`.
Recorded numbers therefore come from a throw-away tiny model and are not physical values.

Real recorded traces of live runs (`b20mlip agent run --backend anthropic --record`) are written
to `runs/agent/traces/<task_id>.jsonl` (redacted; `runs/` is not committed).
