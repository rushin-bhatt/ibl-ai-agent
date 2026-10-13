# Evals

Questions that test whether a model can load and query the IBL datasets, run
against Claude, Mistral and Lightning AI models, graded with
[deepeval](https://deepeval.com). Every night CI runs them against one model per
provider.

## Layout

| Path | Role |
|---|---|
| `questions/*.json` | Questions with objective answers and the `checks` that grade them |
| `actors/` | One interface over three providers: `run(prompt) -> RunResult(reply, usage)` |
| `metrics.py` | Metrics built from a question's `checks` |
| `test_dataloading.py` | Data questions (`task: codegen`) x every model in the grid |
| `test_skill_selection.py` | Skill-routing questions (`task: skill_selection`) x every model |
| `summarize_results.py` | Per-run log (`usage.csv`) and the summary table (`summary.md`) |
| `models.default.json` | One model per provider: the nightly grid |
| `models.grid.json` | 15 models verified callable on the team keys (2026-10-05) |

## Actors

```
Actor                       run(prompt) -> RunResult(reply, usage)
├── CodeExecutingActor
│   └── ClaudeSDKActor      Claude Code agent (claude-agent-sdk)
└── ChatCompletionActor     chat endpoint + a run_python tool loop
    ├── LightningActor      OpenAI-compatible API
    └── MistralActor        Mistral SDK
```

Claude arrives as an agent with its own tools. Mistral and Lightning are chat
endpoints, so they get one tool, `run_python`, and a loop: call the model, run the
code it asks for, feed the output back, until it answers or hits `max_turns`.
Without that a data question is unanswerable for them and the comparison
meaningless. `"tools": false` in a grid entry removes the tool: the no-data
baseline.

Actors do not implement deepeval's `DeepEvalBaseLLM`: its `generate` returns a
bare string, with nowhere for token usage or a tool loop. `ActorJudge` adapts an
actor back where a metric requires a model object.

To add a provider: one `Actor` subclass and one entry in `PROVIDERS`
(`actors/__init__.py`). Chat providers implement `_client`, `_complete`,
`_assistant_message` and `_tool_message`; the loop is shared.

## Running

Unit tests mock every API call and need no keys, data or network:

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra dev --extra evals
UV_CACHE_DIR=.uv-cache uv run --no-sync pytest tests/evals
```

Live evals call model APIs, so they only run with `RUN_LIVE_EVALS=1`. They need
`bwm_ephys` (`scripts/download_datasets.py`) and keys in `tests/evals/.env.local`
(gitignored): `ANTHROPIC_API_KEY` (or a local `claude login`), `MISTRAL_API_KEY`,
`LITAI_API_KEY`.

```bash
RUN_LIVE_EVALS=1 UV_CACHE_DIR=.uv-cache uv run --no-sync pytest \
    tests/evals/test_dataloading.py tests/evals/test_skill_selection.py -n 3
UV_CACHE_DIR=.uv-cache uv run --no-sync python tests/evals/summarize_results.py
```

`EVAL_MODELS_FILE` picks another grid (e.g. `tests/evals/models.grid.json`) and
`EVAL_RESULTS_DIR` moves the output from `tests/evals/results/`.

## Reading results

`usage.csv` has one row per run: timestamp, commit, model, question, pass, tokens,
cost, turns and `stop_reason`. Failures that score the same are not the same:

- `answered`: the model gave an answer (right or wrong)
- `max_turns`: it kept working and ran out of turns
- a provider reason such as `malformed_function_call`: nothing usable came back
- an exception name such as `AuthenticationError`: the call itself failed

Only Claude reports cost; tokens are reported by all providers.

## Nightly CI

`.github/workflows/nightly-evals.yaml` runs at 06:00 UTC and on demand (Actions
tab, with a choice of grid file). It needs three repository secrets:
`ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, `LITAI_API_KEY`. `bwm_ephys` is extracted
without `spikes/` (553 MB instead of 6 GB), checked against its sha1 and cached.
A wrong answer does not fail the job; a broken harness does. Results are in the
run summary and the `eval-results-<run id>` artifact.

## Writing a question

Each entry in `questions/*.json` has an `id`, the `question`, a `task`, the
`packets` of AGENTS.md whose skill files go in the prompt, `requires_data`, and
`checks`. Prefer answers a plausible mistake gets wrong:
`bwm_visual_hemisphere.json` documents its traps in `_notes`.
