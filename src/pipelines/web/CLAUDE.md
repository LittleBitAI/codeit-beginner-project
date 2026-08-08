# Web Pipeline

Read the repository root `CLAUDE.md` first. This file adds only what is specific to `src/pipelines/web/`.

## Scope

The training GUI: a FastAPI backend plus a React frontend that let a person prepare data, start training, watch progress, and compare results. This is the largest directory in the repository and the only one with its own frontend build.

**It is deliberately not a stage in `src/main_pipeline.py`.** It consumes what the pipelines produce; it is not part of their run order.

## Boundaries

You own `src/pipelines/web/`. Never import data, train, evaluate, or registry. Two consequences carry the whole design:

- **Running a pipeline** happens only by subprocess, through `build_argv()` in `jobs/runner.py`, which builds `python -m src.main_pipeline --only <stage>` for the stages in `ALLOWED_STAGES`. That is the only permitted way. Never `shell=True`, and never let user input reach argv.
- **Reading an experiment record** happens only through `read_experiment_record()` in `src/common`. No `open`, no `Path`, no creating storage yourself, no importing registry.

Because you cannot import train, `train_config.py` keeps a copy of train's defaults and validation rules, and `tests/test_web_train_contract.py` parses train's source with `ast` and fails when the copies drift. It exists because an architecture name once drifted and the GUI showed the wrong one. When it fails, fix the copy here — never edit train.

## Interface

`run(config) -> dict` is the only public symbol. It reads one experiment record when `config["web"]["experiment_record_uri"]` is set, and otherwise returns the dummy result. The GUI itself is served separately and is not part of this contract.

## Outputs

All mutable state lives under `artifacts/web/`, which is gitignored. Never write outside the repository, and never send an absolute path to the browser — `masking.py` also strips home directories so the operating-system username cannot leak into a response.

## Run and Test

```
python -m src.pipelines.web.server            # 127.0.0.1:8000
python -m pytest src/pipelines/web/tests -q
cd src/pipelines/web/frontend && npm test
```

The server binds to localhost on purpose. Exposing it to other hosts is a deliberate decision, never a default.

## Local Rules

- `artifacts/web/` and port 8000 are shared runtime state. Two people running the server from one clone will collide, so only one at a time. Tests avoid this by pointing the repository root at a temporary directory.
- This pipeline picks its storage backend and whether to record to the team from `PILL_` environment variables, so tests must not inherit them. An autouse fixture strips every `PILL_` name; a test that needs one sets it with `monkeypatch.setenv`. Without that, `PILL_STORAGE_S3_BUCKET` makes the experiment list read the team's real registry and `PILL_TEAM_SYNC_ENABLED` makes starting a job demand a token — green on CI, red only on a developer machine.
- Subprocess output must be drained continuously, or a large log deadlocks the reader thread. A test guards this.
- Cancelling a job must report cancelled, not failed.
- A request must never resolve to a path outside the repository; traversal attempts return 404.
- Probing for a GPU must not initialise a CUDA context.
- The frontend is the least-tested part of the repository. New screens need tests; do not assume the Python tests cover them.
