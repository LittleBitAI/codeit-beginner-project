# Web Pipeline

Read the repository root `AGENTS.md` first. This file adds only what is specific to `src/pipelines/web/`.

## Scope

The training GUI: a FastAPI backend plus a React frontend that let a person prepare data, start training, watch progress, and compare results. This is the largest directory in the repository and the only one with a frontend build.

**It is deliberately not a stage in `src/main_pipeline.py`.** It consumes what the pipelines produce; it is not part of their run order.

## Boundaries

You own `src/pipelines/web/`. Never import data, train, evaluate, or registry. Two consequences carry the whole design:

- **Running a pipeline** happens only by subprocess, through `build_argv()` in `jobs/runner.py`, which builds `python -m src.main_pipeline --only <stage>` for the stages in `ALLOWED_STAGES`. That is the only way. Never `shell=True`, never let user input reach argv.
- **Reading an experiment record** happens only through `read_experiment_record()` in `src/common`. No `open`, no `Path`, no importing registry.

Because you cannot import train, `train_config.py` copies train's defaults and validation rules, and `tests/test_web_train_contract.py` parses train's source with `ast` and fails when they drift. It exists because an architecture name once drifted and the GUI showed the wrong one. When it fails, fix the copy here — never edit train.

The two contract checks pull in opposite directions, which fixes the order of any joint change. The architecture list is compared for **equality**, so web cannot list a name train has not opened yet. Numeric defaults are walked from **train's** side, so a default train adds before web mirrors it breaks web immediately — web has to go first there. The MMDetection pair carries its own rules: `input_size` is offered and sent only for those architectures, and the 8GB combination is enforced here so the wrong box is named on screen rather than after the GPU is already busy.

## Interface

`run(config) -> dict` is the only public symbol. It reads one experiment record when `config["web"]["experiment_record_uri"]` is set, else returns the dummy result. The GUI is served separately and is not part of this contract.

## Outputs

All mutable state lives under `artifacts/web/`, which is gitignored. Never write outside the repository, and never send an absolute path to the browser — `masking.py` strips home directories so the OS username cannot leak into a response.

## Run and Test

```
python -m src.pipelines.web.server            # 127.0.0.1:8000
python -m pytest src/pipelines/web/tests -q
cd src/pipelines/web/frontend && npm test
```

The server binds to localhost on purpose; exposing it is a deliberate decision, never a default.

## Local Rules

- `artifacts/web/` and port 8000 are shared runtime state. Two people running the server from one clone collide, so only one at a time. Tests avoid this by pointing the repository root at a temporary directory.
- This pipeline picks its storage backend and team recording from `PILL_` environment variables, so tests must not inherit them. An autouse fixture strips every `PILL_` name; a test that needs one sets it with `monkeypatch.setenv`. Without that, `PILL_STORAGE_S3_BUCKET` makes the experiment list read the team's real registry and `PILL_TEAM_SYNC_ENABLED` makes starting a job demand a token — green on CI, red only on a dev machine.
- Subprocess output must be drained continuously, or a large log deadlocks the reader thread. A test guards this.
- Cancelling a job must report cancelled, not failed.
- The training queue is meant to be left running overnight, so two rules follow. A **failed** run still advances — one out-of-memory setting must not waste the night. A **cancelled** run, and a server restart, leave it paused, because a person who pressed stop did not ask for the next training. Tests guard both.
- A request must never resolve to a path outside the repository; traversal attempts return 404.
- Probing for a GPU must not initialise a CUDA context.
- Evaluation uses the GPU when one is present. `evaluate` defaults to `cpu`, which took about 55 minutes on 2,942 images and hit `EVALUATE_TIMEOUT_SECONDS`; a GPU takes about 2 minutes. `resolve_device` rejects `cuda` on a machine without one, so the failure arrives before the inference.
- The frontend is the least-tested part of the repository. New screens need tests; the Python tests do not cover them.
- The frontend follows `Training Console.html` at the repository root: dark, one amber accent, three routes (`/` records, `/canvas` compare, `/monitor` live) plus right-hand sheets for 새 실험, 설정, dataset 준비. Colors and type live only in `frontend/src/design/tokens.ts`; never write a hex anywhere else. The token table in `design_handoff_pill_detect_platform/README.md` mirrors that file.
- The record list merges two sources by `run_id` in `lib/records.ts`: registry experiments carry Kaggle and mAP and include teammates, job records carry failures and the actions only this machine can take. Showing one alone loses either the team's runs or this machine's failures.
- The EDA sheet (rail, under dataset 준비) runs `--only data` with `data.eda`, the same subprocess route as preparation, and renders `eda/report.json`. It draws the charts because the data pipeline returns numbers only — no plotting dependency in this repository. `EdaRunner` subclasses `PreparationRunner` for the same one-at-a-time lifecycle but keeps its own state; sharing one would swap one panel's progress for the other's. When the report says its own ruler failed on train, the size comparison must stay hidden rather than be drawn with a caveat.
- Automatic evaluation stays off until a person picks a mode in the settings sheet, and it only ever evaluates runs **this server just finished** — never the succeeded records already on disk. Both rules exist so that starting the server can never put someone else's GPU to work for hours. `settings.py` returns `None` for an unset mode and the manager's worker does nothing on `None`.
- A run left unnamed is named from its settings: `retina-basic-e15-b4-lr6e3-s42-a7f3`. Identical settings and seed give an identical name, which is how a duplicate experiment announces itself. Resume keeps the timestamp name instead, because a deterministic one would equal the interrupted run's and train would refuse to start. No schedule and no warmup sends no `lr_scheduler` key, so those names stay as before.
- Resuming an interrupted run always takes a **new** `run_id`. Reusing the old one makes train refuse to start — that run's working directory is still there — and mixes two runs into one name. `epochs` is the whole plan, not what is left, so leaving it empty carries the original target over.
- `PILL_WEB_STATE_WORKSPACE` mirrors job records and runtime configs into that name's own S3 slot, so a vanished Colab runtime stays resumable from the next one. Off by default; logs stay local, a local file wins, and a restored record drops its `process_id`.
- Nobody deletes an interrupted run's `.<run_id>.partial` directory, this screen included. It holds the only copy of that training, so removing it is a person's decision.
