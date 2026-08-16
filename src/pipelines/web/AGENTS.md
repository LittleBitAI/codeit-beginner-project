# Web Pipeline

Read the repository root `AGENTS.md` first. This file adds only what is specific to `src/pipelines/web/`.

## Scope

The training GUI: a FastAPI backend plus a React frontend to prepare data, start training, watch progress, and compare results. The only directory with a frontend build.

**It is deliberately not a stage in `src/main_pipeline.py`.** It consumes what the pipelines produce; it is not part of their run order.

## Boundaries

You own `src/pipelines/web/`. Never import data, train, evaluate, or registry. Two consequences carry the whole design:

- **Running a pipeline** happens only by subprocess, through `build_argv()` in `jobs/runner.py`, which builds `python -m src.main_pipeline --only <stage>` for the stages in `ALLOWED_STAGES`. That is the only way. Never `shell=True`, never let user input reach argv.
- **Reading an experiment record** happens only through `read_experiment_record()` in `src/common`. No `open`, no `Path`.

You cannot import train, so every value you both must agree on lives in `src/common/train_contract.py`: model and optimizer names, optimizer profiles, precision and schedule tables, the 8GB combination, settings defaults. Read them from there and never re-type them here — they were copied once, watched by a test that parsed train's source, and a name drifted anyway. Train owns them; ask before changing one.

`train_config.py` still mirrors train's **rules** — which values are refused, which key belongs to which selection — because the GUI must refuse before the GPU is busy; `tests/test_web_train_contract.py` checks the shape of what web sends, not train's literals. The MMDetection pair carries its own: `input_size` is offered and sent only for those architectures, and the 8GB combination is enforced here so the wrong box is named on screen.

## Interface

`run(config) -> dict` reads one experiment record when `config["web"]["experiment_record_uri"]` is set, else returns the dummy result. The GUI is served separately and is not part of this contract.

## Outputs

All mutable state lives under `artifacts/web/`, which is gitignored. Never write outside the repository and never resolve a request to a path outside it — traversal attempts return 404. Never send an absolute path to the browser: `masking.py` strips home directories so the OS username cannot leak into a response.

## Run and Test

```
python -m src.pipelines.web.server            # 127.0.0.1:8000
python -m pytest src/pipelines/web/tests -q
cd src/pipelines/web/frontend && npm test
```

The server binds to localhost on purpose; exposing it is a deliberate decision, never a default.

## Local Rules

- `artifacts/web/` and port 8000 are shared runtime state: two people running the server from one clone collide, so only one at a time. Tests point the repository root at a temporary directory.
- The storage backend and team recording come from `PILL_` environment variables, so tests must not inherit them. An autouse fixture strips every `PILL_` name; a test needing one sets it with `monkeypatch.setenv`. Without that a test reads the team's real registry or demands a login token — green on CI, red only on a dev machine.
- Subprocess output must be drained continuously, or a large log deadlocks the reader thread; a test guards this.
- Cancelling a job must report cancelled, not failed.
- The queue runs overnight, so a **failed** run still advances — one out-of-memory setting must not waste the night. A **cancelled** run and a server restart leave it paused: whoever pressed stop did not ask for the next training. Tests guard both.
- Probing for a GPU must not initialise a CUDA context.
- Evaluation uses the GPU when present: `cpu` took ~55 minutes on 2,942 images and hit `EVALUATE_TIMEOUT_SECONDS`, a GPU ~2 minutes. `resolve_device` rejects `cuda` where there is none.
- The frontend is the least-tested part of the repository. New screens need tests; the Python tests do not cover them.
- The frontend follows `Training Console.html` at the root: dark, one amber accent. Colors and type live only in `frontend/src/design/tokens.ts`; never write a hex elsewhere. `design_handoff_pill_detect_platform/README.md` mirrors that file.
- The rail lists work in order, never datasets: a dataset row there read as "this is what training uses". Training input changes only in dataset 준비 — never the `/records` pick, which chooses what to look at.
- `lib/records.ts` merges two sources by `run_id`: registry experiments carry Kaggle, mAP and teammates; job records carry failures and this machine's actions. One alone loses one of the two.
- `GET /api/data/datasets` lists the folders under `PROCESSED_ROOT` so a person clicks a dataset instead of pasting its path. It opens no file: one listing call, then names and which artifacts exist. The paste box stays, for a folder outside that root. `PROCESSED_ROOT` and the artifact file names are copied from data; if they drift the list is empty rather than wrong.
- The EDA sheet runs `--only data` with `data.eda` and renders `eda/report.json`. It draws the charts because the data pipeline returns numbers only — no plotting dependency. `EdaRunner` subclasses `PreparationRunner` for the same one-at-a-time lifecycle but keeps its own state; sharing one would swap one panel's progress for the other's. When the report says its own ruler failed on train, the size comparison stays hidden rather than drawn with a caveat.
- Automatic evaluation stays off until a person picks a mode in the settings sheet, and only evaluates runs **this server just finished** — never succeeded records already on disk. Starting the server must never put someone else's GPU to work for hours. `settings.py` returns `None` for an unset mode and the worker does nothing on `None`.
- A run left unnamed is named from its settings: `retina-basic-e15-b4-lr6e3-s42-a7f3`. Identical settings and seed give an identical name, which is how a duplicate announces itself. Resume keeps the timestamp name, because a deterministic one would equal the interrupted run's and train would refuse to start. No schedule and no warmup sends no `lr_scheduler` key.
- Resuming always takes a **new** `run_id`; reusing the old one makes train refuse to start and mixes two runs into one name. `epochs` is the whole plan, so leaving it empty keeps the original target. However the run ended, the checkpoint must be **there**, or a knowable failure moves past creating a run and restarting the queue. `GET .../resume` decides whether the button goes up, because finished epochs only say one was *likely* saved. Failing to read the store differs from having nothing: that answer keeps the button and says why.
- `PILL_WEB_STATE_WORKSPACE` mirrors job records and runtime configs into that name's own S3 slot, so a vanished Colab runtime stays resumable from the next one. Off by default; logs stay local, a local file wins, and a restored record drops its `process_id`.
- Nobody deletes an interrupted run's `.<run_id>.partial` directory, this screen included. It holds the only copy of that training, so removing it is a person's decision.
- The 앙상블 screen fuses runs' `test_predictions.json` into one submission. Its point is the **diagnosis**: a fusion's gain is unknown until a Kaggle submission, so a bad pick burns one. It warns, never blocks — alike pools gain nothing, a weak member drags the result toward the pool mean, mixed test sets are refused, a fused input cannot be re-fused, past three the class count falls. Agreement is cached in web state. `ensemble.py` carries the measurements behind each rule.
- The 훑기 판 scores every archived epoch on a sample, ranks them by the settings sheet's three metrics (3:2:1 over columns normalised across candidates, or the widest-moving one decides alone), and re-evaluates the winner in full as `<run_id>-e12` — a separate run, so the original keeps its results. No metrics chosen, no sweep: a default ranking would erase the question. Refused while a training or evaluation runs.
