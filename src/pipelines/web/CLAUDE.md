# Web Pipeline

Read the repository root `CLAUDE.md` first. This file adds only what is specific to this directory.

## Scope

The training GUI: a FastAPI backend plus a React frontend to prepare data, start training, watch progress, and compare results. The only directory with a frontend.

**It is deliberately not a stage in `src/main_pipeline.py`.** It consumes what the pipelines produce, outside their run order.

## Boundaries

You own `src/pipelines/web/`. Never import data, train, evaluate, or registry. Two consequences carry the whole design:

- **Running a pipeline** happens only by subprocess, through `build_argv()` in `jobs/runner.py`, which builds `python -m src.main_pipeline --only <stage>` for the stages in `ALLOWED_STAGES`. Never `shell=True`, never let user input reach argv.
- **Reading an experiment record** happens only through `read_experiment_record()` in `src/common`. No `open`, no `Path`.

You cannot import train, so every value you both must agree on lives in `src/common/train_contract.py`: model and optimizer names, optimizer profiles, precision and schedule tables, the one combination, settings defaults. Read them from there and never re-type them here — they were copied once, watched by a test that parsed train's source, and a name drifted anyway. Train owns them; ask first.

`train_config.py` still mirrors train's **rules** — which values are refused, which key belongs to which selection — because the GUI must refuse before the GPU is busy. The MMDetection ones carry their own: `input_size` goes only to those architectures, and the one combination is enforced here so the wrong box is named on screen.

## Interface

`run(config) -> dict` reads one experiment record when `config["web"]["experiment_record_uri"]` is set, else returns the dummy result. The GUI is served separately, outside this contract.

## Outputs

All mutable state lives under `artifacts/web/`, which is gitignored. Never write outside the repository and never resolve a request to a path outside it — traversal returns 404. Never send an absolute path to the browser: `masking.py` strips home directories so the OS username cannot leak.

## Run and Test

```
python -m src.pipelines.web.server            # 127.0.0.1:8000
python -m pytest src/pipelines/web/tests -q
cd src/pipelines/web/frontend && npm run typecheck && npm test
```

`typecheck` is its own CI step and reads the test files too; `npx tsc` misses that. The server binds to localhost on purpose; exposing it is a decision, never a default.

## Local Rules

- `artifacts/web/` and port 8000 are shared: two servers from one clone collide, so only one at a time.
- The storage backend and team recording come from `PILL_` environment variables, so tests must not inherit them. An autouse fixture strips every `PILL_` name; a test needing one sets it with `monkeypatch.setenv`. Without that a test reads the real registry or demands a login token — green on CI, red only on a dev machine.
- Subprocess output must be drained continuously, or a large log deadlocks the reader thread.
- A cancelled job reports cancelled, not failed.
- The queue runs overnight, so a **failed** run still advances — one out-of-memory setting must not waste the night. A **cancelled** run and a restart leave it paused: whoever pressed stop did not ask for the next training.
- Probing for a GPU must not start a CUDA context.
- Evaluation uses the GPU when present: `cpu` took ~55 min on 2,942 images and hit `EVALUATE_TIMEOUT_SECONDS`, a GPU ~2 min. `resolve_device` rejects `cuda` where there is none.
- New screens need tests of their own; Python tests do not cover the frontend at all.
- The frontend follows `Training Console.html` at the root: dark, one amber accent. Colors and type live only in `frontend/src/design/tokens.ts`; never write a hex elsewhere.
- The rail lists work in order, never datasets: a dataset row there read as "this is what training uses". Training input changes only in dataset 준비 — never the `/records` pick, which chooses what to see.
- `lib/records.ts` merges two sources by `run_id`: registry experiments carry Kaggle, mAP and teammates; job records carry failures and this machine's doing.
- `GET /api/data/datasets` lists the folders under `PROCESSED_ROOT` so a person clicks a dataset instead of pasting its path. It opens no file: one listing call, then names and which artifacts exist, the crop bank included. The paste box stays, for a folder outside that root. `PROCESSED_ROOT` and the names are copied from data; drift empties the list rather than skewing it.
- The EDA sheet runs `--only data` with `data.eda` and renders `eda/report.json`; it draws the charts because data returns numbers only. `EdaRunner` subclasses `PreparationRunner` for the same one-at-a-time lifecycle but keeps its own state — sharing one swaps one panel's progress for the other's. When the report says its ruler failed on train, the size comparison stays hidden rather than drawn with a caveat.
- The 진단 sheet says **why** a run scored low, from its saved `metrics.json`. Send **pairs, not the 119×119 matrix**, and say how many are cut. `background` is 없음, not a class; a missing cause block stays missing, or 0 reads as real.
- Automatic evaluation stays off until a person picks a mode in the settings sheet, and only evaluates runs **this server just finished** — never succeeded records already on disk. Starting the server must never put someone else's GPU to work for hours.
- A run left unnamed is named from its settings: `retina-basic-e15-b4-lr6e3-s42-a7f3`. Identical settings and seed give an identical name — that is how a duplicate announces itself. No schedule and no warmup sends no `lr_scheduler` key.
- Resuming takes a **new** `run_id` reading as a lineage — `A`, `A.2`, `A.3`, skipping numbers this server knows — because reusing one makes train refuse to start. `epochs` is the whole plan; empty keeps the original target, so a **finished** run needs a bigger one. It continues from the published `last_checkpoint_uri`; one that stopped early cannot, its checkpoint carrying spent patience. That checkpoint must be **there**, so `GET .../resume` decides whether the button goes up — finished epochs only say one was likely saved.
- `PILL_WEB_STATE_WORKSPACE` mirrors job records and runtime configs into that name's own S3 slot, so a vanished Colab runtime stays resumable from the next. Off by default; logs stay local, a local file wins, and a restored record drops its `process_id`.
- Nobody deletes an interrupted run's `.<run_id>.partial`, this screen included: it holds the only copy of that training, so removing it is a person's call.
- The 앙상블 screen makes one submission from finished runs two ways: **모델** fuses their `test_predictions.json`; **임베딩** reranks one run's boxes only, re-inferring test — evaluate refuses a single fusion input. Its point is the **diagnosis**: a fusion's gain is unknown until a Kaggle submission, so a bad pick burns one. It warns, never blocks — alike pools gain nothing, a weak member drags the result toward the pool mean, mixed test sets and re-fused inputs are refused, past three the class count falls. `ensemble.py` carries each measurement.
- The 훑기 판 scores every archived epoch on a sample, ranks them by the settings sheet's three metrics (3:2:1 over columns normalised across candidates, or the widest-moving one alone), and re-evaluates the winner in full as `<run_id>-e12` — a separate run, so the original keeps its results. No metrics, no sweep. Refused while training or evaluation runs.
- The screen **picks** the crop embeddings that rerank (`embedding.py`), their only use; the rail's embedding 학습 sheet **trains** them. Training rides the **training queue** — a second door to the GPU starts two runs at once — and automatic evaluation walks past it: a detector evaluation cannot read an embedding checkpoint. One pick may not mix crop banks, or a model meets crops it never saw. Picking none leaves 모델 unchanged; 임베딩 refuses it.
