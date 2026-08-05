# Train Pipeline

Read the repository root `CLAUDE.md` first. This file adds only what is specific to `src/pipelines/train/`.

## Scope

Trains the Faster R-CNN baseline named by `ARCHITECTURE` in `model.py` — a MobileNetV3 320 FPN variant chosen so the pipeline stays runnable on CPU — and writes checkpoints plus a training history. It does not prepare data, compute metrics, or register experiments.

## Boundaries

You own `src/pipelines/train/`. Do not edit another pipeline and never import one; `src/common` is the only shared code available to you. `config["inputs"]` is read-only.

Web cannot import you, so `src/pipelines/web/train_config.py` keeps its own copy of your defaults and validation rules, and `src/pipelines/web/tests/test_web_train_contract.py` reads your source with `ast` and fails when the copies drift. **If you change a default, the architecture name, or the run-id rule and that test fails, the alarm is working.** Tell the web owner; do not edit their file.

## Interface

`run(config) -> dict` is the only public symbol. On success `artifacts` carries `run_id`, `best_checkpoint_uri`, `last_checkpoint_uri`, and `training_history_uri`. On failure, `status="error"` and `artifacts={}`.

It requires the four data artifacts in `config["inputs"]["data"]`.

## Outputs

Checkpoints and history go to the configured repository-relative output directory, or to the configured S3 prefix. Checkpoints are written into a temporary directory and moved into place, so an interrupted run never leaves a half-written checkpoint behind. Existing files are not overwritten.

## Run and Test

```
python -m src.main_pipeline --only train
python -m pytest src/pipelines/train/tests -q
```

Tests train a tiny model on CPU. They need no GPU and no AWS.

## Local Rules

- **The checkpoint payload is a contract with evaluate.** It must stay loadable by `src/pipelines/evaluate/predictor.py`. Renaming or dropping a key there is a `contracts/proposals/` proposal.
- `progress.py` emits `train.progress/1` JSON Lines on **stderr**. stdout belongs to the single JSON document `main_pipeline` prints — never write there. The emitter swallows every exception on purpose: progress output must never be able to fail a training run.
- Validation must not update BatchNorm running statistics. A test guards this.
- A seeded run must reproduce. If you introduce randomness, seed it.
