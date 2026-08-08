# Train Pipeline

Read the repository root `AGENTS.md` first. This file adds only what is specific to `src/pipelines/train/`.

## Scope

Trains a config-selected torchvision detector and writes checkpoints plus a training history. The CPU-friendly MobileNetV3 320 FPN Faster R-CNN remains the legacy architecture when the setting is absent. It does not prepare data, compute metrics, or register experiments.

## Boundaries

You own `src/pipelines/train/`. Do not edit another pipeline and never import one; `src/common` is the only shared code available to you. `config["inputs"]` is read-only.

Web cannot import you, so `src/pipelines/web/train_config.py` keeps its own copy of your defaults and validation rules, and `src/pipelines/web/tests/test_web_train_contract.py` reads your source with `ast` and fails when the copies drift. **If you change a default, the architecture name, or the run-id rule and that test fails, the alarm is working.** Tell the web owner; do not edit their file.

## Interface

`run(config) -> dict` is the only public symbol. On success `artifacts` carries `run_id`, `best_checkpoint_uri`, `last_checkpoint_uri`, and `training_history_uri`. On failure, `status="error"` and `artifacts={}`.

It requires the four data artifacts in `config["inputs"]["data"]`.

## Outputs

Checkpoints and history go to the configured repository-relative output directory, or to the configured S3 prefix.

While training runs, both checkpoints live in a working directory named `.<run_id>.partial` beside the final one, rewritten every `checkpoint_every` epochs through a temporary file so a crash mid-write keeps the previous copy. A successful local run renames that directory into place; an S3 run uploads from memory, mirrors the working checkpoints under `<prefix>/<run_id>/running/`, and removes the directory afterwards. On an S3 backend the working directory is still local, so `output_dir` needs disk even there.

Published files are never overwritten. A run also stops before its first batch when the same `run_id` already has a non-empty working directory, an S3 `running/` checkpoint, or a finished result. The S3 checks carry the weight: a new Colab runtime has an empty disk, so only the bucket still knows that an interrupted run is there, and overwriting it would destroy the one copy this whole feature exists to keep.

## Configurable Training

- Supported architectures are declared in `model.py`; never accept arbitrary import or builder names.
- Supported optimizers are AdamW, SGD, and Adam. A missing optimizer means legacy SGD, while new callers should explicitly send AdamW.
- Reject optimizer-specific settings that the selected optimizer does not use; never ignore them silently.
- Augmentation defaults to `none`. `pill_basic` applies only to the train split and must update bounding boxes with geometric transforms.
- Checkpoints record the normalized model, optimizer, augmentation, and seed settings under `training_config`. Keep that metadata JSON-safe and free of storage credentials.
- `resume_from` continues an interrupted run from its `last_checkpoint.pt`, which carries `resume_state`. It also reads the `best_checkpoint.pt` written beside it, so the resumed run can still publish a best epoch from before the interruption. `epochs` counts the whole run, not the part that remains.
- Every reason a resume cannot work is checked before the first batch: a missing `resume_state`, a history with gaps, a different architecture or class map, `epochs` no larger than the resumed epoch, and patience already used up.

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
- **A resumed run must reproduce an uninterrupted one.** One test compares four epochs against two plus two; a second one breaks the random-state restore on purpose and requires the two to diverge, so the first test cannot pass by accident. The guarantee is CPU-only: deterministic algorithms run with `warn_only=True`, so CUDA kernels may still differ.
- Nobody deletes an orphaned `.<run_id>.partial`. It holds the only copy of an interrupted run, so removing it is a person's decision.
