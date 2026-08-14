# Train Pipeline

Read the repository root `AGENTS.md` first. This file adds only what is specific to `src/pipelines/train/`.

## Scope

Trains a config-selected torchvision detector and writes checkpoints plus a training history. The CPU-friendly MobileNetV3 320 FPN Faster R-CNN is the legacy default. It does not prepare data, score it, or register runs.

## Boundaries

You own `src/pipelines/train/`. Do not edit another pipeline and never import one; `src/common` is the only shared code available to you. `config["inputs"]` is read-only.

Web cannot import you, so `src/pipelines/web/train_config.py` copies your defaults and validation rules, and `test_web_train_contract.py` reads your source with `ast` and fails when they drift. **That failure is the alarm working.** Tell the web owner; do not edit their file.

## Interface

`run(config) -> dict` is the only public symbol. On success `artifacts` has `run_id`, `best_checkpoint_uri`, `last_checkpoint_uri`, and `training_history_uri`. On failure, `status="error"` and `artifacts={}`.

It requires the four data artifacts in `config["inputs"]["data"]`.

## Outputs

Checkpoints and history go to the configured repository-relative output directory or S3 prefix.

During training, `.<run_id>.partial` sits beside the final output. Every `checkpoint_every` epochs each file is replaced from a temporary file. Last is self-contained and replaced before best, so a failed best replacement stays resumable. Local renames the directory into place; S3 keeps it there until upload succeeds, so `output_dir` needs disk.

The S3 mirror `<prefix>/<run_id>/running/last_checkpoint.pt` is **one** self-contained object, so no pair can end up half updated. Its first conditional write claims the `run_id`; only the winner overwrites it.

The cache fills **before the first batch**, `PREFETCH_WORKERS` images at a time, because fetching one per batch leaves the GPU waiting out one S3 round trip per image and that wait is the whole first epoch. Images already there are skipped, so an interrupted run downloads only the rest. An image that cannot be fetched is left to the training loop rather than stopping the run, and that isolation catches **every** exception on purpose: `src/common` converts only some storage failures, so a narrower clause lets one image end a night of training. Its `image_cache_progress` line reports images **ready**, never images attempted (proposal 016).

`artifacts/train-image-cache/` keeps **one** dataset, because one namespace holds a whole one and Colab does not have room for two. Starting a run on a different dataset trashes every other unleased namespace before the first image is fetched; coming back to it refetches. A run **holds its lease file locked** for its whole lifetime, so liveness is asked of the OS rather than of a timestamp: a run that is quiet for hours before its first image keeps its cache, and a killed run stops protecting tens of gigabytes the moment it dies rather than at a TTL.

Published files are never overwritten, and a run stops before its first batch when its `run_id` already has a non-empty working directory, an S3 `running/` checkpoint, or a finished result. On the empty disk of a new Colab runtime only the bucket knows an interrupted run is there, so the S3 checks carry the weight.

## Configurable Training

- Supported architectures are declared in `model.py`; never accept arbitrary builder names. Two of them are MMDetection models built through `mmdetection_adapter.py`, and `model.py` unpacks that list rather than repeating the names.
- The MMDetection pair only fits 8GB at `device="cuda"`, `precision="amp"`, `optimizer="AdamW"`, `batch_size=1`, so anything else is refused before the first batch instead of dying partway through the night. `input_size` belongs to them alone and is refused with a torchvision architecture rather than accepted and ignored.
- Their checkpoints carry `backend` and `model_config` for evaluate; a checkpoint without `backend` still reads as torchvision, which is what keeps older ones loadable.
- `num_workers` defaults to a few workers on CUDA and to `0` on CPU, where nothing waits on the decode, and on Windows, where a worker is spawned by pickling the dataset and its S3 client does not pickle. An explicit value is used as given, everywhere. Web copied the old fixed `0` (proposal 015).
- Supported optimizers are AdamW, SGD, and Adam. A missing optimizer means legacy SGD; new callers send AdamW.
- Reject optimizer- or schedule-specific settings the selection does not use; never ignore them silently.
- Augmentation defaults to `none`. `pill_basic` applies only to the train split and must update bounding boxes with geometric transforms.
- `precision` is `fp32`, `amp`, `fp16`, or `bf16`; all but `fp32` need CUDA. `amp` uses fp16 plus a scaler for MMDetection because its custom CUDA ops do not dispatch bf16; other architectures take native bf16 else fp16 plus a scaler. Explicit `bf16` is refused rather than emulated.
- `lr_scheduler` is absent by default: the constant learning rate of before. Its factor is recomputed on every **optimizer update**, so warmup counts updates rather than microbatches; only `step` decays per epoch. With `gradient_accumulation_steps` above 1 those differ, and measuring the schedule in microbatches leaves `linear` and `cosine` short of the floor they were configured to reach.
- `gradient_accumulation_steps` groups that many microbatches into one update, which is how a GPU too small for a larger `batch_size` still gets one. Three parts of it fail silently rather than loudly: clearing gradients every microbatch throws away what was gathered, dividing by the configured size rather than the size a group actually held misweights a short final group, and dropping that final group drops those images from the epoch. It is recorded in `training_config` and compared before a resume, because a different value moves the optimizer and the schedule differently — a checkpoint written before the key existed reads as 1.
- Checkpoints record the normalized model, optimizer, augmentation, schedule, and seed settings under `training_config`. Keep it JSON-safe and free of storage credentials.
- `resume_from` continues an interrupted run from its self-contained `last_checkpoint.pt`, including the best epoch from before interruption. `epochs` counts the whole run, not the part that remains.
- Every reason a resume cannot work is checked before the first batch: missing `resume_state`, a history with gaps, a different architecture, class map, optimizer, or schedule, missing AMP scaler or schedule state, `epochs` no larger than the resumed epoch, and spent patience.

## Run and Test

```
python -m src.main_pipeline --only train
python -m pytest src/pipelines/train/tests -q
```

Tests train a tiny model on CPU: no GPU, no AWS.

## Local Rules

- **The checkpoint payload is a contract with evaluate.** It must stay loadable by `src/pipelines/evaluate/predictor.py`. Renaming or dropping a key there is a `contracts/proposals/` proposal.
- `progress.py` emits `train.progress/1` JSON Lines on **stderr**. stdout belongs to the single JSON document `main_pipeline` prints — never write there. The emitter swallows every exception: progress must never fail a run.
- Validation must not update BatchNorm running statistics. A test guards this.
- A seeded run must reproduce. If you introduce randomness, seed it.
- **A resumed run must reproduce an uninterrupted one.** Tests compare four epochs against two plus two, with a control that breaks the random-state restore. CPU-only: deterministic algorithms use `warn_only=True`, so CUDA kernels may differ.
- Nobody deletes an orphaned `.<run_id>.partial`. It holds the only copy of an interrupted run, so removing it is a person's decision.
