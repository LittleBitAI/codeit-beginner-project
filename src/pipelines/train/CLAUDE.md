# Train Pipeline

Read the repository root `CLAUDE.md` first. This file adds only what is specific to `src/pipelines/train/`.

## Scope

Trains a config-selected detector, or a crop embedding, and writes checkpoints plus a history. It does not prepare data, score it, or register runs.


## Boundaries

You own `src/pipelines/train/`. Do not edit another pipeline and never import one; `src/common` is the only shared code available. `config["inputs"]` is read-only.

Web cannot import you, so values you both agree on live in `src/common/train_contract.py`: model, optimizer, precision and schedule tables, the 8GB combination, defaults. **You own that file** — a name added there is offered by the GUI at once, so add it only once you accept it, and never re-type its values here.

## Interface

`run(config) -> dict` is the only public symbol. On success `artifacts` has `run_id`, `best_checkpoint_uri`, `last_checkpoint_uri`, and `training_history_uri`. On failure, `status="error"` and `artifacts={}`.

The detector requires the four data artifacts in `config["inputs"]["data"]`.

## Outputs

Checkpoints and history go to the configured output directory or S3 prefix.

During training, `.<run_id>.partial` sits beside the final output. Every `checkpoint_every` epochs each file is replaced from a temporary. Last is self-contained and replaced before best, so a failed best replacement stays resumable. Local renames the directory into place; S3 keeps it there until upload succeeds, so `output_dir` needs disk.

The S3 mirror `<prefix>/<run_id>/running/last_checkpoint.pt` is **one** self-contained object, so no pair ends up half updated. Its first conditional write claims the `run_id`; only the winner overwrites it.

From `archive_epochs_from` each write also drops an optimizer-less copy in `epochs/`, listed in `epoch_checkpoint_uris`: it scores an epoch, never resumes one. Absent, none are kept (018).

The cache fills **before the first batch**, `PREFETCH_WORKERS` at a time: one per batch leaves the GPU waiting out an S3 round trip per image, a whole first epoch. One that cannot be fetched is left to the training loop rather than stopping the run, and that isolation catches **every** exception on purpose — `src/common` converts only some storage failures, so a narrower clause ends a night. `image_cache_progress` counts images **ready**, never attempted (016).

`artifacts/train-image-cache/` keeps **one** dataset — Colab has no room for two. Starting on a different one trashes every other unleased namespace before the first fetch. A run **holds its lease file locked** for its lifetime, so liveness is asked of the OS, not a timestamp: a killed run stops protecting tens of gigabytes at once.

Published files are never overwritten, and a run stops before its first batch when its `run_id` already has a working directory, an S3 `running/` checkpoint, or a finished result. Output paths stay inside the repository, or a checkpoint lands outside it and its absolute path becomes the artifact URI. On a new Colab runtime's empty disk only the bucket knows an interrupted run exists.

## Configurable Training

- Architectures come from the contract; never accept arbitrary builder names. Two are MMDetection, via `mmdetection_adapter.py`.
- The MMDetection pair only fits 8GB at `device="cuda"`, `precision="amp"`, `optimizer="AdamW"`, `batch_size=1`; anything else is refused before the first batch, not partway through the night. `input_size` is theirs alone, refused with a torchvision architecture, not ignored.
- Their checkpoints carry `backend` and `model_config` for evaluate; no `backend` still reads as torchvision, which keeps older ones loadable.
- `num_workers` defaults to a few on CUDA, `0` on CPU and wherever `WORKERS_ARE_SPAWNED`: a spawned worker gets the dataset by pickle, which its S3 client cannot do. A **forked** worker inherits that client, which boto3 cannot share; `give_worker_its_own_storage` reconnects it.
- A missing optimizer means legacy SGD; new callers send AdamW. Reject optimizer- or schedule-specific settings the selection does not use; never ignore them.
- Augmentation defaults to `none`; `pill_basic` and `pill_geometric` are train-split only and geometric transforms must move the boxes with them. Rotate by quarter turns only: other angles loosen an axis-aligned box and cost the IoU 0.75-0.95 score. Crops drop pills they cut.
- `precision` is `fp32`, `amp`, `fp16`, or `bf16`; all but `fp32` need CUDA. `amp` uses fp16 plus a scaler for MMDetection, whose custom CUDA ops do not dispatch bf16; others take native bf16 else fp16 and a scaler. Explicit `bf16` is refused, not emulated.
- `lr_scheduler` is absent by default: a constant learning rate. Its factor is recomputed on every **optimizer update**, so warmup counts updates, not microbatches; only `step` decays per epoch. Above 1 `gradient_accumulation_steps` those differ, leaving `linear` and `cosine` short of their floor.
- `gradient_accumulation_steps` groups that many microbatches into one update — how a GPU too small for a larger `batch_size` still gets one. Three parts fail silently: clearing gradients every microbatch throws away what was gathered, dividing by the configured size rather than what a group held misweights a short final group, and dropping that group drops its images. Predating the key reads as 1.
- Checkpoints record normalized settings under `training_config`. Keep it JSON-safe and credential-free.
- `resume_from` continues a run from its self-contained `last_checkpoint.pt`, best epoch included. `epochs` counts the whole run, not the part that remains.
- Every reason a resume cannot work is checked before the first batch: missing `resume_state`, a gapped history, a different architecture, class map, optimizer, schedule, or `num_workers`, missing AMP scaler or schedule state, `epochs` no larger than the resumed epoch, spent patience. Worker count counts because augmentation draws from a per-worker RNG.

## Embedding Task

`train.task` is `detector` (default) or `embedding`, which reads a crop bank, not manifests, and takes `EMBEDDING_SETTING_KEYS` — one list would offer a backbone on the detector screen. Detector-only fields are refused, not dropped, and the same four artifacts come out. The checkpoint carries `backbone` and `category_ids`; consumers drop the head and compare distances to reference crops, so an untrained class is matched. Quarter-turn and flip augmentation is the point.

The guards above are its too. Three are its own. `best` is judged every epoch, not on the `checkpoint_every` beat, or a worse model ships under that name. `class_map_uri` is required, so it is **read** — an input nobody opens cannot catch a bank paired with another dataset. Bank entries are bounded by path, not string prefix, links and device files refused, the listing's own paths checked too: a safe tar can still name a file outside it.

## Run and Test

```
python -m src.main_pipeline --only train
python -m pytest src/pipelines/train/tests -q
```

Tests train a tiny model on CPU: no GPU or AWS.

## Local Rules

- **The checkpoint payload is a contract with evaluate.** It must stay loadable by `evaluate/predictor.py`. Renaming or dropping a key there is a `contracts/proposals/` proposal.
- `progress.py` emits `train.progress/1` JSON Lines on **stderr**. stdout belongs to the single JSON document `main_pipeline` prints. The emitter swallows every exception: progress must never fail a run.
- Validation must not update BatchNorm running statistics. A test guards this.
- A seeded run must reproduce. Seed randomness you add and pin the deterministic mode, as both tasks do.
- **A resumed run must reproduce an uninterrupted one.** Tests compare four epochs against two plus two, with a control that breaks the random-state restore. CPU-only: deterministic algorithms use `warn_only=True`, so CUDA kernels may differ.
- Nobody deletes an orphaned `.<run_id>.partial`: it holds the only copy of that run.
