# Evaluate Pipeline

Read the repository root `AGENTS.md` first. This file adds only what is specific to `src/pipelines/evaluate/`.

## Scope

Takes a validation manifest plus either a checkpoint or supplied predictions, and produces COCO-style detection metrics together with the predictions that were actually scored. It does not train, and it never modifies an artifact another pipeline produced.

## Boundaries

You own `src/pipelines/evaluate/`. Do not edit another pipeline and never import one; `src/common` is the only shared code available to you. `config["inputs"]` is read-only.

## Interface

`run(config) -> dict` is the only public symbol. On success `artifacts` carries exactly `run_id`, `metrics_uri`, and `predictions_uri`. On failure, `status="error"` and `artifacts={}` — a partial success is never reported as `ok`.

Settings come from `config["evaluate"]`, falling back to `config["inputs"]`. Set `predictions_input_uri` to skip inference and only score existing predictions. In dummy mode with no `config["evaluate"]`, evaluation is skipped.

Manifests are accepted as JSONL or as a single COCO document. Both are validated for duplicate ids, annotation references, positive image sizes, and boxes that stay inside the image.

## Outputs

A metrics file and a predictions file under the configured output directory. Both are written only after every metric has been computed, so a failure leaves no partial result.

Two rules exist for reasons that are easy to undo by accident:

- Stored boxes and scores are **not rounded**. Rounding would make a re-run over the saved predictions produce different numbers.
- A metric that was not computed is `null`, never `0.0`, so "not measured" stays distinguishable from "measured as zero".

## Run and Test

```
python -m src.main_pipeline --only evaluate
python -m pytest src/pipelines/evaluate/tests -q
```

Tests run from contract-shaped fixtures with no upstream pipeline, and checkpoint inference runs on CPU.

## Local Rules

- Metrics are implemented on numpy, deliberately without pycocotools: 101-point interpolated AP, score-descending greedy matching, and classes with no ground truth excluded from the mAP mean.
- Reads and writes stay inside the repository. `..` and outside-repository absolute paths are errors.
- A configuration where both output filenames are identical is rejected before anything runs.
- If a later write fails, only files **this run created** are removed. A file that already existed and was overwritten on purpose is left alone, and S3 objects are never auto-deleted.
- Every failure is an `EvaluateError` subclass, returned as `status="error"`, never raised out of `run()`.
- The checkpoint payload you load is a contract with train. If it does not match, stop with a clear error instead of guessing.
