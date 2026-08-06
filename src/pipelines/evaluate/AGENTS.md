# Evaluate Pipeline

Read the repository root `AGENTS.md` first. This file adds only what is specific to `src/pipelines/evaluate/`.

## Scope

Takes a validation manifest plus either a checkpoint or supplied predictions, and produces COCO-style detection metrics together with the predictions that were actually scored. When Data also supplies an unlabeled COCO test manifest, it runs checkpoint inference and writes a competition submission CSV. It does not train, score test data, or modify an artifact another pipeline produced.

## Boundaries

You own `src/pipelines/evaluate/`. Do not edit another pipeline and never import one; `src/common` is the only shared code available to you. `config["inputs"]` is read-only.

## Interface

`run(config) -> dict` is the only public symbol. On success `artifacts` always carries exactly `run_id`, `metrics_uri`, and `predictions_uri`, plus `submission_uri` only when `test_manifest_uri` is present. On failure, `status="error"` and `artifacts={}` — a partial success is never reported as `ok`.

Settings come from `config["evaluate"]`, falling back to `config["inputs"]`. Set `predictions_input_uri` to skip validation inference and only score existing predictions; test inference still requires a checkpoint. In dummy mode with no `config["evaluate"]`, evaluation is skipped.

Manifests are accepted as JSONL or as a single COCO document. Both are validated for duplicate ids, annotation references, positive image sizes, and boxes that stay inside the image.

## Outputs

A metrics file and a predictions file under the configured output directory, plus optional `submissions/{run_id}/submission.csv`. They are written only after validation metrics and optional test predictions are complete, so a failure leaves no partial result.

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

- AP and mAP come from pycocotools `COCOeval`. numpy only builds the IoU threshold array and aggregates `evalImgs` — never write the matching logic yourself.
- The main metric is `mAP@[0.75:0.95]`, and that interval is also the default. `COCOeval` always receives the fixed ten points `0.50:0.95`; the main interval is a slice of them, so `mAP50`/`mAP75`/`mAP50_95` are always available. `evaluate.iou_thresholds` is not injectable — any other value is rejected with a `ConfigurationError` rather than silently ignored.
- `maxDets` is the configured value (4). Conditions with no ground truth stay `null`, never `0.0`. `COCOeval` writes to stdout, so its calls are wrapped in `redirect_stdout` — web parses the subprocess log. `summarize()` is never called; index `eval` directly.
- `precision50`/`recall50` are aggregate counts at `score >= 0.5` (previously the last point of the PR curve). The matching GUI label change is requested in `contracts/proposals/`, and this pipeline does not wait for it to merge.
- `analysis.py` is the diagnosis layer: threshold sweep, best F1, confusion matrix, per-image failures, false-positive causes. All of it reads `evalImgs`; extra IoU comes from `maskUtils.iou`. Three traps:
  - `gtMatches` reflects matching **before** the score filter, so judging misses with it erases any ground truth that only a low-score detection touched. Use the ids surviving detections actually claimed.
  - The confusion matrix needs the `useCats = 0` pass; the default pass matches within a class and cannot show a mix-up.
  - False positives bucket strongest-overlap-first: `duplicate`, `classification`, `localization`, `background`. `duplicate` fits none of the other three and would otherwise inflate one. `LOCALIZATION_IOU_FLOOR` keeps far-off boxes out of `localization`.
- Competition runs use the same validation IoU thresholds; test labels and test metrics are never accepted or produced.
- Reads and writes stay inside the repository. `..` and outside-repository absolute paths are errors.
- A configuration where both output filenames are identical is rejected before anything runs.
- If a later write fails, only files **this run created** are removed. A file that already existed and was overwritten on purpose is left alone, and S3 objects are never auto-deleted.
- Every failure is an `EvaluateError` subclass, returned as `status="error"`, never raised out of `run()`.
- The checkpoint payload you load is a contract with train. If it does not match, stop with a clear error instead of guessing.
