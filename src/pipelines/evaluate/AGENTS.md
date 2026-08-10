# Evaluate Pipeline

Read the repository root `AGENTS.md` first. This file adds only what is specific to `src/pipelines/evaluate/`.

## Scope

Takes a validation manifest plus either a checkpoint or supplied predictions, and produces COCO-style detection metrics with the predictions that were scored. Given an unlabeled COCO test manifest it also runs inference and writes a competition submission CSV. It never trains or scores test data.

## Boundaries

You own `src/pipelines/evaluate/`. Never edit or import another pipeline; `src/common` is the only shared code available. `config["inputs"]` is read-only.

## Interface

`run(config) -> dict` is the only public symbol. On success `artifacts` carries exactly `run_id`, `metrics_uri`, and `predictions_uri`, plus `submission_uri` when `test_manifest_uri` is present. On failure, `status="error"` and `artifacts={}` — a partial success is never reported as `ok`.

Settings come from `config["evaluate"]`, falling back to `config["inputs"]`. `predictions_input_uri` skips validation inference and only scores existing predictions; test inference still needs a checkpoint. In dummy mode with no `config["evaluate"]`, evaluation is skipped.

Manifests are JSONL or one COCO document, validated for duplicate ids, annotation references, positive image sizes, and boxes inside the image.

## Outputs

A metrics file and a predictions file under the configured output directory, plus `submissions/{run_id}/submission.csv` when asked. They are written only after validation metrics and test predictions are complete, so a failure leaves no partial result.

Two rules are easy to undo by accident:

- Stored boxes and scores are **not rounded**. Rounding would make a re-run over the saved predictions give different numbers.
- A metric that was not computed is `null`, never `0.0`, so "not measured" stays distinct from "zero".

## Run and Test

```
python -m src.main_pipeline --only evaluate
python -m pytest src/pipelines/evaluate/tests -q
```

Tests use contract-shaped fixtures with no upstream pipeline; inference is on CPU.

## Local Rules

- AP and mAP come from pycocotools `COCOeval`. numpy only builds the IoU threshold array and aggregates `evalImgs`; never write the matching logic yourself.
- The main metric is `mAP@[0.75:0.95]`. `COCOeval` always receives the fixed ten points `0.50:0.95` and the main interval is a slice, so `mAP50`/`mAP75`/`mAP50_95` are always available. `evaluate.iou_thresholds` is not injectable — any other value raises `ConfigurationError` rather than being ignored.
- `maxDets` is the configured value (4). `COCOeval` writes to stdout, so calls are wrapped in `redirect_stdout` — web parses the log. `summarize()` is never called; index `eval` directly.
- Progress goes to stderr as `evaluate.progress/1` JSON Lines; the emitter never raises and never writes stdout.
- `precision50`/`recall50` are aggregate counts at `score >= 0.5`, not the last PR-curve point. The GUI label change is requested in `contracts/proposals/`.
- `analysis.py` is the diagnosis layer: threshold sweep, best F1, confusion matrix, per-class summary, per-image failures, false-positive causes. All from `evalImgs`; extra IoU from `maskUtils.iou`. Four traps:
  - `gtMatches` reflects matching **before** the score filter, so judging misses with it erases ground truth a low-score detection touched. Use the ids surviving detections claimed.
  - The confusion matrix needs the `useCats = 0` pass; the default pass matches within a class and cannot show a mix-up.
  - False positives bucket strongest-overlap-first: `duplicate`, `classification`, `localization`, `background`. `duplicate` would otherwise inflate one of the others. `LOCALIZATION_IOU_FLOOR` keeps far-off boxes out of `localization`.
  - `per_class_summary` only re-sorts `per_class`: `truth_count` splits the groups, `ap = null` is never read as 0, and it is not IoU-keyed.
- Competition runs use the same validation IoU thresholds; test labels and metrics are never accepted or produced.
- `evaluate.submission_excluded_category_ids` drops those categories from the **submission CSV only**, inside `filter_predictions` *before* the per-image cap so the 4 slots go to scorable rows. Test call only. Default empty changes nothing.
- `evaluate.metrics_excluded_category_ids` drops them from **validation scoring only**, so local mAP averages the classes the competition scores. Ground-truth images stay — dropping one turns its predictions into false positives — and saved predictions keep every row. Default empty changes nothing.
- Identical output filenames are rejected before anything runs.
- If a later write fails, only files **this run created** are removed. A file overwritten on purpose is left alone; S3 objects are never auto-deleted.
- Every failure is an `EvaluateError` subclass returned as `status="error"`, never raised out of `run()`.
- The checkpoint payload is a contract with train. If it does not match, stop with a clear error instead of guessing.
