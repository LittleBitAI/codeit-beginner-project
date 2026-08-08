# Evaluate Pipeline

Read the repository root `CLAUDE.md` first. This file adds only what is specific to `src/pipelines/evaluate/`.

## Scope

Takes a validation manifest plus either a checkpoint or supplied predictions, and produces COCO-style detection metrics together with the predictions that were actually scored. When Data also supplies an unlabeled COCO test manifest, it runs checkpoint inference and writes a competition submission CSV. It does not train or score test data.

## Boundaries

You own `src/pipelines/evaluate/`. Never edit or import another pipeline; `src/common` is the only shared code available to you. `config["inputs"]` is read-only.

## Interface

`run(config) -> dict` is the only public symbol. On success `artifacts` always carries exactly `run_id`, `metrics_uri`, and `predictions_uri`, plus `submission_uri` only when `test_manifest_uri` is present. On failure, `status="error"` and `artifacts={}` — a partial success is never reported as `ok`.

Settings come from `config["evaluate"]`, falling back to `config["inputs"]`. Set `predictions_input_uri` to skip validation inference and only score existing predictions; test inference still requires a checkpoint. In dummy mode with no `config["evaluate"]`, evaluation is skipped.

Manifests are JSONL or a single COCO document. Both are validated for duplicate ids, annotation references, positive image sizes, and boxes inside the image.

## Outputs

A metrics file and a predictions file under the configured output directory, plus optional `submissions/{run_id}/submission.csv`. They are written only after validation metrics and optional test predictions are complete, so a failure leaves no partial result.

Two rules are easy to undo by accident:

- Stored boxes and scores are **not rounded**. Rounding would make a re-run over the saved predictions produce different numbers.
- A metric that was not computed is `null`, never `0.0`, so "not measured" stays distinct from "measured as zero".

## Run and Test

```
python -m src.main_pipeline --only evaluate
python -m pytest src/pipelines/evaluate/tests -q
```

Tests use contract-shaped fixtures with no upstream pipeline; inference runs on CPU.

## Local Rules

- AP and mAP come from pycocotools `COCOeval`. numpy only builds the IoU threshold array and aggregates `evalImgs` — never write the matching logic yourself.
- The main metric and default interval is `mAP@[0.75:0.95]`. `COCOeval` always receives the fixed ten points `0.50:0.95` and the main interval is a slice, so `mAP50`/`mAP75`/`mAP50_95` are always available. `evaluate.iou_thresholds` is not injectable — any other value raises `ConfigurationError` instead of being ignored.
- `maxDets` is the configured value (4). `COCOeval` writes to stdout, so its calls are wrapped in `redirect_stdout` — web parses the log. `summarize()` is never called; index `eval` directly.
- Progress goes to stderr as `evaluate.progress/1` JSON Lines; the emitter never raises and never writes to stdout.
- `precision50`/`recall50` are aggregate counts at `score >= 0.5`, not the last point of the PR curve. The GUI label change is requested in `contracts/proposals/`; this pipeline does not wait.
- `analysis.py` is the diagnosis layer: threshold sweep, best F1, confusion matrix, per-class summary, per-image failures, false-positive causes. All reads `evalImgs`; extra IoU from `maskUtils.iou`. Four traps:
  - `gtMatches` reflects matching **before** the score filter, so judging misses with it erases ground truth that only a low-score detection touched. Use the ids surviving detections claimed.
  - The confusion matrix needs the `useCats = 0` pass; the default pass matches within a class and cannot show a mix-up.
  - False positives bucket strongest-overlap-first: `duplicate`, `classification`, `localization`, `background`. `duplicate` fits none of the others and would otherwise inflate one. `LOCALIZATION_IOU_FLOOR` keeps far-off boxes out of `localization`.
  - `per_class_summary` only re-sorts `per_class`: `truth_count` splits the groups, `ap = null` is never read as 0, and it is not IoU-keyed. `per_class` order stays.
- Competition runs use the same validation IoU thresholds; test labels and test metrics are never accepted or produced.
- `evaluate.submission_excluded_category_ids` drops those categories from the **submission CSV only**, inside `filter_predictions` *before* the per-image cap so the 4 slots go to scorable rows. Test call only; validation keeps it. Default empty changes nothing.
- Identical output filenames are rejected before anything runs.
- If a later write fails, only files **this run created** are removed. A file that already existed and was overwritten on purpose is left alone; S3 objects are never auto-deleted.
- Every failure is an `EvaluateError` subclass, returned as `status="error"`, never raised out of `run()`.
- The checkpoint payload you load is a contract with train. If it does not match, stop with a clear error instead of guessing.
