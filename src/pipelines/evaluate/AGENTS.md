# Evaluate Pipeline

Read the repository root `AGENTS.md` first. This file adds only what is specific to `src/pipelines/evaluate/`.

## Scope

Takes a validation manifest plus a checkpoint or supplied predictions, and produces COCO-style detection metrics with the predictions that were scored. Given an unlabeled COCO test manifest it also runs inference and writes a competition submission CSV. It never trains or scores test data.

## Boundaries

You own `src/pipelines/evaluate/`. Never edit or import another pipeline; `src/common` is the only shared code available. `config["inputs"]` is read-only.

## Interface

`run(config) -> dict` is the only public symbol. On success `artifacts` carries exactly `run_id`, `metrics_uri`, and `predictions_uri`, plus `submission_uri` and `test_predictions_uri` when `test_manifest_uri` is present. On failure, `status="error"` and `artifacts={}` — a partial success is never reported as `ok`.

Settings come from `config["evaluate"]`, falling back to `config["inputs"]`. `predictions_input_uri` skips validation inference and only scores existing predictions; test inference still needs a checkpoint. In dummy mode with no `config["evaluate"]`, evaluation is skipped.

Manifests are JSONL or one COCO document, validated for duplicate ids, annotation references, image sizes, and boxes inside the image.

## Outputs

A metrics file and a predictions file under the output directory, plus `submissions/{run_id}/submission.csv` and `test_predictions.json` when asked. They are written only after validation metrics and test predictions are complete, so a failure leaves no partial result.

Two rules are easy to undo by accident:

- Stored boxes and scores are **not rounded**. Rounding would make a re-run over the saved predictions give different numbers.
- A metric that was not computed is `null`, never `0.0`, so "not measured" stays distinct from "zero".

## Run and Test

```
python -m src.main_pipeline --only evaluate
python -m pytest src/pipelines/evaluate/tests -q
```

Tests use contract-shaped fixtures with no upstream pipeline; inference on CPU.

## Local Rules

- AP and mAP come from pycocotools `COCOeval`. numpy only builds the IoU threshold array and aggregates `evalImgs`; never write the matching logic.
- The main metric is `mAP@[0.75:0.95]`. `COCOeval` always receives the fixed ten points `0.50:0.95` and the main interval is a slice, so `mAP50`/`mAP75`/`mAP50_95` are always available. `evaluate.iou_thresholds` is not injectable — any other value raises `ConfigurationError` rather than being ignored.
- `maxDets` is the configured value (4). `COCOeval` writes to stdout, so calls are wrapped in `redirect_stdout` — web parses the log. `summarize()` is never called; index `eval`.
- Progress goes to stderr as `evaluate.progress/1` JSON Lines; the emitter never raises and never writes stdout.
- `precision50`/`recall50` are aggregate counts at `score >= 0.5`, not the last PR-curve point. A GUI label change is requested in `contracts/proposals/`.
- `analysis.py` diagnoses: threshold sweep, best F1, confusion matrix, per-class summary, per-image failures, false-positive causes. All from `evalImgs`; extra IoU from `maskUtils.iou`. Four traps:
  - `gtMatches` reflects matching **before** the score filter, so judging misses with it erases ground truth a low-score detection touched. Use the ids surviving detections claimed.
  - The confusion matrix needs the `useCats = 0` pass; the default pass matches within a class and cannot show a mix-up.
  - False positives bucket strongest-overlap-first: `duplicate`, `classification`, `localization`, `background`. `duplicate` would otherwise inflate one of the others. `LOCALIZATION_IOU_FLOOR` keeps far-off boxes out of `localization`.
  - `per_class_summary` only re-sorts `per_class`: `truth_count` splits the groups, `ap = null` is never read as 0, and it is not IoU-keyed.
- Competition runs use the same validation IoU thresholds; test labels and metrics are never accepted or produced.
- `test_predictions.json` carries the **same rows as the submission CSV**, so a later stage reads boxes as numbers instead of parsing the CSV back. It is written whenever a test manifest is given. For more fusion candidates than the four a submission keeps, set `max_detections_per_image` to `false`; that run's CSV is then not submittable, which is the point — it exists to be fused. The file records the excluded ids, or a reader takes a dropped class for one the model missed.
- `evaluate.submission_excluded_category_ids` drops those categories from the **submission CSV only**, inside `filter_predictions` *before* the per-image cap so the 4 slots go to scorable rows. Test call only. Default empty changes nothing.
- `evaluate.metrics_excluded_category_ids` drops them from **validation scoring only**, so local mAP averages the classes the competition scores. It filters *before* the per-image cap: filtering after lets an excluded high scorer crowd out a scorable one. Ground-truth images stay — dropping one turns its predictions into false positives — and saved predictions keep every row. Default empty changes nothing.
- Identical output filenames are rejected before anything runs.
- If a later write fails, only files **this run created** are removed; S3 objects are never auto-deleted.
- Every failure is an `EvaluateError` subclass returned as `status="error"`, never raised from `run()`.
- The checkpoint payload is a contract with train. If it does not match, stop with a clear error instead of guessing.
- `mmdetection_backend.py` reads `backend="mmdetection"` checkpoints; no `backend` key still means torchvision, and any other value stops. Contract: `contracts/proposals/012-mmdetection-checkpoint-inference.md`. Five traps:
  - Detector settings are **copied** from train — the ownership boundary forbids importing it. A changed module layout fails loudly on `state_dict`, but a drift in **values only** (thresholds, normalization constants) still loads and only lowers the score. `model_config.schema_version` covers that, and raising it is train's duty.
  - MMDetection gets `num_classes - 1`. Predicted labels `0..N-1` get 1 added back to repository labels `1..N` before the `category_ids` lookup, so `category_ids[0]` stays the background slot.
  - `category_ids` is **required** here, unlike the torchvision path. Missing, it turns a model label straight into a COCO category id; too short, a label that happens to be in range returns another pill's id with no error. Its length is checked against `num_classes` before the model is built.
  - Every `state_dict` key must start with `detector.` — train saves the wrapping adapter. Dropping unprefixed keys quietly would score with a partly loaded model.
  - mmdet is imported only once this backend is chosen. mmcv ships a compiled extension, so a broken install raises things other than `ImportError`; all of it becomes `PredictionError`. mmdet's own version check raises `AssertionError`, which is why the catch is broad.
  - The config goes to the registry as `ConfigDict`, not a plain `dict`. Two-stage detectors read `train_cfg.rpn` as an attribute, so `cascade_rcnn_swin_t_fpn` could not be built at all while a plain dict was passed — and every mocked test still passed. Contract tests that build the **real** model cover this; they skip where mmcv's compiled extension is missing.
  - mmdet 3.3.0 caps mmcv below 2.2.0, but the only extension wheel for this torch is 2.2.0 and mmdet has had no release since. `_shimmed_mmcv_version` lifts that cap for the **one version it was checked against**, nothing else — a range would carry unreleased patches along, and a genuinely wrong pairing would then break somewhere unrecognizable instead of reporting an install problem. The real-model tests skip only when the packages are absent; a broken install has to fail, or a bad wheel reads as a green run.
