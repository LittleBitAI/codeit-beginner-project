# Evaluate Pipeline

Read the repository root `AGENTS.md` first. This file adds only what is specific to `src/pipelines/evaluate/`.

## Scope

Takes a validation manifest plus a checkpoint or supplied predictions and produces COCO-style detection metrics with the scored predictions. Given an unlabeled COCO test manifest it also writes a competition submission CSV. It never trains, and never scores test data.

## Boundaries

You own `src/pipelines/evaluate/`. Never edit or import another pipeline; `src/common` is the only shared code. `config["inputs"]` is read-only.

## Interface

`run(config) -> dict` is the only public symbol. On success `artifacts` carries exactly `run_id`, `metrics_uri`, `predictions_uri`, plus `submission_uri` and `test_predictions_uri` with a `test_manifest_uri`. On failure, `status="error"` and `artifacts={}` — a partial success is never `ok`.

Settings come from `config["evaluate"]`, falling back to `config["inputs"]`. `predictions_input_uri` skips validation inference; `test_predictions_input_uris` fuses saved test predictions and is the one path needing no checkpoint. In dummy mode with no `config["evaluate"]`, evaluation is skipped.

Manifests are JSONL or one COCO document, validated for duplicate ids, annotation references, sizes, and boxes inside the image.

## Outputs

A metrics file and a predictions file under the output directory, plus `submissions/{run_id}/submission.csv` and `test_predictions.json` when asked. All are written only after every result is complete, so a failure leaves no partial output. Two rules are easy to undo by accident:

- Stored boxes and scores are **not rounded**, or a re-run over the saved predictions gives different numbers.
- A metric that was not computed is `null`, never `0.0`, keeping "not measured" distinct from "zero".

## Rerank

`evaluate.rerank_checkpoint_uris` multiplies each **submission** score by `(1 + margin) / 2`: a crop embedding's similarity to the predicted class's reference crops minus its similarity to the nearest crop of any other class (`rerank.py`). Several checkpoints average their margins; the same one twice is refused, or it votes twice. `rerank_crop_bank_uri` (or `inputs.data.crop_bank_uri`) holds those crops; their size and padding come from the bank, not a constant here.

Test path only, **after** filtering: the detector still picks which four boxes stay and the embedding only reorders them **inside a class**, which is where class-averaged AP is won. Reranking before the cap, or before fusion, both scored lower. A class with no reference crop keeps its score — a margin that could not be measured is not zero.

## Run and Test

```
python -m src.main_pipeline --only evaluate
python -m pytest src/pipelines/evaluate/tests -q
```

Fixtures are contract-shaped; inference on CPU.

## Local Rules

- AP and mAP come from pycocotools `COCOeval`; numpy only builds the threshold array and aggregates `evalImgs`. Never write matching logic.
- The main metric is `mAP@[0.75:0.95]`. `COCOeval` always gets the ten fixed points `0.50:0.95` and the main interval is a slice, so `mAP50`/`mAP75`/`mAP50_95` are always there. `evaluate.iou_thresholds` is not injectable.
- `maxDets` is the configured 4. `COCOeval` writes to stdout, so calls are wrapped in `redirect_stdout` — web parses the log. `summarize()` is never called; index `eval`.
- Progress goes to stderr as `evaluate.progress/1` JSON Lines; the emitter never raises or writes stdout.
- `precision50`/`recall50` are aggregate counts at `score >= 0.5`, not the last PR-curve point (proposal 002).
- `analysis.py` diagnoses from `evalImgs`, taking extra IoU from `maskUtils.iou`:
  - `gtMatches` reflects matching **before** the score filter, so judging misses with it erases ground truth a low-score detection touched. Use ids surviving detections claimed.
  - The confusion matrix needs the `useCats = 0` pass; the default pass matches within a class and cannot show a mix-up.
  - False positives bucket strongest-overlap-first, or `duplicate` inflates another; `LOCALIZATION_IOU_FLOOR` keeps far-off boxes out of `localization`. `per_class_summary` only re-sorts, and `ap = null` is never read as 0.
- Competition runs use the validation IoU thresholds; test labels and metrics are never accepted or made.
- `test_predictions.json` carries the **same rows as the submission CSV**, so a later stage reads boxes as numbers. Written with any test manifest, it records the excluded ids and any `rerank`, or a reader takes a dropped class for a miss and a reranked score for a raw one. For more fusion candidates than the four a submission keeps, raise `max_detections_per_image`; that CSV is then unsubmittable, which is the point.
- `evaluate.test_predictions_input_uris` fuses runs' `test_predictions.json` into one submission (`fusion.py`), then filters it as usual. **Two inputs minimum**, each naming a **checkpoint** and a manifest matching this run's ids, image locations, sizes and categories. A fused or reranked result is never an input: re-fusing counts runs as files. Scores scale by how many runs agreed, counted once each at their best box; `fused_from` records them. `fusion_allow_copied_images` exempts **only** locations: a wrong join is silent, a wrong block loud.
- `evaluate.submission_excluded_category_ids` drops those categories from the **submission CSV only**, inside `filter_predictions` *before* the cap so the 4 slots go to scorable rows. Default empty changes nothing.
- `evaluate.validation_sample_size` scores that many validation images, picked class by class so a rare one still appears and identically per `seed` — candidates sit one paper. Every document records it: a sampled score is not a full one.
- `evaluate.metrics_excluded_category_ids` drops them from **validation scoring only**, so local mAP averages the classes the competition scores; it too filters *before* the cap. Ground-truth images stay — dropping one turns its predictions into false positives — and saved predictions keep every row.
- Identical output names are rejected before anything runs.
- If a later write fails, only files **this run created** go; S3 objects are never deleted.
- The checkpoint payload is a contract with train; a mismatch stops with a clear error.
- `mmdetection_backend.py` reads `backend="mmdetection"` checkpoints; no `backend` key still means torchvision, and any other value stops (proposal 012). Traps:
  - Detector settings are **copied** from train — the boundary forbids importing it. A changed layout fails loudly on `state_dict`, but a drift in **values only** (thresholds, normalization) still loads and only lowers the score. `model_config.schema_version` covers that; raising it is train's duty.
  - MMDetection gets `num_classes - 1`, and labels `0..N-1` get 1 added back before the `category_ids` lookup, so `category_ids[0]` stays the background slot.
  - `category_ids` is **required** here, unlike the torchvision path. Missing, a model label becomes a COCO id; too short, an in-range label returns another pill's id with no error. Its length is checked against `num_classes` first.
  - Every `state_dict` key must start with `detector.` — train saves the wrapping adapter. Dropping unprefixed keys quietly would score a partly loaded model.
  - mmdet is imported only once this backend is chosen. A broken mmcv extension raises more than `ImportError`, and mmdet's version check raises `AssertionError`; all of it becomes `PredictionError`.
  - The config goes to the registry as `ConfigDict`: two-stage detectors read `train_cfg.rpn` as an attribute, so `cascade_rcnn_swin_t_fpn` could not be built with a plain dict — and every mocked test passed. Only a test that builds the **real** model covers this.
  - mmdet 3.3.0 caps mmcv below 2.2.0, but this torch's only extension wheel is 2.2.0. `_shimmed_mmcv_version` lifts that cap for the **one version it was checked against**. Real-model tests skip only when the packages are absent; a broken install has to fail.
