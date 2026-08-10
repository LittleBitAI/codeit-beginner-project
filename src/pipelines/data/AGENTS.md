# Data Pipeline

Read the repository root `AGENTS.md` first. This file adds only what is specific to `src/pipelines/data/`.

## Scope

Turns raw sources into five dataset artifacts, or validates four legacy URIs. It does not train, evaluate, or register.

Three execution paths, chosen in `__init__.py`:

1. `execution.mode == "dummy"` — return the dummy result untouched.
2. `data.prepare == true` — build five artifacts, or add a missing test manifest to an exact four-artifact legacy set (`preparation.py`).
3. otherwise — validate the four URIs in `config["inputs"]["data"]` and republish them.

## Boundaries

You own `src/pipelines/data/`. Do not edit another pipeline and never import one; `src/common` is the only shared code available. `config["inputs"]` is read-only — copy values out, never mutate it.

## Interface

`run(config) -> dict` is the only public symbol. With `prepare == true`, `artifacts` has exactly five non-empty strings:

`train_manifest_uri`, `validation_manifest_uri`, `class_map_uri`, `dataset_summary_uri`, `test_manifest_uri`

The dummy result is unchanged. `prepare == false` returns only the first four URIs after validation. Failures return `status="error"`, `artifacts={}`; none escape `run()`.

## Outputs

Artifacts land under `data.processed_root` (default `datasets/pill_detection/processed/`), named for source version, ratio, seed, and method. The version is the `v<number>` segment of `raw_prefix`; a prefix without one is rejected. Local URIs are repository-relative and S3 URIs unchanged. Only `overwrite` permits replacement.

`test_manifest.json` uses decoded `test_images/` dimensions and the `class_map.json` object, with no annotations.

`dataset_summary.json` records sha256 and byte size of the exact split-manifest bytes under `split.checksums`. Same source, seed, and ratio reproduce digests; changed source changes them (`schema_version` `1.1`).

The split unit is a **group**, not one image: sibling shots split apart inflate validation. The group name is the combination code before the first `_` (`GROUP_KEY_DELIMITER`, `GROUP_KEY_TOKENS`) and lands in one split. A nonmatching name is its own group.

Because groups move whole, priorities are leakage, coverage, class distribution, then target count. A category that cannot reach validation without splitting a group stays in train and `split.train_only_categories` (`schema_version` `1.3`). The closest reachable ratio is `split.validation_image_ratio`.

`data.split_method` is `"group"` (default), `"image"` (the previous image-level split), or `"group-angle"`, part of the directory name (`…-group/`, `…-8020/`, `…-group-angle/`). `split.method` and `split.grouping` say which data a model trained on (`schema_version` `1.2`).

`"group-angle"` also gives validation one camera angle only (`data.validation_angle`, default `90`, the token third from the end) and takes that angle out of train entirely. Disjoint combinations are not enough: 79% of validation crops were indistinguishable from a train crop, 0.6% once the angle is held out. It costs a third of the training images, so such a model compares runs rather than being submitted. `split.angle_holdout` records the angle and dropped count; a category left with no training example, including one vanishing from both splits, stops the run.

`data.measure_validation_similarity` (default `false`) records under `split.validation_similarity` how close each validation crop is to the nearest same-class train crop. It reads raw image locations so both backends work, and runs before publishing so a failure leaves no artifacts. When off, the key is absent (`schema_version` `1.4`).

With `overwrite == false`, an exact legacy set of four artifacts is the backfill case: only the missing test manifest is written. Any other partial set fails; an existing file is never replaced.

## Run and Test

```
python -m src.main_pipeline --only data
python -m pytest src/pipelines/data/tests -q
```

Tests use a fake storage.

## Local Rules

- **Leakage is the one unrecoverable mistake here.** Competition test images may be read only to build the test manifest; they never enter train/validation, and test annotations are never read. Never relax this so a run passes.
- `split_ratio` accepts only `SPLIT_RATIO_OPTIONS`. Adding one changes the output directory name and so the artifact layout — agree with the train and evaluate owners first.
- Every category that can reach validation must appear in both splits, and one seed must reproduce one split. Never split a group to reach a category that cannot — that is leakage. It becomes train-only and is reported.
- Never use `boto3` directly. Go through `src/common/storage.py`.
- Manifest and class-map field names are a contract with train and evaluate. Changing one is a `contracts/proposals/` proposal, not an edit.
