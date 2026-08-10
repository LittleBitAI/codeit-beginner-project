# Data Pipeline

Read the repository root `CLAUDE.md` first. This file adds only what is specific to `src/pipelines/data/`.

## Scope

Turns raw sources into five dataset artifacts, or validates the four legacy URIs that already exist. It does not train, evaluate, or register anything.

Three execution paths, chosen in `__init__.py`:

1. `execution.mode == "dummy"` — return the dummy result untouched.
2. `data.prepare == true` — build five artifacts from raw sources, or add only a missing
   test manifest when all four legacy artifacts exist (`preparation.py`).
3. otherwise — validate the four URIs in `config["inputs"]["data"]` and republish them.

## Boundaries

You own `src/pipelines/data/`. Do not edit another pipeline and never import one; `src/common` is the only shared code available. `config["inputs"]` is read-only — copy values out, never mutate it.

## Interface

`run(config) -> dict` is the only public symbol. With `prepare == true`, `artifacts` carries exactly these five non-empty strings:

`train_manifest_uri`, `validation_manifest_uri`, `class_map_uri`, `dataset_summary_uri`, `test_manifest_uri`

The dummy result is unchanged. `prepare == false` validates and returns only the first four URIs. On failure, `status="error"` and `artifacts={}`; errors are returned, never raised out of `run()`.

## Outputs

Prepared artifacts land under `data.processed_root` (default `datasets/pill_detection/processed/`), named for the source version, split ratio, seed, and method. The version is the `v<number>` segment of `raw_prefix` (`raw/v2/original/` → `v2-…`); one without it is rejected, so two sources can never aim at one directory. Local URIs come back repository-relative, S3 URIs unchanged. Nothing is overwritten unless `overwrite` is set.

`test_manifest.json` comes from decoded `test_images/` dimensions and the same class-map object as `class_map.json`, and has no annotations.

`dataset_summary.json` records under `split.checksums` the sha256 and byte size of both split manifests, from the exact bytes storage writes, so `sha256sum` on the stored file matches. Same source, seed, and ratio give the same digests; a changed source changes them even with seed and ratio fixed (`schema_version` `1.1`).

The split unit is a **group**, not one image: the source shoots each combination at several angles, and sibling shots split apart inflate validation. The group name is the file name up to the first `_` — the combination code (`GROUP_KEY_DELIMITER`, `GROUP_KEY_TOKENS`) — and lands in one split only. A name that does not fit is its own group.

Because a group moves whole, the ratio comes last: leakage, coverage, class distribution, then the target count. A category that cannot reach validation without splitting a group stays in train, listed in `split.train_only_categories` (`schema_version` `1.3`). Covering the rest may push validation past the target; the fill stops at the closest point, in `split.validation_image_ratio`.

`data.split_method` is `"group"` (default), `"image"` (the previous image-level split), or `"group-angle"`, part of the directory name (`…-group/`, `…-8020/`, `…-group-angle/`). `split.method` and `split.grouping` say which data a model trained on (`schema_version` `1.2`).

`"group-angle"` also gives validation one camera angle only (`data.validation_angle`, default `90`, the token third from the end) and takes that angle out of train entirely. Disjoint combinations are not enough: 79% of validation crops were indistinguishable from a train crop, 0.6% once the angle is held out. It costs a third of the training images, so such a model compares runs rather than being submitted. `split.angle_holdout` records the angle and dropped count; a category left with no training example, including one vanishing from both splits, stops the run.

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
