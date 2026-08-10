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

You own `src/pipelines/data/`. Do not edit another pipeline and never import one; `src/common` is the only shared code available. `config["inputs"]` is read-only — copy values out rather than mutating it.

## Interface

`run(config) -> dict` is the only public symbol. With `prepare == true`, `artifacts` carries exactly these five non-empty strings:

`train_manifest_uri`, `validation_manifest_uri`, `class_map_uri`, `dataset_summary_uri`, `test_manifest_uri`

The dummy result is unchanged. `prepare == false` validates and returns only the first four URIs. On failure, `status="error"` and `artifacts={}`. Errors are returned, never raised out of `run()`.

## Outputs

Prepared artifacts land under `data.processed_root` (default `datasets/pill_detection/processed/`), in a directory named for the source version, split ratio, seed, and split method. The version is the `v<number>` segment of `raw_prefix` (`raw/v2/original/` → `v2-…`); one without it is rejected rather than guessed, so two sources can never aim at one directory. Local URIs come back repository-relative, S3 URIs unchanged. Nothing is overwritten unless `overwrite` is set.

`test_manifest.json` comes from decoded `test_images/` dimensions and the same class-map object as `class_map.json`. It has no annotations.

`dataset_summary.json` records under `split.checksums` the sha256 and byte size of both split manifests, from the exact bytes storage writes, so `sha256sum` on the stored file matches. Same source, seed, and ratio reproduce the same digests; a changed source changes them even with seed and ratio fixed (`schema_version` `1.1`).

The split unit is a **group**, not one image: the source shoots each pill combination at several angles, and sibling shots split apart inflate validation. The group name is the file name up to the first `_` — the combination code (`GROUP_KEY_DELIMITER`, `GROUP_KEY_TOKENS`) — and lands in one split only. A name that does not fit becomes its own one-image group instead of failing.

Because a group moves whole, the ratio comes last: leakage, category coverage, class distribution, then the target count. A category that cannot reach validation without splitting a group stays in train, listed in `split.train_only_categories` (`schema_version` `1.3`). Covering the rest may push validation past the target; the fill stops at the closest reachable point, recorded in `split.validation_image_ratio`. `split.py` has the full trade-off.

`data.split_method` is `"group"` (default) or `"image"` (the previous image-level split), part of the directory name (`…-group/` beside `…-8020/`). `split.method` and `split.grouping` (rule, group counts) say which data a model trained on; adding them made `schema_version` `1.2`.

`data.measure_validation_similarity` (default `false`) records under `split.validation_similarity` how close each validation crop sits to the nearest **same-class** train crop — a group split cannot stop the same pill looking identical on both sides. It decodes the whole dataset, so it is off by default and the key is then absent, never `0` (`schema_version` `1.4`).

With `overwrite == false`, an exact legacy set of four artifacts is the Scope backfill case: only the missing test manifest is written. Any other partial set fails; an existing file is never replaced.

## Run and Test

```
python -m src.main_pipeline --only data
python -m pytest src/pipelines/data/tests -q
```

Tests run against a fake storage; they need no AWS.

## Local Rules

- **Leakage is the one unrecoverable mistake here.** Competition test images may be read only to build the test manifest; they never enter train/validation, and test annotations are never read. Never relax this so a run passes.
- `split_ratio` accepts only `SPLIT_RATIO_OPTIONS`. Adding one changes the output directory name and so the artifact layout — agree with the train and evaluate owners first.
- Every category that can reach validation must appear in both splits, and one seed must reproduce one split. Never split a group to reach a category that cannot — that is leakage. It becomes train-only and the run reports it.
- Never use `boto3` directly. Go through `src/common/storage.py`.
- The manifest and class-map field names are a contract with train and evaluate. Changing one is a `contracts/proposals/` proposal, not an edit.
