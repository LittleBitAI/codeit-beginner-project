# Data Pipeline

Read the repository root `CLAUDE.md` first. This file adds only what is specific to `src/pipelines/data/`.

## Scope

Turns raw sources into five dataset artifacts for downstream training and evaluation, or validates the four legacy URIs that already exist. It does not train, evaluate, or register anything.

Three execution paths, chosen in `__init__.py`:

1. `execution.mode == "dummy"` — return the dummy result untouched.
2. `data.prepare == true` — build five artifacts from raw sources, or add only a missing
   test manifest when all four legacy artifacts already exist (`preparation.py`).
3. otherwise — validate the four URIs already in `config["inputs"]["data"]` and republish them.

## Boundaries

You own `src/pipelines/data/`. Do not edit another pipeline and never import one; `src/common` is the only shared code available to you. `config["inputs"]` is read-only — copy values out rather than mutating it.

## Interface

`run(config) -> dict` is the only public symbol. With `prepare == true`, successful `artifacts` carries exactly these five non-empty strings:

`train_manifest_uri`, `validation_manifest_uri`, `class_map_uri`, `dataset_summary_uri`, `test_manifest_uri`

The dummy result is unchanged. With `prepare == false`, legacy pass-through still validates and returns only the first four URIs. On failure, `status="error"` and `artifacts={}`. Errors are returned, never raised out of `run()`.

## Outputs

Prepared artifacts land under the processed dataset prefix, in a directory whose name encodes the split ratio, the seed, and the split method, so those runs can never overwrite each other. Local URIs come back repository-relative; S3 URIs are passed through unchanged. Nothing is overwritten unless `overwrite` is set.

`test_manifest.json` is generated from decoded `test_images/` dimensions and the same class-map object written to `class_map.json`. It has no annotations.

`dataset_summary.json` records under `split.checksums` the sha256 and byte size of both split manifests, computed from the exact bytes storage writes, so `sha256sum` on the stored file returns the same digest. Same source, seed, and ratio reproduce the same digests; a changed source changes the digest even when seed and ratio stayed fixed (`schema_version` `1.1`).

The split unit is a **group**, not one image: the source shoots each pill combination repeatedly at different angles, and sibling shots split apart inflate validation. The group name is the file name up to the first `_` — the combination code (`GROUP_KEY_DELIMITER`, `GROUP_KEY_TOKENS`) — derived from the name, never listed per file, and a group lands in one split only. A name that does not fit the rule (no delimiter, empty prefix) becomes its own one-image group instead of failing the run.

Because a group moves whole, the ratio comes last: leakage, then category coverage, then class distribution, then the target count. A category in only one group cannot reach both splits without splitting it, so it stays in train, listed in `split.train_only_categories` (`schema_version` `1.3`). Covering the rest may push validation past the target; the fill step stops at the closest reachable point, and `split.validation_image_ratio` records what the run produced. `split.py` states the trade-off in full.

`data.split_method` is `"group"` (default) or `"image"` (the previous image-level split), and it is part of the directory name: `v1-seed42-8020-group/` beside the untouched `v1-seed42-8020/`. `split.method` and `split.grouping` (rule, group counts) say which data a model was trained on; adding them made `schema_version` `1.2`.

With `overwrite == false`, an exact legacy set of four artifacts is a safe backfill case:
the pipeline reads its class map and writes only the missing test manifest. Any other
partial set still fails, and an existing file is never replaced.

## Run and Test

```
python -m src.main_pipeline --only data
python -m pytest src/pipelines/data/tests -q
```

Tests run against a fake storage; they need no AWS.

## Local Rules

- **Leakage is the one unrecoverable mistake here.** Competition test images may be read only to build the test manifest; they never enter train/validation, and test annotations are never read. Never relax this to make a run succeed.
- `split_ratio` accepts only the values in `SPLIT_RATIO_OPTIONS`. Adding one changes the output directory name, and therefore the artifact layout — agree with the train and evaluate owners first.
- Every category with two or more groups must appear in both splits, and the same seed must reproduce the same split. Never relax that, and never split a group to give a one-group category a validation image — that is leakage. Such a category becomes train-only and the run reports it.
- Never use `boto3` directly. Go through `src/common/storage.py`.
- The manifest and class-map field names are a contract with train and evaluate. Changing one is a `contracts/proposals/` proposal, not an edit.
