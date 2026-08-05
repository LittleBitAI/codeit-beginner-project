# Data Pipeline

Read the repository root `AGENTS.md` first. This file adds only what is specific to `src/pipelines/data/`.

## Scope

Turns raw sources into the four dataset artifacts everything downstream trains and evaluates on, or validates URIs that already exist. It does not train, evaluate, or register anything.

Three execution paths, chosen in `__init__.py`:

1. `execution.mode == "dummy"` — return the dummy result untouched.
2. `data.prepare == true` — build the artifacts from raw sources (`preparation.py`).
3. otherwise — validate the four URIs already in `config["inputs"]["data"]` and republish them.

## Boundaries

You own `src/pipelines/data/`. Do not edit another pipeline and never import one; `src/common` is the only shared code available to you. `config["inputs"]` is read-only — copy values out rather than mutating it.

## Interface

`run(config) -> dict` is the only public symbol. On success `artifacts` carries exactly these four non-empty strings:

`train_manifest_uri`, `validation_manifest_uri`, `class_map_uri`, `dataset_summary_uri`

On failure, `status="error"` and `artifacts={}`. Errors are returned, never raised out of `run()`.

## Outputs

Prepared artifacts land under the processed dataset prefix, in a directory whose name encodes the split ratio and the seed, so an 8:2 run and a 9:1 run can never overwrite each other. Local URIs come back repository-relative; S3 URIs are passed through unchanged. Nothing is overwritten unless `overwrite` is set.

## Run and Test

```
python -m src.main_pipeline --only data
python -m pytest src/pipelines/data/tests -q
```

Tests run against a fake storage; they need no AWS.

## Local Rules

- **Leakage is the one unrecoverable mistake here.** `FORBIDDEN_MARKERS` keeps the competition evaluation paths out of every split, and the tests assert storage is never even *asked* for them. Never relax this to make a run succeed.
- `split_ratio` accepts only the values in `SPLIT_RATIO_OPTIONS`. Adding one changes the output directory name, and therefore the artifact layout — agree with the train and evaluate owners first.
- Every category must appear in both splits, and the same seed must reproduce the same split.
- Never use `boto3` directly. Go through `src/common/storage.py`.
- The manifest and class-map field names are a contract with train and evaluate. Changing one is a `contracts/proposals/` proposal, not an edit.
