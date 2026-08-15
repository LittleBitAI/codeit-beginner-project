# Registry Pipeline

Read the repository root `CLAUDE.md` first. This file adds only what is specific to `src/pipelines/registry/`.

## Scope

Collects the artifacts produced by data, train, and evaluate into a single experiment record, so a finished run can be found and reproduced later. It does not train, evaluate, or build dataset artifacts, and it never rewrites the artifacts it points at.

## Boundaries

You own `src/pipelines/registry/`. Do not edit another pipeline and never import one; `src/common` is the only shared code available to you. `config["inputs"]` is read-only.

Consumers must not import you either. They read records through `read_experiment_record()` in `src/common`, passing back the exact URI you returned. Keep that facade working — web depends on it.

## Interface

`run(config) -> dict` is the only public symbol. On success `artifacts` carries `run_id`, `experiment_record_uri`, and `experiment_summary_uri` (the last one only when the index was written). On failure, `status="error"` and `artifacts={}`.

It reads `config["inputs"]["data"]`, `["train"]`, and `["evaluate"]`. Every consumed value must be a non-empty string; booleans, numbers, lists, and nested objects are rejected. When no upstream artifacts are present at all, the run is treated as a dummy run.

Schema 1.3 keeps every required artifact unchanged. It additionally accepts optional `data.test_manifest_uri`, `evaluate.submission_uri`, and `evaluate.test_predictions_uri`; when present, they follow the same URI safety, verification, provenance, and hashing rules as required artifact URIs. The last one holds the same rows the submission CSV holds, so several runs can be fused later without the record losing track of what went in.

`verify_artifacts` (default on) additionally confirms the referenced artifacts exist and match their checksums. **URI shape is validated even when verification is turned off** — turning it off must never turn off safety.

A local `evaluate.submission_uri` is also read and checked against the CSV spec in `contracts/proposals/001`. Only what that document already states is checked; whether a `category_id` exists in the test manifest belongs to evaluate.

## Outputs

One experiment record under the registry prefix, in a directory named for the run, plus one summary sidecar per run under the index prefix (`registry/index`, configurable). Local URIs must be repository-relative: absolute paths, Windows drive letters, paths escaping the repository, and schemes other than `s3://` are all errors.

## Run and Test

```
python -m src.main_pipeline --only registry
python -m pytest src/pipelines/registry/tests -q
python -m src.pipelines.registry.rebuild_index --config configs/env.local.json
```

`smoke_s3.py` is a separate CLI that needs real AWS credentials. Run it only with approval; without AWS it exits 1 rather than pretending to pass.

## Local Rules

- Secrets are redacted before anything is written. A record is a permanent artifact — treat every field in it as public.
- **The record is the truth and the index is a cache.** A record is written first; if the index write then fails, the run still succeeds and reports `summary["index_status"] = "failed"`. Rebuild it rather than failing the run.
- `summary.py` is the only file allowed to depend on another pipeline's document shape (evaluate's `metrics.json`, train's `training_history.json`). It reads defensively — an unreadable file yields null values, never an error.
- Both files are read wherever they live, `s3://` included, and **only through the storage `src/common` hands back**. Never reach for `boto3` here. A missing storage, a denied bucket, or a broken document falls back to `unavailable`; registration must never fail because a value could not be read.
- `metrics`/`metrics_source`, `losses`/`losses_source` (`training_history` or `unavailable`), and `training`/`training_source` (`config_snapshot` or `unavailable`) share one shape. `losses` carries `best_epoch`, `best_validation_loss`, `final_train_loss`, and `final_validation_loss` under the names train already uses. `training` is copied out of the record's `config_snapshot.train` and must carry **every** setting that changes a run, nested ones included — a screen that offers to rerun a setting it cannot see reruns something else. A missing or wrongly typed value becomes `null` — never a default, never `0.0`, never an error. A summary must not claim a value it did not read.
- `per_class_summary` copies evaluate's own `analysis.per_class_summary` verbatim. Never recount it here — a second opinion on which class is weak makes two screens disagree. It is already a top-N digest, so it does not grow the index; an evaluation that predates the key is `null`.
- `SUMMARY_VERSION` is `"4"`, which only adds keys; every earlier key keeps its name and meaning. Summaries written earlier keep the old shape until their run is registered again, so consumers must survive `losses` or `per_class_summary` being absent, not merely `null`.
- Failures are typed (`MissingInputError`, `InvalidSchemaError`, `CorruptedArtifactError`, `InvalidSubmissionError`) and returned as `status="error"`, never raised out of `run()`.
- The record schema is what every consumer reads. Adding, renaming, or removing a field is a `contracts/proposals/` proposal, not an edit.
- The repository root is derived from this file's location, never hardcoded. Keep it that way.
