# Registry Pipeline

Read the repository root `AGENTS.md` first. This file adds only what is specific to `src/pipelines/registry/`.

## Scope

Collects the artifacts produced by data, train, and evaluate into a single experiment record, so a finished run can be found and reproduced later. It does not train, evaluate, or build dataset artifacts, and it never rewrites the artifacts it points at.

## Boundaries

You own `src/pipelines/registry/`. Do not edit another pipeline and never import one; `src/common` is the only shared code available to you. `config["inputs"]` is read-only.

Consumers must not import you either. They read records through `read_experiment_record()` in `src/common`, passing back the exact URI you returned. Keep that facade working — web depends on it.

## Interface

`run(config) -> dict` is the only public symbol. On success `artifacts` carries `run_id` and `experiment_record_uri`. On failure, `status="error"` and `artifacts={}`.

It reads `config["inputs"]["data"]`, `["train"]`, and `["evaluate"]`. Every consumed value must be a non-empty string; booleans, numbers, lists, and nested objects are rejected. When no upstream artifacts are present at all, the run is treated as a dummy run.

Schema 1.2 keeps every required artifact unchanged. It additionally accepts optional `data.test_manifest_uri` and `evaluate.submission_uri`; when present, they follow the same URI safety, verification, provenance, and hashing rules as required artifact URIs.

`verify_artifacts` (default on) additionally confirms the referenced artifacts exist and match their checksums. **URI shape is validated even when verification is turned off** — turning it off must never turn off safety.

A local `evaluate.submission_uri` is also read and checked against the CSV spec in `contracts/proposals/001`. Only what that document already states is checked; whether a `category_id` exists in the test manifest belongs to evaluate.

## Outputs

One experiment record under the registry prefix, in a directory named for the run. Local URIs must be repository-relative: absolute paths, Windows drive letters, paths escaping the repository, and schemes other than `s3://` are all errors.

## Run and Test

```
python -m src.main_pipeline --only registry
python -m pytest src/pipelines/registry/tests -q
```

`smoke_s3.py` is a separate CLI that needs real AWS credentials. Run it only with approval; without AWS it exits 1 rather than pretending to pass.

## Local Rules

- Secrets are redacted before anything is written. A record is a permanent artifact — treat every field in it as public.
- Failures are typed (`MissingInputError`, `InvalidSchemaError`, `CorruptedArtifactError`, `InvalidSubmissionError`) and returned as `status="error"`, never raised out of `run()`.
- The record schema is what every consumer reads. Adding, renaming, or removing a field is a `contracts/proposals/` proposal, not an edit.
- The repository root is derived from this file's location, never hardcoded. Keep it that way.
