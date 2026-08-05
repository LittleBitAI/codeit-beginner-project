# Which Tests Are Essential

Detail behind the prune step in the root instruction file. TDD produces tests that exist only to drive one step. Once the feature works, some of them guard nothing. This page decides which.

## The rule

Delete the test, then **deliberately break the code it guarded**.

- Another test fails → it was a duplicate. Delete it.
- Nothing fails → it is the only guard. Keep it.

That is a hand-run mutation test. It replaces judgement about "importance" with something you can check in one command.

## Keep

**1. Contract** — a format shared between pipelines. Breaking it breaks somebody else's work, and they will find out late.
`tests/test_contract.py` (the four-key `run()` return), `src/pipelines/registry/tests/test_input_contract.py`, `src/pipelines/web/tests/test_web_train_contract.py` — the last one AST-parses train's source as text to prove web's mirrored constants have not drifted. It exists because a model name once drifted and the GUI showed the wrong one.

**2. Regression** — a bug that actually happened. The knowledge is not in the code and cannot be guessed from it.
`tests/test_s3_smoke_test.py::test_delete_marker_does_not_count_as_successful_cleanup`: on a versioned bucket `exists()` returns `False` because of the delete marker while the object version is still there, still costing money.

**3. Safety** — things you cannot undo: train/test leakage, reads or writes outside the repository, credential exposure, overwriting an artifact that existed before the run.
`src/pipelines/data/tests/test_dataset_preparation.py` asserts storage is never even *asked* for test images, so leakage cannot happen through any path. Also `web/tests/test_web_masking.py`, `web/tests/test_web_paths.py`.

**4. Edge and failure paths** — empty, zero, out of range, malformed, and cleanup after a mid-run failure. These are what the red step was really for.
`evaluate/tests/test_evaluate_pipeline.py` (a failed second write must not destroy pre-existing predictions), `web/tests/test_web_jobs.py` (reader-thread deadlock on large subprocess output).

**5. One happy-path smoke test per public entry point** — each `run(config)`, CLI, and API route. Proves the thing actually runs.

## Delete

**1. Self-evident** — restating a constant as a literal, `is` alias checks, subclass checks, "the import works". The code already says it, and the assertion has to be edited whenever the code is.
`evaluate/tests/test_evaluate_pipeline.py` has `assert public_run is run`; `tests/test_contract.py` restates `REQUIRED_RETURN_KEYS` as its own literal.

**2. Superseded** — a stronger test already covers it automatically.
`web/tests/test_web_train_config.py` hardcodes eleven of train's default values, but `test_web_train_contract.py` derives the same values from train's source. The hardcoded copy has to be edited by hand every time train changes; the derived one cannot fall behind.

**3. Implementation detail** — private function names, internal call order, exact log strings. Blocks refactoring, guarantees nothing a user can observe.

**4. Duplicate parametrize** — the same equivalence class again and again. One typical value, one boundary, one invalid is enough.
`web/tests/test_web_api.py` runs the same six traversal ids through four separate tests — 24 cases for one rule.

**5. Mock-only** — asserts a mock was called but never checks a result, so it tests the wiring written in the test itself.
Exception: keep tests asserting a call must **not** happen, such as `storage.exists.assert_not_called()`. That is a real guarantee.

## Protocol

1. Prune in a **separate PR** from the implementation, so one revert undoes it.
2. Delete one test, run the full suite, apply the rule above. If nothing else catches the break, restore it.
3. In the PR, name each deleted test and say what now guards that case.
4. **Never target a ratio or a count.** "Cut tests by half" is not a goal.
5. A contract test for a pipeline you do not own is not yours to delete.

## What this repository actually measures

Python implementation 11,333 lines against 9,239 test lines (0.82:1); the frontend is 4,589 against 893 (0.19:1).

Roughly 1,000–1,300 Python test lines are fake storage and process classes, not assertions — `test_train_pipeline.py` reaches its first test only at line 168. Those are **not** deletion targets; consolidating them into `conftest.py` is the fix.

Applying this page to the current suite removes about 2%. A test:implementation ratio near 0.8 is healthy for a repository with real cross-pipeline contracts. The point of this page is to stop scaffolding from accumulating, not to shrink a number.
