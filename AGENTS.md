# Repository-Wide AI Work Rules

## Which File You Read

Read **`AGENTS.md` only** — `CLAUDE.md` is the same content for another tool. In each directory read this file plus the nearest `AGENTS.md`. Every `AGENTS.md` has a twin `CLAUDE.md`, identical apart from the tool name, so always edit both together.

## Document Map

Korean, for the team: `README.md`, `contracts/README.md`, `docs/shared-files.md`. English, for you: this file (**how to work**, everywhere), `src/pipelines/<area>/AGENTS.md` (**what you may do there**), `docs/testing.md`.

Every document stays under 5,000 characters. **Every pipeline directory carries both files and no `README.md`** (READMEs there go stale). This file never names a pipeline; a pipeline file never restates a root rule. A pipeline rule contradicting this file is not written — ask.

## Before Starting Work

Inspect the repository, `git status`, and the relevant files first. Change only what was asked: no unrelated cleanup or refactors, and never widen scope silently. If the request is only a rough idea, stop and offer concrete, mutually exclusive choices with their tradeoffs in beginner-friendly language.

## Ownership and Boundaries

- `src/pipelines/<area>/` belongs to one owner. Do not edit another area.
- Never import another pipeline's internals; a pipeline exposes only `run(config) -> dict`.
- `src/main_pipeline.py` is the only place that orders and connects pipelines.
- Never modify, move, or delete an artifact another component produced.
- Files in `docs/shared-files.md` are shared: changing one takes its own single-purpose PR.
- To request a change you do not own, write `contracts/proposals/NNN-<topic>.md`; the owner implements it.

## How to Write Code

- Python 3.11. Type hints on public functions; Korean docstrings and comments.
- **No editable install.** `from src...` needs the repository root on `sys.path`: run from there as `python -m pytest` and `python -m src.main_pipeline`. Bare `pytest` breaks imports.
- `run(config)` returns exactly `status`, `artifacts`, `summary`, `message`. Never raise across that boundary — return `status="error"`.
- Raise typed internal errors (an `<Area>Error` subclass), never bare `Exception`.
- No absolute paths; local URIs are repository-relative POSIX and stay inside the repository.
- Never `shell=True`; build explicit argv. Default `overwrite=False` — never destroy what existed before this run.

## TDD: red → green → refactor → prune

Write the failing test, make it pass, clean up, then **prune before opening the PR** — every cycle, not a separate cleanup campaign.

**The rule:** delete the test, then deliberately break the code it guarded. Another test fails → it was a duplicate, delete it. Nothing fails → it is the only guard, keep it.

**Keep** contract tests, regression tests, safety tests (leakage, paths outside the repository, credential exposure, overwriting artifacts), edge and failure paths, and one happy-path smoke test per public entry point.

**Delete** self-evident assertions, tests a stronger test supersedes, implementation-detail tests, duplicate parametrize cases, and mock-only tests that never check a result.

Never target a ratio. Delete in a separate PR, one at a time, rerunning the full suite, and say what now guards the deleted case. `docs/testing.md` has examples.

## Git, Branches, Pull Requests

- Never commit unless asked, and never to `main`. Everything lands through a Pull Request.
- Branch `pipeline/<area>/<task-summary>`; `<area>` is `data`, `train`, `evaluate`, `registry`, `web`, or `docs` for repository-wide work. Never invent one — ask.
- `onboarding/<github-username>` only changes your own line in `onboarding/docs/onboarding-status.md`.
- One focused change per branch and PR; delete the branch after merge.
- Commit messages in Korean, standard technical terms in English.
- Push and open a PR only when asked: clean tree, passing checks, `git pr --dry-run`, then `git pr`.

## Stop and Ask

A shared contract or public interface must change; a change crosses an ownership or integration boundary; something must be deleted, overwritten, or history-rewritten; credentials are needed or could leak; the work could cost money; train, validation, test, or competition data could leak; competition rules are unclear in a way that changes the result.

## Security and Hygiene

Never put credentials, tokens, `.env` contents, or secrets into code, docs, logs, or examples. Never commit datasets, checkpoints, weights, TensorBoard events, training logs, caches, environments, or large generated files.

## Reporting

Explain in beginner-friendly language, not just filenames: each changed file and why; inputs, outputs, defaults, assumptions, or that there are none; failure behaviour and cleanup; every check run, its result, and why any was skipped; which pipelines are affected, or that none are; that no personal absolute path was introduced; then changed files, TODOs, and `git status`.
