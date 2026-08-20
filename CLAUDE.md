# Repository-Wide AI Work Rules

## Which File You Read

Read **`CLAUDE.md` only** — `AGENTS.md` is the same content for another tool. In each directory read this file plus the nearest `CLAUDE.md`. Every `CLAUDE.md` has a twin `AGENTS.md`, identical apart from the tool name, so always edit both together.

## Document Map

Korean team docs: `README.md` (how to use the repository), `docs/team.md` (how to contribute to it), `contracts/README.md`, `docs/shared-files.md`. English instructions: this file, `src/pipelines/<area>/CLAUDE.md`, `docs/testing.md`.

Keep every document under 8,000 characters. **Every pipeline has both instruction files and no `README.md`.** Root rules never name a pipeline; pipeline rules do not repeat or contradict them. Ask on conflict.

## Before Starting Work

Inspect the repository, `git status`, and relevant files first. Change only what was asked; never widen scope silently. For a rough idea, stop and offer concrete, mutually exclusive choices with beginner-friendly tradeoffs.

## Ownership and Boundaries

- `src/pipelines/<area>/` belongs to one owner. Do not edit another area.
- Never import another pipeline's internals; a pipeline exposes only `run(config) -> dict`.
- `src/main_pipeline.py` is the only place that orders and connects pipelines.
- Never modify, move, or delete an artifact another component produced.
- Files in `docs/shared-files.md` are shared: changing one takes its own single-purpose PR.
- To request a change you do not own, write `contracts/proposals/NNN-<topic>.md`; the owner implements it.

## How to Write Code

- Python 3.11. Type hints on public functions; Korean docstrings and comments.
- Save every text file as UTF-8 without BOM.
- **No editable install.** `from src...` needs the repository root on `sys.path`: run from there as `python -m pytest` and `python -m src.main_pipeline`. Bare `pytest` breaks imports.
- `run(config)` returns exactly `status`, `artifacts`, `summary`, `message`. Never raise across that boundary — return `status="error"`.
- Raise typed internal errors (an `<Area>Error` subclass), never bare `Exception`.
- No absolute paths; local URIs are repository-relative POSIX and stay inside the repository.
- Never `shell=True`; build explicit argv. Default `overwrite=False` — never destroy what existed before this run.

## TDD: red → green → refactor → prune

Write the failing test, make it pass, clean up, then **prune before opening the PR** — every cycle. To prune, delete one test and break the code it guarded. Another test fails → delete the duplicate. Nothing fails → restore the only guard.

**Keep** contract tests, regression tests, safety tests (leakage, paths outside the repository, credential exposure, overwriting artifacts), edge and failure paths, and one happy-path smoke test per public entry point.

**Delete** self-evident assertions, tests a stronger test supersedes, implementation-detail tests, duplicate parametrize cases, and mock-only tests that never check a result.

Never target a ratio. Test-only deletion gets a separate PR; delete one at a time, rerun the full suite, and name the remaining guard. See `docs/testing.md`.

## Git, Branches, Pull Requests

- Never commit unless asked, and never to `main`. Everything lands through a Pull Request.
- Branch `pipeline/<area>/<task-summary>`; `<area>` is `data`, `train`, `evaluate`, `registry`, `web`, or `docs` for repository-wide work. Never invent one — ask.
- `onboarding/<github-username>` only changes your own line in `onboarding/docs/onboarding-status.md`.
- One focused change per branch and PR; delete the branch after merge.
- Commit messages in Korean, standard technical terms in English.
- Before `git pr`, inspect `git diff --stat origin/main...HEAD` and the full diff. Pass Korean `--summary` and `--reason` values based on that diff: state concrete behaviour or structure changed, then the prior problem and why it was needed. Never repeat the title or commit message as the explanation; cover inputs, outputs, defaults, and failure behaviour when relevant.
- Push and open a PR only when asked: clean tree, passing checks, `git pr --dry-run`, then `git pr`. The command rejects missing, non-Korean, or identical summaries and reasons.

## Stop and Ask

A shared contract or public interface must change; a change crosses an ownership or integration boundary; something must be deleted, overwritten, or history-rewritten; credentials are needed or could leak; the work could cost money; train, validation, test, or competition data could leak; competition rules are unclear in a way that changes the result.

## Security and Hygiene

Never expose credentials, tokens, `.env` contents, or secrets. Never commit datasets, checkpoints, weights, events, logs, caches, environments, or large generated files.

## Reporting

Report in beginner-friendly language: each file and why; inputs, outputs, defaults, assumptions; failures and cleanup; checks and skipped reasons; affected pipelines; no personal absolute paths; changed files, TODOs, and `git status`.
