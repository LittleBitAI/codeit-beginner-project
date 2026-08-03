# Repository-Wide AI Work Rules

## Before Starting Work

- Inspect the repository structure, `git status`, and all relevant files first.
- Read and follow both the repository-root instructions and the nearest `CLAUDE.md` or `AGENTS.md` for the target files.
- Modify only the scope specified by the user. Do not perform out-of-scope cleanup, add unrelated features, or make unrelated refactors.

## Clarifying Vague Implementation Requests

- If the requested implementation is only a rough idea and cannot be made concrete from the repository context, stop before implementation and ask the user to choose a direction.
- Present a concise set of concrete, mutually exclusive choices through an interactive selector that the user can navigate with the arrow keys and confirm with Enter.
- Explain the main outcome or tradeoff of each choice in beginner-friendly language.
- If an arrow-key-and-Enter selector is not available in the current environment, present numbered choices and ask the user to select one before continuing.
- Do not guess or silently expand the scope when the choice could materially change the implementation.

## Ownership and Boundaries

- Do not cross directory ownership boundaries. If another component must change, report it to the user instead of modifying it directly.
- Do not directly import or call another pipeline's internal modules.
- Adjust pipeline connections and execution order only at the designated integration entry point.
- Do not modify, move, overwrite, or delete artifacts created or owned by another component.
- Do not change shared contracts, configuration, dependencies, common code, or integration files unless they are explicitly assigned to you.
- Follow the nearest directory-specific instructions for each pipeline's responsibilities and implementation rules; do not add those rules to the repository-root instructions.

## Security and Repository Hygiene

- Do not hardcode absolute paths, including paths specific to an individual's computer.
- Do not expose credentials, tokens, secrets, `.env` contents, or other sensitive values in code, documentation, logs, or examples.
- Do not add raw or processed datasets, checkpoints or weights, TensorBoard events, training logs or runs, caches, local environments, or large generated files to commits.
- Do not create a commit unless the user explicitly asks. Apply all changes through a Pull Request, and do not commit directly to `main`.
- Pull Request branches must use the format `pipeline/<area>/<task-summary>`. Only during onboarding status checks before role assignment may you use `onboarding/<github-username>`, and then you may change only your own status line in `onboarding/docs/onboarding-status.md`. Both branch types are temporary and must be deleted after merge. Each branch and Pull Request must contain one focused change.
- `<area>` must be the assigned pipeline name: `data`, `train`, `evaluate`, `registry`, or `web`. For repository-wide work or any task without a defined area, do not invent one; ask the user.
- If `main` or a GitHub remote is unavailable and a Pull Request cannot be created, do not substitute a commit or local merge; ask the user.
- When the user requests a commit, write the commit message in Korean while retaining necessary standard technical terms in English.
- When the user explicitly requests a push and Pull Request after a commit, verify the relevant checks and a clean working tree, then use `git pr`. Do not push or create a Pull Request without an explicit request.
- Run `git pr` only from a valid work branch. First use `git pr --dry-run` to confirm the target branch and plan, then review the generated draft Pull Request template and validation details.
- Store text files as UTF-8 without BOM with LF line endings.

## Beginner-Friendly Implementation Report

- After completing the implementation, clearly explain what was implemented in beginner-friendly language. Do not report only filenames or say that the work is complete.
- For every changed file, explain why that file was changed and connect the change to the user's request.
- Describe the implementation's inputs and outputs, including important types, formats, defaults, and assumptions when relevant. If there is no direct input or output, state that explicitly.
- Explain what happens when the implementation fails, including expected errors, fallback behavior, partial results, cleanup, and recovery steps when relevant.
- List every test and check that was run, explain what each one verified, and report its result. If a test was not run, explain why and state the remaining risk.
- State whether the change affects any other pipeline. Name each affected pipeline and explain the impact. If there is no cross-pipeline impact, explicitly say so.
- Confirm that no personal-computer-specific absolute path was introduced. Such paths are prohibited; if one is found, remove it before reporting completion.

## Validation and Completion Reporting

- Run and review the checks and tests relevant to the changed scope.
- On completion, report the changed files, the tests and checks run with their results, unresolved issues or TODOs, and `git status`.
- If any test could not be run, explain why it was skipped and describe the remaining risk.

## Situations That Require Stopping and Asking

In the following situations, do not guess or expand the scope. Stop and ask the user:

- A shared contract or public interface must change.
- A modification would cross a directory ownership boundary or change an integration boundary.
- A destructive action is required, such as deletion, overwriting, or history modification.
- Credentials or secrets are required or could be exposed.
- The work could incur costs, such as creating or modifying paid infrastructure.
- There is a risk of train, validation, test, or competition data leakage.
- Competition rules are unclear in a way that could change the implementation or validation result.
