## What does this PR do?

Brief description of the changes.

## Why?

Motivation and context. Link related issues with `Fixes #123` or `Relates to #456`.

## Security Impact

Does this PR touch file handling, classification output, or external input?
If yes, explain how the [security invariants](docs/SECURITY.md) are maintained.

- [ ] Malware never executes
- [ ] No file paths in output
- [ ] Inputs validated at boundaries
- [ ] Sandboxing preserved
- [ ] Errors fail safely

## Checklist

- [ ] Tests added/updated
- [ ] Documentation updated (if usage changed)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] All tests pass locally (`pytest tests/ -v`)
- [ ] Linting passes (`ruff check .`)
