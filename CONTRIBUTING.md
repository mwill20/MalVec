# Contributing to MalVec

Thank you for considering contributing to MalVec. This guide covers setup, standards, and process.

## Security First

MalVec processes potentially hostile files. All contributions must maintain these invariants:

1. **Malware never executes** — static analysis only
2. **File paths never in output** — use SHA256 hashes
3. **All inputs validated at boundaries**
4. **Sandboxing enforced** — timeouts, memory caps, no network
5. **Fail safely** — errors produce "needs review," never a false clean

If your change touches file handling, classification output, or external input, explain how it preserves these invariants in your PR description.

## Development Setup

```bash
# Clone and enter the project
git clone https://github.com/mwill20/MalVec.git
cd MalVec

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
.\venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .

# Verify setup
pytest tests/ -v
```

### Prerequisites

- Python 3.11+
- 8 GB RAM minimum (16 GB recommended for full EMBER dataset)

### Optional: EMBER Dataset

For development with real data (not required — tests use synthetic data):

```bash
pip install git+https://github.com/elastic/ember.git
```

## Development Workflow

1. Create a branch from `master`: `git checkout -b feature/your-feature`
2. Make your changes
3. Add or update tests for any new functionality
4. Run the full test suite: `pytest tests/ -v`
5. Run linting: `ruff check .`
6. Run formatting: `ruff format .`
7. Commit with a clear message
8. Push and open a Pull Request

## Code Standards

### Style

- **PEP 8** compliance, enforced by [ruff](https://github.com/astral-sh/ruff)
- **Type hints** required on all public functions
- **Docstrings** required on all public classes and functions (Google style)
- **snake_case** for functions and variables, **PascalCase** for classes

### Testing

- Tests required for all new functionality
- Organize tests by type: `tests/unit/`, `tests/security/`, `tests/integration/`, `tests/polish/`
- Use shared fixtures from `tests/conftest.py`
- Security-sensitive code requires tests in `tests/security/`

### Commits

Use clear, descriptive commit messages:

```
Add batch export to CSV format
Fix sandbox timeout not enforced on Windows
Update API reference for archive commands
```

## Pull Request Process

1. Fill out the PR template completely
2. Update documentation if your change affects usage
3. Add an entry to `CHANGELOG.md` under `[Unreleased]`
4. Ensure CI passes (tests, linting, type checking)
5. Request review from a maintainer

## Architecture Overview

```
PE Binary → Feature Extraction → Embedding (384-dim) → FAISS K-NN → Verdict
               (2381 features)     (JL projection)     (k=5 vote)
```

Key modules:

| Module | Responsibility |
|--------|---------------|
| `extractor.py` | PE feature extraction (LIEF/pefile) |
| `embedder.py` | Gaussian random projection (sklearn) |
| `vector_store.py` | FAISS index operations |
| `classifier.py` | K-NN classification with confidence |
| `sandbox.py` | Sandboxed execution (timeout, memory) |
| `isolation.py` | Subprocess isolation for extraction |
| `audit.py` | Structured JSON audit logging |
| `validator.py` | Input validation (size, format, magic bytes) |

## Reporting Bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:

- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, MalVec version)
- Relevant log output (ensure no file paths or sensitive data)

## Requesting Features

Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) and describe:

- The problem you're trying to solve
- Your proposed solution
- How it maintains the security invariants
- Alternatives you've considered
