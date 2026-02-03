# MalVec

**Signature-based antivirus misses 40% of malware variants. MalVec catches them.**

Traditional antivirus relies on exact-match signatures. An attacker changes one byte, and the signature fails. MalVec takes a different approach: it converts executables into mathematical fingerprints and classifies them by *similarity* to known threats. Change a few bytes? The fingerprint barely moves. The malware is still caught.

[![CI](https://github.com/mwill20/MalVec/actions/workflows/ci.yml/badge.svg)](https://github.com/mwill20/MalVec/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## How It Works

```
PE Binary → Feature Extraction → Embedding → K-NN Search → Verdict
               (2381 features)     (384-dim)    (FAISS)

                                                 "4 of 5 neighbors are malware"
                                                  → MALWARE (87% confidence)
```

1. **Extract** structural features from a PE file (imports, sections, entropy, headers) — static analysis, the file never runs
2. **Embed** those features into a 384-dimensional vector using Johnson-Lindenstrauss random projection
3. **Search** a FAISS index for the 5 most similar known samples
4. **Vote** — if the majority of neighbors are malware, so is this file

| | Signature-based AV | MalVec |
|---|---|---|
| Exact known malware | Detected | Detected |
| Same malware, one byte changed | **Missed** | Detected |
| New variant, same family | **Missed** | Detected |
| Completely new technique | **Missed** | Flagged for review |
| Explainability | "Matched signature X" | "Similar to samples A, B, C" |

---

## Quick Start

### Install

```bash
git clone https://github.com/mwill20/MalVec.git
cd MalVec
python -m venv venv && source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .
```

### Train

```bash
python -m malvec.cli.train \
  --output ./model \
  --max-samples 50000 \
  --verbose
```

### Classify

```bash
# Single file
python -m malvec.cli.classify \
  --model ./model \
  --file suspicious.exe \
  --show-confidence \
  --show-neighbors

# Batch
python -m malvec.cli.batch \
  --model ./model \
  --input-dir ./samples \
  --output results.csv
```

### Distribute

```bash
# Package model as single file
python -m malvec.cli.archive create --model ./model --output model_v1.malvec

# Inspect without extracting
python -m malvec.cli.archive inspect model_v1.malvec
```

---

## Python API

```python
from pathlib import Path
from malvec.classifier import KNNClassifier
from malvec.extractor import FeatureExtractor
from malvec.embedder import EmbeddingGenerator

clf = KNNClassifier.load("./model")
extractor = FeatureExtractor(sandbox=True)
embedder = EmbeddingGenerator()

features = extractor.extract(Path("suspicious.exe"))
embedding = embedder.generate(features.reshape(1, -1))
result = clf.predict_with_review(embedding)

print(result['predictions'][0])   # MALWARE
print(result['confidences'][0])   # 0.87
print(result['needs_review'][0])  # False
```

---

## Security

MalVec processes potentially hostile files. Five defense layers ensure it does so safely:

| Layer | What It Does |
|-------|-------------|
| **Input Validation** | Rejects files by size (>50MB), magic bytes, format |
| **Process Isolation** | Feature extraction runs in a separate subprocess |
| **Sandboxing** | 30-second timeout, 512MB memory cap |
| **Audit Logging** | Every classification logged as structured JSON (SHA256, no file paths) |
| **Fail-Safe** | Errors produce "needs review," never a false clean |

Malware **never executes**. All analysis is static.

See [docs/SECURITY.md](docs/SECURITY.md) for threat model and controls.

---

## Performance

| Metric | Value |
|--------|-------|
| Classification latency | <1 second per file |
| Accuracy (EMBER 2018) | ~92% |
| Precision | ~91% |
| Recall | ~89% |

---

## Configuration

Create `malvec.yaml` or use environment variables:

```yaml
model_path: ./model
k: 5
confidence_threshold: 0.7
sandbox_enabled: true
timeout: 30
```

```bash
export MALVEC_MODEL_PATH=./model
export MALVEC_TIMEOUT=60
```

---

## Project Structure

```
MalVec/
├── malvec/                 # Core package
│   ├── classifier.py       # K-NN classification
│   ├── embedder.py         # JL random projection
│   ├── extractor.py        # PE feature extraction
│   ├── validator.py        # Input validation
│   ├── sandbox.py          # Sandboxed execution
│   ├── isolation.py        # Process isolation
│   ├── audit.py            # Structured audit logging
│   ├── model.py            # .malvec archive format
│   ├── config.py           # YAML/env configuration
│   ├── exceptions.py       # User-friendly errors
│   ├── progress.py         # Terminal UX (rich)
│   └── cli/                # CLI tools
├── tests/                  # Unit, security, integration, polish
├── docs/                   # User guide, security, deployment, API
├── lessons/                # Educational content (novice + professional tracks)
├── data/                   # Data directory (see data/README.md)
├── .github/                # CI, issue templates, PR template, Dependabot
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Development dependencies
├── pyproject.toml          # Tool configuration (ruff, mypy, coverage)
├── setup.py                # Package metadata
├── Makefile                # Development shortcuts
├── Dockerfile              # Container build
└── pytest.ini              # Test configuration
```

---

## Documentation

| Document | Contents |
|----------|----------|
| [User Guide](docs/USER_GUIDE.md) | Installation, training, classification, batch processing, troubleshooting |
| [Security](docs/SECURITY.md) | Threat model, defense-in-depth layers, audit log format |
| [Deployment](docs/DEPLOYMENT.md) | Docker, Kubernetes, systemd, monitoring, backup |
| [API Reference](docs/API.md) | Full Python API with examples |

---

## Lessons

MalVec includes a two-track educational curriculum:

- **[Novice Track](lessons/novice/)** — Start from "what is malware detection?" and build understanding through guided exercises
- **[Professional Track](lessons/professional/)** — Production architecture, performance tuning, deployment patterns

See [lessons/README.md](lessons/README.md) for the full curriculum.

---

## Requirements

- Python 3.11+
- 8 GB RAM (16 GB recommended)
- 10 GB disk (EMBER dataset)

Core dependencies: `numpy`, `faiss-cpu`, `scikit-learn`, `lief`, `pefile`
Optional: `rich` (terminal UX), `pyyaml` (config files)

---

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt
pre-commit install

# Run all tests
make test

# Run with coverage
make test-coverage

# Lint and format
make lint
make format

# Type checking
make typecheck

# Run everything pre-commit checks
make pre-commit
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide.

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions,
coding standards, and the PR process. All contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## License

MIT — See [LICENSE](LICENSE)

## Acknowledgments

- [EMBER](https://github.com/elastic/ember) dataset by Elastic Security
- [FAISS](https://github.com/facebookresearch/faiss) by Meta AI Research
- [LIEF](https://github.com/lief-project/LIEF) for PE parsing
