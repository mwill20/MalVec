<p align="center">
  <img src="docs/MalVec.png" alt="MalVec Logo" width="300">
</p>

<h1 align="center">MalVec</h1>

<p align="center"><strong>Explainable malware classification using structural similarity.</strong></p>

<p align="center">
  <a href="https://github.com/mwill20/MalVec/actions/workflows/ci.yml"><img src="https://github.com/mwill20/MalVec/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/mwill20/MalVec"><img src="https://codecov.io/gh/mwill20/MalVec/branch/master/graph/badge.svg" alt="Coverage"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Code style: ruff"></a>
</p>

MalVec classifies PE files as malware or benign by comparing them to known samples. Unlike commercial detection systems that return opaque risk scores, MalVec shows you *why*: it lists the 5 most similar samples and their labels.

**Example output:**
```
Sample: suspicious.exe
Prediction: MALWARE (confidence: 0.87)

Most similar samples:
  #1: Emotet variant (92% similar)
  #2: Emotet variant (89% similar)
  #3: Emotet variant (85% similar)
  #4: Benign update tool (78% similar)
  #5: Emotet variant (76% similar)
```

This explainability helps analysts understand *what* the file resembles, not just whether it's risky.

---

## How It Works

```
PE Binary → Feature Extraction → Embedding → K-NN Search → Vote
               (2381 features)     (384-dim)    (FAISS)    (5 neighbors)
```

1. **Extract** 2381 structural features (imports, sections, entropy, headers) using EMBER methodology
2. **Project** to 384 dimensions using Johnson-Lindenstrauss random projection
3. **Search** FAISS index for 5 most similar known samples
4. **Vote** — majority label determines classification, agreement rate determines confidence

---

## Why MalVec?

### What Makes It Different

**Explainability**
- Most detection systems: "This file is suspicious" (black box)
- MalVec: "92% similar to these 5 known Emotet samples"
- Value: Threat intelligence, analyst training, incident reports

**Offline Operation**
- No cloud lookups required
- Build and query local FAISS index
- Value: Air-gapped networks, privacy, compliance

**Reproducibility**
- Deterministic: same file → same classification always
- Open methodology: EMBER features + K-NN with public datasets
- Value: Research, forensics, legal evidence

**No Agent Required**
- API/CLI/container deployment
- Scan files anywhere: disk images, file shares, object storage
- Value: Flexibility, forensics, integration

**Open Source**
- Transparent implementation
- Auditable security controls
- Customizable for research
- Value: Trust, education, modification

### Limitations

**What MalVec doesn't do:**
- ❌ Behavioral detection (static analysis only)
- ❌ Runtime protection (scans files, doesn't monitor processes)
- ❌ Automatic updates (you maintain the index)
- ❌ Zero-day detection via anomaly (relies on similarity to known samples)

**When to use something else:**
- Real-time endpoint protection → Use commercial EDR
- Behavioral analysis → Use sandbox or EDR
- Comprehensive security suite → Use commercial platform

---

## Use Cases

### 1. Forensic Analysis

**Scenario:** Analyzing 50,000 files from a compromised disk image.

**Why MalVec:**
- Scan offline forensic images (no live endpoint needed)
- Explainable results help threat intelligence
- Deterministic classifications for legal evidence

```bash
# Mount forensic image
mount -o ro /evidence/disk.img /mnt/analysis

# Scan all files
malvec batch --input /mnt/analysis --output results.csv

# Review high-confidence malware
cat results.csv | grep "MALWARE" | awk -F, '$3 > 0.85' | sort -t, -k3 -nr
```

---

### 2. Threat Intelligence Enrichment

**Scenario:** Security team wants to understand malware family relationships.

**Why MalVec:**
- See which known families a sample resembles
- Cluster similar samples for family analysis
- Export neighbor graphs for reporting

```python
result = clf.predict_with_neighbors(sample, k=10)

print(f"This sample is {result.confidence:.0%} similar to:")
for neighbor in result.neighbors:
    print(f"  - {neighbor.family} (sample {neighbor.id})")
```

---

### 3. Air-Gapped Environments

**Scenario:** Classified network with no internet access.

**Why MalVec:**
- Fully offline operation (no cloud dependencies)
- Build index from trusted dataset
- Update via sneakernet when needed

---

### 4. Security Research

**Scenario:** Academic research on malware detection methods.

**Why MalVec:**
- Open source methodology (EMBER + K-NN)
- Reproducible experiments
- Cite-able implementation
- Educational value (transparent algorithm)

---

### 5. Second Opinion Analysis

**Scenario:** Analyst wants independent validation of primary scanner's verdict.

**Why MalVec:**
- Different methodology (K-NN vs ML/behavioral)
- Explainable disagreements ("your scanner says clean, but this is 90% similar to Emotet")
- Catches edge cases via structural similarity

```python
# Compare verdicts
primary_verdict = existing_scanner.scan(file)
malvec_result = malvec.classify(file)

if primary_verdict != malvec_result.prediction:
    flag_for_review(file, {
        'primary': primary_verdict,
        'malvec': malvec_result.prediction,
        'confidence': malvec_result.confidence,
        'similar_to': malvec_result.neighbors
    })
```

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
# Train on EMBER dataset (or your own labeled samples)
python -m malvec.cli.train \
  --output ./model \
  --max-samples 50000 \
  --verbose
```

### Classify

```bash
# Single file with neighbor details
python -m malvec.cli.classify \
  --model ./model \
  --file suspicious.exe \
  --show-neighbors

# Batch processing
python -m malvec.cli.batch \
  --model ./model \
  --input-dir ./samples \
  --output results.csv
```

### Distribute

```bash
# Package model as single .malvec archive
python -m malvec.cli.archive create --model ./model --output model_v1.malvec

# Inspect metadata without extracting
python -m malvec.cli.archive inspect model_v1.malvec
```

---

## Python API

```python
from pathlib import Path
from malvec.classifier import KNNClassifier
from malvec.extractor import FeatureExtractor
from malvec.embedder import EmbeddingGenerator

# Load model
clf = KNNClassifier.load("./model")
extractor = FeatureExtractor(sandbox=True)
embedder = EmbeddingGenerator()

# Classify file
features = extractor.extract(Path("suspicious.exe"))
embedding = embedder.generate(features.reshape(1, -1))
result = clf.predict_with_review(embedding)

print(f"Prediction: {result['predictions'][0]}")      # MALWARE or BENIGN
print(f"Confidence: {result['confidences'][0]:.2f}")  # 0.87
print(f"Review needed: {result['needs_review'][0]}")  # True/False
```

---

## Security

MalVec processes potentially hostile files using defense-in-depth:

| Layer | Protection |
|-------|------------|
| **1. Input Validation** | Reject oversized files (>50MB), verify PE magic bytes, validate format |
| **2. Process Isolation** | Feature extraction runs in separate subprocess with resource limits |
| **3. Sandboxing** | 30-second timeout, 512MB memory cap, no network access |
| **4. Audit Logging** | All operations logged with SHA256 hashes (not file paths) |
| **5. Fail-Safe** | Errors trigger manual review flag, never auto-classify as clean |

**Malware never executes.** All analysis is static.

See [docs/SECURITY.md](docs/SECURITY.md) for full threat model.

---

## Performance

### EMBER 2018 Benchmark (600K samples)

| Metric | Value |
|--------|-------|
| **Accuracy** | 92.3% |
| **Precision** | 91.8% |
| **Recall** | 89.4% |
| **F1 Score** | 90.6% |
| **Speed** | 0.6s per file |
| **Throughput** | ~6,000 files/hour (single CPU core) |
| **Memory** | 4GB for 1M sample index |

### Scaling

| Deployment | Throughput | Example Use Case |
|------------|-----------|------------------|
| Single CPU | 6,000 files/hour | Development, small teams |
| 10-core server | 60,000 files/hour | Medium enterprise |
| 100-pod K8s | 600,000 files/hour | Large-scale scanning |
| GPU-accelerated FAISS | 1M+ files/hour | Forensic analysis |

---

## Configuration

**YAML config:**
```yaml
model_path: ./model
k: 5
confidence_threshold: 0.7
sandbox_enabled: true
timeout: 30
```

**Environment variables:**
```bash
export MALVEC_MODEL_PATH=./model
export MALVEC_K=5
export MALVEC_CONFIDENCE_THRESHOLD=0.7
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
│   ├── audit.py            # Structured logging
│   ├── model.py            # .malvec archive format
│   ├── config.py           # YAML/env configuration
│   ├── exceptions.py       # Helpful error messages
│   ├── progress.py         # Terminal UI
│   └── cli/                # Command-line tools
├── tests/                  # 260+ tests (unit, security, integration)
├── docs/                   # Architecture, security, deployment, API docs
├── lessons/                # Educational content (novice + professional tracks)
├── .github/                # CI/CD, issue templates, PR template
└── requirements.txt        # Dependencies
```

---

## Documentation

| Document | Contents |
|----------|----------|
| [Architecture](docs/ARCHITECTURE.md) | System design, data flow, design decisions |
| [User Guide](docs/USER_GUIDE.md) | Installation, training, classification workflows |
| [Security](docs/SECURITY.md) | Threat model, defense layers, audit logging |
| [Deployment](docs/DEPLOYMENT.md) | Docker, Kubernetes, production setup |
| [API Reference](docs/API.md) | Python API documentation with examples |
| [Roadmap](ROADMAP.md) | Planned features and research directions |

---

## Lessons

MalVec includes educational content for two audiences:

- **[Novice Track](lessons/novice/)** — Learn malware detection from scratch through guided exercises
- **[Professional Track](lessons/professional/)** — Production architecture, performance tuning, advanced topics

See [lessons/README.md](lessons/README.md) for the full curriculum.

---

## Requirements

- Python 3.11+
- 8 GB RAM (16 GB recommended for large datasets)
- 10 GB disk space (for EMBER dataset)

**Core dependencies:** `numpy`, `faiss-cpu`, `scikit-learn`, `lief`, `pefile`  
**Optional:** `rich` (terminal UI), `pyyaml` (config files)

---

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt
pre-commit install

# Run tests
make test                # All tests
make test-coverage       # With coverage report
make test-security       # Security tests only

# Code quality
make lint                # Check style
make format              # Auto-format
make typecheck           # mypy type checking

# Full pre-commit suite
make pre-commit
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development guide.

---

## Contributing

Contributions welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Coding standards
- PR process
- Testing requirements

All contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## License

MIT — See [LICENSE](LICENSE)

---

## Acknowledgments

- [EMBER](https://github.com/elastic/ember) dataset by Elastic Security
- [FAISS](https://github.com/facebookresearch/faiss) by Meta AI Research
- [LIEF](https://github.com/lief-project/LIEF) for PE parsing
