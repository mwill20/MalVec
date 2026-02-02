# MalVec

**Malware Detection via Embedding-Space Analysis**

Detect polymorphic malware variants using embedding geometry and K-NN classification. MalVec identifies malicious binaries by their semantic similarity, catching variants that evade signature-based detection.

---

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train MalVec (with synthetic data for demo)
python -m malvec.cli.train --output ./model --max-samples 1000

# 4. Classify a file
python -m malvec.cli.classify --model ./model --file ./path/to/suspect_file.exe
```

---

## 🎯 What is MalVec?

Traditional signature-based antivirus fails against polymorphic malware. Attackers mutate malware to evade exact-match signatures, causing a cat-and-mouse game where detection always lags behind.

**MalVec solves this differently:**

1. Convert malware binaries to high-dimensional vectors (embeddings)
2. Similar malware clusters together in embedding space
3. Classify unknowns by their nearest neighbors
4. Detect polymorphic variants without retraining

**Key Insight:** Malware families live in "neighborhoods" - distance in embedding space reveals relationships that signature matching misses.

---

## 🏗️ Architecture

```
Binary → Validate → Extract Features → Generate Embedding → Vector DB → K-NN → Classification
   ↓         ↓              ↓                  ↓              ↓        ↓          ↓
[Untrust] [Layer1]     [Sandboxed]        [Isolated]     [FAISS]  [Voting]  [Result]
```

**Two Modes:**

- **EMBER Mode:** Pre-extracted features for development/research (safe)
- **Binary Mode:** Real PE/ELF analysis for production (requires security layers)

---

## 🛡️ Security Invariants

These are **NON-NEGOTIABLE**:

| Invariant | Enforcement |
|-----------|-------------|
| Malware NEVER executes | Static analysis only |
| File paths NEVER in output | Use hashes only |
| All inputs validated | Type + size + magic bytes |
| Sandboxing enforced | Timeouts, no network |
| Fail safely | Flag for review on errors |

---

## 📁 Project Structure

```
malvec/
├── README.md              # This file
├── requirements.txt       # Dependencies
├── setup.py              # Package installation
├── malvec/               # Main package
│   ├── __init__.py
│   ├── embedder.py       # Embedding generation
│   ├── store.py          # Vector storage (FAISS)
│   ├── classifier.py     # K-NN classification
│   ├── validator.py      # Input validation (Phase 7)
│   └── extractor.py      # Feature extraction (Phase 7)
├── tests/                # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── malvec/cli/           # CLI tools
│   ├── train.py
│   ├── classify.py
│   └── batch.py
├── docs/                 # Documentation
│   ├── PRODUCT_VISION.md
│   ├── Phase7_Handover.md
│   └── lessons_learned.md
├── research/             # Jupyter notebooks
└── data/                 # Gitignored!
    ├── samples/
    ├── embeddings/
    └── models/
```

---

## 📊 Development Status

**Phase 1-6:** Core Infrastructure ✅ Complete
**Phase 7:** Input Validation & Binary Pipeline ✅ Complete

- [x] Input Validator (DoS protection)
- [x] Native Feature Extractor (LIEF)
- [x] CLI support for real binaries
- [x] End-to-End verification

**Phase 8:** Security Hardening 🔄 Next

- [ ] Process isolation
- [ ] Resource limits
- [ ] Sandbox implementation

---

## 📚 Documentation

- [PROJECT_BRIEF.md](PROJECT_BRIEF.md) - Problem/solution context
- [NORTHSTAR.md](NORTHSTAR.md) - Architecture blueprint
- [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) - Implementation guide
- [DATA_SOURCES.md](DATA_SOURCES.md) - Dataset decisions
- [docs/PRODUCT_VISION.md](docs/PRODUCT_VISION.md) - End-state goals
- [docs/lessons_learned.md](docs/lessons_learned.md) - Decisions log

---

## 🎓 Learning

MalVec is designed to teach malware analysis through dual-track lessons:

- **Novice Track:** 16 lessons from zero to job-ready
- **Professional Track:** 22 lessons for production deployment

Lessons are generated in Phase 10, after the tool is complete.

---

## ⚠️ Security Notice

This tool is for **defensive security research only**. The `/data/` directory is gitignored to prevent accidental malware commits. Never commit executable samples to version control.

---

## License

MIT License - See LICENSE file for details.
