# MalVec Data Sources

**QUICK START:** Download EMBER → Build embedder → Skip validator/extractor until Phase 7

---

> **Decision:** EMBER-First Approach (Option A)  
> **Status:** ✅ Confirmed  
> **Date:** 2026-02-01

---

## 🎯 Development Strategy

MalVec uses a **two-phase data strategy** to balance security and functionality:

**Phase 1 (Development):** EMBER pre-extracted features  
**Phase 2 (Production):** Raw binary analysis

---

## 📊 Primary Dataset: EMBER

### Overview

**Source:** Elastic Security (formerly Endgame)  
**Repository:** <https://github.com/elastic/ember>  
**License:** Apache 2.0 (Free)  
**Size:** 1.1M samples (EMBER 2018)

### Why EMBER?

✅ **Pre-extracted features** - No raw malware binaries needed  
✅ **Industry standard** - Cited in 100+ research papers  
✅ **Labeled & balanced** - 300K malicious, 300K benign, 300K unlabeled  
✅ **Safe for development** - JSON files, not executables  
✅ **Known baseline** - Compare against EMBER's LightGBM model

### Dataset Structure

```
ember2018/
├── train_features_0.jsonl  # 300K malicious samples
├── train_features_1.jsonl  # 300K benign samples
├── train_features_2.jsonl  # 300K unlabeled samples
└── test_features.jsonl     # 200K test samples (100K mal, 100K benign)
```

### Features Included

Each sample contains:

- **General:** File size, virtual size, has_debug, has_imports, etc.
- **Header:** COFF header, optional header characteristics
- **Imports:** List of imported functions (API calls)
- **Exports:** Exported function names
- **Sections:** Section names, sizes, entropy, characteristics
- **Byte histogram:** 256-value distribution
- **Byte-entropy histogram:** Joint distribution of entropy and byte values
- **Strings:** Extracted string statistics

**These map directly to what our FeatureExtractor would produce!**

### Installation

```bash
# Option 1: Python package (recommended)
pip install ember-ml

# Option 2: Direct download
wget https://ember.elastic.co/ember_dataset_2018_2.tar.bz2
tar -xvf ember_dataset_2018_2.tar.bz2
```

### Loading EMBER Features

```python
import ember

# Load pre-vectorized features
X_train, y_train = ember.read_vectorized_features('/data/ember2018/', 'train')
X_test, y_test = ember.read_vectorized_features('/data/ember2018/', 'test')

# Or load raw metadata
metadata = ember.read_metadata('/data/ember2018/')
```

---

## 🔧 Development Modes

### Mode 1: EMBER Features (Phases 1-6)

**Input:** Pre-extracted EMBER feature dict  
**Pipeline:** Embedder → Vector Store → Classifier  
**Security:** No malware handling needed  
**Use Case:** Training, research, development

```python
# Example EMBER sample
ember_features = {
    "general": {"size": 123456, "vsize": 200000, ...},
    "header": {"coff": {...}, "optional": {...}},
    "imports": ["kernel32.dll!CreateFileA", ...],
    "sections": [{"name": ".text", "size": 8192, ...}],
    ...
}

# Process through MalVec
embedding = embedder.generate(ember_features)
```

### Mode 2: Binary Analysis (Phase 7+)

**Input:** Raw PE/ELF binary file  
**Pipeline:** Validator → Extractor → Embedder → Vector Store → Classifier  
**Security:** Full defense-in-depth (6 layers)  
**Use Case:** Production deployment

```python
# Example binary processing
binary_path = Path("/samples/suspicious.exe")

# Full pipeline with security
validated = validator.validate(binary_path)
features = extractor.extract(validated)  # Produces EMBER-compatible format
embedding = embedder.generate(features)
```

### Two-Mode Pipeline Implementation

```python
# malvec/pipeline.py

class MalVecPipeline:
    def __init__(self, mode: str = 'ember'):
        """
        Initialize MalVec pipeline.
        
        Args:
            mode: 'ember' for pre-extracted features, 'binary' for raw files
        """
        self.mode = mode
        self.embedder = EmbeddingGenerator(...)
        self.store = VectorStore(...)
        self.classifier = KNNClassifier(...)
        
        # Only load extractor in 'binary' mode
        if mode == 'binary':
            self.validator = SampleValidator()
            self.extractor = FeatureExtractor()
    
    def process(self, input_data):
        """Process input and return classification."""
        if self.mode == 'ember':
            # Input is already feature dict from EMBER
            features = input_data
        else:
            # Input is binary file path - full security pipeline
            validated = self.validator.validate(input_data)
            features = self.extractor.extract(validated)
        
        embedding = self.embedder.generate(features)
        neighbors = self.store.find_similar(embedding)
        result = self.classifier.classify(neighbors)
        
        return result
```

---

## 🛡️ Security Implications

### During Development (EMBER)

**Risk Level:** ZERO  
**Reason:** JSON files, no executables  
**Allowed Environment:** Any development machine  
**Restrictions:** None

### During Production (Binaries)

**Risk Level:** HIGH  
**Reason:** Actual malware samples  
**Allowed Environment:** Isolated VM only  
**Restrictions:**

- No network access
- Sandboxed execution
- Encrypted storage
- Legal compliance required

---

## 📚 Additional Datasets (Future)

### BODMAS (Secondary Validation)

**Purpose:** Temporal drift analysis  
**Size:** 134K samples, 5+ years  
**Access:** Free (email request required)  
**Use Case:** Cross-dataset validation, family clustering

### VirusTotal API (Label Enrichment)

**Purpose:** Hash lookups, metadata enrichment  
**Size:** Unlimited  
**Access:** Free tier (4 req/min) or Premium ($500/month)  
**Use Case:** Validate cluster assignments, gather metadata

### MalwareBazaar (Fresh Samples)

**Purpose:** Recent malware for testing  
**Size:** 100K+ samples  
**Access:** Free API  
**Use Case:** Continuous evaluation, drift detection

---

## 🎯 Dataset Usage by Phase

### Phase 1-6: Core Development

**Dataset:** EMBER 2018  
**Mode:** Features only  
**Focus:** Prove embedding approach works  
**Security:** Development machine safe

**Commands:**

```bash
# Download EMBER
python -c "import ember; ember.create_vectorized_features('/data/ember2018/')"

# Train MalVec
python scripts/train.py --data /data/ember2018/ --output /data/malvec/

# Classify sample
python scripts/classify.py --sample-id 12345 --data /data/ember2018/
```

### Phase 7: Binary Integration

**Dataset:** 10-20 samples from MalwareBazaar  
**Mode:** Full pipeline  
**Focus:** Prove end-to-end works with real binaries  
**Security:** Isolated VM required

**Commands:**

```bash
# In isolated VM only!
python scripts/classify.py --binary /samples/suspicious.exe
```

### Phase 8+: Production

**Dataset:** Real-world SOC samples  
**Mode:** Full pipeline  
**Focus:** Operational deployment  
**Security:** Production hardening required

---

## 📖 Feature Format Specification

### EMBER Format (What We Consume)

```json
{
  "sha256": "abc123...",
  "appeared": "2018-02-15",
  "label": 1,  // 0=benign, 1=malicious, -1=unlabeled
  "general": {
    "size": 123456,
    "vsize": 200000,
    "has_debug": 0,
    "exports": 5,
    "imports": 42,
    ...
  },
  "header": {
    "coff": {...},
    "optional": {...}
  },
  "imports": ["kernel32.dll!CreateFileA", ...],
  "sections": [
    {
      "name": ".text",
      "size": 8192,
      "entropy": 6.8,
      "vsize": 8192,
      ...
    }
  ],
  "datadirectories": [...],
  "bytehistogram": [0, 42, 13, ...],  // 256 values
  "byteentropy": [...],
  "strings": {...}
}
```

### MalVec Internal Format (What We Produce)

Our FeatureExtractor (when processing binaries) produces the SAME format as EMBER.

This enables:

- Seamless switching between EMBER and binary modes
- Direct comparison with EMBER baseline
- Consistent embedding generation

---

## 🔗 External References

**EMBER Dataset:**

- GitHub: <https://github.com/elastic/ember>
- Paper: <https://arxiv.org/abs/1804.04637>
- Blog: <https://www.elastic.co/blog/introducing-ember>

**BODMAS Dataset:**

- Website: <https://whyisyoung.github.io/BODMAS/>
- Paper: <https://arxiv.org/abs/2103.00846>

**VirusTotal:**

- API Docs: <https://docs.virustotal.com/reference/overview>
- Pricing: <https://www.virustotal.com/gui/pricing>

**MalwareBazaar:**

- Website: <https://bazaar.abuse.ch/>
- API Docs: <https://bazaar.abuse.ch/api/>

---

## ✅ Decision Log

**Date:** 2026-02-01  
**Decision:** Use EMBER-first approach  
**Rationale:**

- Safest development path (no malware on dev machine)
- Faster MVP (skip validator/extractor initially)
- Industry-standard benchmark
- Enables safe lessons for novice track

**Impact:**

- Build order changed (embedder before extractor)
- Two-mode architecture (EMBER vs binary)
- Security constraints relaxed during dev, enforced in production

**Next Review:** After Phase 6 completion (before adding binary pipeline)

---

**Remember: Document as you build. Update this file when:**

- Adding new datasets
- Changing data sources
- Discovering data quality issues
- Making architectural decisions related to data
