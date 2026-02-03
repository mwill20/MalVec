# MalVec Architecture

## System Overview

```
                    ┌─────────────┐
                    │  PE Binary  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────────┐
                    │   Validator     │
                    │  (Layer 1)      │
                    └──────┬──────────┘
                           │
                    ┌──────▼──────────┐
                    │  Feature        │
                    │  Extractor      │
                    │  (Layer 2-3)    │
                    └──────┬──────────┘
                           │
                      2381 features
                           │
                    ┌──────▼──────────┐
                    │  Embedding      │
                    │  Generator      │
                    │  (JL Projection)│
                    └──────┬──────────┘
                           │
                      384-dim vector
                           │
                    ┌──────▼──────────┐
                    │  FAISS Index    │
                    │  (K-NN Search)  │
                    └──────┬──────────┘
                           │
                    ┌──────▼──────────┐
                    │  Classifier     │
                    │  (Vote + Review)│
                    └──────┬──────────┘
                           │
                    ┌──────▼──────────┐
                    │  Audit Logger   │
                    │  (Layer 5)      │
                    └─────────────────┘
```

## Core Components

### 1. Feature Extractor (`malvec/extractor.py`)

- **Purpose:** Parse PE files, extract EMBER-compatible features
- **Technology:** LIEF library
- **Output:** 2381-dimensional feature vector
- **Security:** Sandboxed execution (subprocess, timeout, memory limits)

**Feature Groups:**

| Group | Dimensions | Source |
|-------|-----------|--------|
| Byte histogram | 256 | Raw byte distribution |
| Byte entropy | 256 | Sliding-window entropy |
| Imports | 1280 | DLL/function hashing |
| Exports | 128 | Export table hashing |
| Sections | 255 | Section properties |
| Headers | 62 | PE header fields |
| General info | 10 | File-level metadata |
| Strings | 104 | String statistics |
| Data directories | 30 | Data directory entries |

### 2. Embedding Generator (`malvec/embedder.py`)

- **Purpose:** Reduce dimensionality while preserving distances
- **Algorithm:** Johnson-Lindenstrauss random projection
- **Input:** 2381 dims → **Output:** 384 dims
- **Theory:** Distance preservation with high probability (JL lemma)

**Why Random Projection?**
- Fast: O(nd) matrix multiply
- No training required
- Distance-preserving (JL lemma guarantees)
- Deterministic with fixed seed

### 3. Vector Store (`malvec/vector_store.py`)

- **Purpose:** Fast nearest neighbor search
- **Technology:** FAISS (Facebook AI Similarity Search)
- **Index Type:** IndexFlatIP (exact search, inner product)
- **Metric:** Cosine similarity (via L2-normalized vectors)

### 4. K-NN Classifier (`malvec/classifier.py`)

- **Purpose:** Classification via neighbor voting
- **Algorithm:** K-Nearest Neighbors (k=5 default)
- **Voting:** Majority or weighted by similarity
- **Output:** Prediction + confidence + review flag

**Review Logic:**
- Confidence < threshold → flagged for manual review
- Confidence ≥ threshold → auto-classified

### 5. Security Layers

| Layer | Component | Protection |
|-------|-----------|------------|
| 1 | Input Validation | File size limit (50MB), magic byte check, PE format validation |
| 2 | Process Isolation | Separate subprocess, 512MB RAM limit, no elevated privileges |
| 3 | Sandboxing | 30-second timeout, memory constraints, filesystem isolation |
| 4 | Audit Logging | Structured JSON logs, SHA256 hashes (not filenames) |
| 5 | Fail-Safe | Errors → "needs review", no false negatives, conservative thresholds |

## Data Flow

```
1. User: malvec classify --file sample.exe
        │
2. CLI validates arguments
        │
3. Validator checks file
   ├── Size < 50MB?
   ├── Magic bytes == MZ?
   └── PE format valid?
        │
4. Extractor runs (sandboxed subprocess)
   ├── Parse PE headers
   ├── Extract 9 feature groups
   └── Output: float[2381]
        │
5. Embedder projects
   ├── Random projection matrix × features
   ├── L2 normalize
   └── Output: float[384]
        │
6. Vector store searches
   ├── FAISS IndexFlatIP
   ├── Find k=5 nearest neighbors
   └── Output: indices[] + similarities[]
        │
7. Classifier votes
   ├── Retrieve neighbor labels
   ├── Majority vote
   ├── Calculate confidence
   └── Output: {prediction, confidence, needs_review}
        │
8. Audit logger records
   └── {event, sha256, prediction, confidence, timestamp}
        │
9. CLI displays result
   ├── Prediction: MALWARE | BENIGN
   ├── Confidence: 0.87
   └── Status: AUTO-CLASSIFIED | NEEDS REVIEW
```

## Design Decisions

### Why K-NN Instead of Neural Networks?

| Advantage | Trade-off |
|-----------|-----------|
| No training time (just index) | Linear scaling with dataset size |
| Explainable (show neighbors) | Memory footprint grows |
| Deterministic predictions | Slightly less accurate (~92% vs ~96%) |
| Works with small datasets | |
| No overfitting risk | |

### Why Random Projection Instead of Autoencoders?

| Advantage | Trade-off |
|-----------|-----------|
| No training required | Not optimal (learned embeddings better) |
| Deterministic (reproducible) | Fixed dimensionality |
| Fast (matrix multiply) | |
| Provable guarantees (JL lemma) | |

### Why EMBER Features?

| Advantage | Trade-off |
|-----------|-----------|
| Industry standard benchmark | PE files only |
| Static analysis (no execution) | Misses behavioral patterns |
| Captures structural properties | |
| Robust to basic obfuscation | |

## Performance Characteristics

| Operation | Complexity | Typical Time |
|-----------|------------|-------------|
| Validation | O(1) | <0.01s |
| Feature extraction | O(n) | ~0.5s |
| Embedding | O(n×d) | ~0.01s |
| K-NN search | O(N×d) | ~0.1s |
| **Total** | **O(n + N×d)** | **~0.6s** |

Where: n = file size, d = embedding dim (384), N = dataset size

## Scaling Considerations

| Dataset Size | RAM | Search Time | Recommendation |
|-------------|-----|-------------|----------------|
| 100K samples | ~400MB | ~0.01s | IndexFlatIP (current) |
| 1M samples | ~4GB | ~0.1s | IndexFlatIP still fine |
| 10M samples | ~40GB | ~1s | Switch to FAISS IVF |
| 100M+ samples | Sharded | Variable | GPU + distributed |

## Extension Points

### Adding New Features
Extend `FeatureExtractor` in `malvec/extractor.py` with additional feature groups.

### Custom Embeddings
Replace `EmbeddingGenerator` in `malvec/embedder.py` with learned embeddings (autoencoder, transformer, etc.).

### Alternative Classifiers
Swap `KNNClassifier` in `malvec/classifier.py` for SVM, random forest, or ensemble methods.

## Testing Strategy

| Test Type | Directory | Count | Purpose |
|-----------|-----------|-------|---------|
| Unit | `tests/unit/` | 200+ | Individual components |
| Security | `tests/security/` | 30+ | Sandbox, isolation, validation |
| Integration | `tests/integration/` | 15+ | End-to-end pipeline |
| Polish | `tests/polish/` | 15+ | UX, progress, archiving |

## Deployment Patterns

See [DEPLOYMENT.md](DEPLOYMENT.md) for full details.

**Development:** `python -m malvec.cli.classify --file test.exe`

**Docker:** `docker run -v /samples:/data malvec:latest classify --file /data/test.exe`

**Production (Kubernetes):**
```yaml
replicas: 10
resources:
  limits:
    memory: "2Gi"
    cpu: "1"
```
