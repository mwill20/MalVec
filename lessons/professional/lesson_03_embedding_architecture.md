# Lesson 03: Embedding Model Selection & Architecture

> **Track:** Professional  
> **Phase:** 3 - Embedding Generation  
> **Duration:** 90-120 minutes  
> **Prerequisites:** Lesson 02, linear algebra, ML model architecture basics

---

## 🎯 Learning Objectives

By the end of this lesson, you will:

1. Evaluate different embedding strategies for numeric features
2. Implement efficient projection layers for dimension reduction
3. Benchmark embedding approaches (speed, memory, quality)
4. Design batch processing for production workloads
5. Understand the trade-offs between embedding quality and inference speed

---

## 📚 Advanced Concepts

### 1. The Numeric Feature Embedding Problem

EMBER features are **not text** - they're numeric. Most embedding models expect text:

```python
# What sentence-transformers expects:
model.encode(["This is a sentence", "Another sentence"])

# What we have:
features = np.array([[0.23, -1.45, 0.89, ...]])  # 2381 numbers
```

**Approaches to bridge this gap:**

| Approach | Pros | Cons |
|----------|------|------|
| Text conversion | Uses pre-trained models | Loses numeric precision |
| Direct projection | Fast, simple | Loses semantic patterns |
| Linear encoder | Trainable, preserves structure | Requires labeled data |
| Autoencoder | Unsupervised, learns compression | Training overhead |

**Our choice:** Projection + Text conversion (hybrid)

- Projection: 2381 → 384 dims (fast, preserves structure)
- Text conversion: Sample features → transformer (captures patterns)

### 2. Johnson-Lindenstrauss Projection

Random projection approximately preserves distances:

```python
def create_projection_matrix(input_dim: int, output_dim: int) -> np.ndarray:
    """Create random orthogonal projection matrix.
    
    JL Lemma: For n points, we need k = O(log(n)/ε²) dimensions
    to preserve pairwise distances within (1±ε).
    
    For n=1M samples, ε=0.1: k ≈ 920 dimensions
    We use 384 (transformer dim) - slight quality trade-off
    """
    rng = np.random.default_rng(42)  # Reproducible
    
    # Random Gaussian matrix
    random_matrix = rng.standard_normal((input_dim, output_dim))
    
    # Orthogonalize for better preservation
    q, _ = np.linalg.qr(random_matrix)
    
    return q.astype(np.float32)
```

**Why orthogonal?**

- Columns are unit length and orthogonal
- Minimizes information loss
- QR decomposition is stable

### 3. Batch Processing Architecture

Production systems need memory-efficient processing:

```python
class EmbeddingGenerator:
    def generate(self, features: np.ndarray) -> np.ndarray:
        """Generate embeddings with memory-efficient batching."""
        n_samples = features.shape[0]
        batch_size = self.config.batch_size
        
        # Pre-allocate output (known size)
        output = np.empty((n_samples, self.embedding_dim), dtype=np.float32)
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = features[start:end]
            
            # Project + embed
            projected = batch @ self.projection_matrix
            embedded = self._embed_batch(projected)
            
            output[start:end] = embedded
        
        return output
```

**Memory analysis:**

- Input: n × 2381 × 4 bytes = 9.5 MB per 1000 samples
- Projection matrix: 2381 × 384 × 4 = 3.6 MB (one-time)
- Output: n × 384 × 4 bytes = 1.5 MB per 1000 samples
- Batch overhead: batch_size × 384 × 4 = 50 KB for batch_size=32

### 4. Model Selection Trade-offs

Embedding models have different characteristics:

| Model | Dims | Speed | Quality | Size |
|-------|------|-------|---------|------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | 80MB |
| all-mpnet-base-v2 | 768 | Medium | Better | 420MB |
| paraphrase-MiniLM-L6 | 384 | Fast | Good | 80MB |
| Custom trained | Varies | Varies | Best | Varies |

**For MalVec, we chose MiniLM because:**

- Small footprint (80MB)
- Fast inference
- 384 dims works well with FAISS
- Good enough quality for prototype

### 5. Normalization Strategies

Different normalization approaches:

```python
# L2 normalization (unit length)
def l2_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)

# Standardization (zero mean, unit variance)
def standardize(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)

# Min-max scaling (0-1 range)
def minmax(x):
    return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
```

**We use L2 for final embeddings:**

- Required for cosine similarity
- FAISS IndexFlatIP assumes unit vectors
- Makes similarity computation = dot product

---

## 🔧 Hands-On Exercises

### Exercise 3.1: Benchmark Embedding Approaches

Compare different embedding strategies:

```python
import time
import numpy as np
from malvec.ember_loader import load_ember_features, EMBER_FEATURE_DIM
from malvec.embedder import EmbeddingGenerator, EmbeddingConfig

# Load test data
X, y = load_ember_features(max_samples=1000)

# Approach 1: Default (projection + transformer)
start = time.time()
gen1 = EmbeddingGenerator()
emb1 = gen1.generate(X)
time1 = time.time() - start

# Approach 2: Projection only (fallback mode)
# (Simulate by checking if transformer is used)

print(f"Full pipeline: {time1:.2f}s for {len(X)} samples")
print(f"Throughput: {len(X)/time1:.0f} samples/sec")
print(f"Embedding shape: {emb1.shape}")
```

### Exercise 3.2: Analyze Distance Preservation

Test if projection preserves distances:

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform
from malvec.ember_loader import load_ember_features
from malvec.embedder import EmbeddingGenerator

# Load features
X, _ = load_ember_features(max_samples=100)

# Original pairwise distances
orig_dist = pdist(X, 'euclidean')

# Generate embeddings
gen = EmbeddingGenerator()
embeddings = gen.generate(X)

# Embedded pairwise distances  
emb_dist = pdist(embeddings, 'euclidean')

# Correlation between original and embedded distances
correlation = np.corrcoef(orig_dist, emb_dist)[0, 1]
print(f"Distance correlation: {correlation:.4f}")

# Perfect preservation = 1.0, random = 0.0
# JL projection typically achieves 0.7-0.9
```

### Exercise 3.3: Design a Custom Embedding Strategy

Implement an alternative embedding approach:

```python
# exercises/custom_embedder.py

import numpy as np
from malvec.ember_loader import EMBER_FEATURE_DIM

class FeatureGroupEmbedder:
    """Embed each EMBER feature group separately, then concatenate.
    
    This approach respects the structure of EMBER features:
    - Histogram (256) → 32 dims
    - ByteEntropy (256) → 32 dims
    - Strings (104) → 16 dims
    - etc.
    
    Total: ~128 dims (vs 384 for full projection)
    """
    
    FEATURE_GROUPS = {
        'histogram': (0, 256),
        'byteentropy': (256, 512),
        'strings': (512, 616),
        'general': (616, 626),
        'header': (626, 688),
        'section': (688, 943),
        'imports': (943, 2223),
        'exports': (2223, 2351),
        'datadirectories': (2351, 2381),
    }
    
    def __init__(self, dims_per_group: int = 16):
        self.dims_per_group = dims_per_group
        self._init_projections()
    
    def _init_projections(self):
        """Create per-group projection matrices."""
        rng = np.random.default_rng(42)
        self.projections = {}
        
        for name, (start, end) in self.FEATURE_GROUPS.items():
            group_dim = end - start
            proj = rng.standard_normal((group_dim, self.dims_per_group))
            q, _ = np.linalg.qr(proj)
            # Handle case where group_dim < dims_per_group
            self.projections[name] = q[:, :min(self.dims_per_group, group_dim)]
    
    def generate(self, features: np.ndarray) -> np.ndarray:
        """Generate per-group embeddings and concatenate."""
        # TODO: Implement
        pass

# Test your implementation
embedder = FeatureGroupEmbedder()
X, _ = load_ember_features(max_samples=10)
emb = embedder.generate(X)
print(f"Custom embeddings: {emb.shape}")
```

---

## ✅ Checkpoint: Architecture Review

### Question 1: Why not train a custom autoencoder?

**Expected points:**

- Requires labeled malware data (expensive to obtain)
- Training infrastructure overhead
- Pre-trained transformers work well enough
- Can upgrade later without changing API
- Trade-off: custom = better quality, more effort

### Question 2: How does batch size affect memory and speed?

**Expected points:**

- Larger batch = more parallel compute (faster GPU)
- Larger batch = more memory per batch
- Sweet spot depends on hardware
- 32 is conservative default (works on most GPUs)
- Monitor GPU memory to tune

### Question 3: What happens if projection loses important information?

**Expected points:**

- JL lemma guarantees approximate preservation
- If quality degrades, increase projection_dim
- Can train learned projection (supervised)
- Feature selection as alternative (domain knowledge)
- Monitor downstream task performance

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Embedding Generation Pipeline                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    EmbeddingConfig                           │    │
│  │  • model_name: 'all-MiniLM-L6-v2'                           │    │
│  │  • batch_size: 32                                            │    │
│  │  • normalize: True                                           │    │
│  │  • projection_dim: 384                                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                               │                                      │
│                               ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  EmbeddingGenerator                          │    │
│  │                                                              │    │
│  │    Input: EMBER features (n, 2381)                          │    │
│  │                     │                                        │    │
│  │                     ▼                                        │    │
│  │    ┌─────────────────────────────┐                          │    │
│  │    │   Projection Layer          │                          │    │
│  │    │   (2381, 384) orthogonal    │                          │    │
│  │    │   JL distance preservation  │                          │    │
│  │    └──────────────┬──────────────┘                          │    │
│  │                   │                                          │    │
│  │                   ▼  (n, 384)                                │    │
│  │    ┌─────────────────────────────┐                          │    │
│  │    │   Batch Processing          │                          │    │
│  │    │   Memory-efficient iteration│                          │    │
│  │    └──────────────┬──────────────┘                          │    │
│  │                   │                                          │    │
│  │                   ▼                                          │    │
│  │    ┌─────────────────────────────┐                          │    │
│  │    │   Transformer Encoder       │                          │    │
│  │    │   (MiniLM-L6-v2)           │                          │    │
│  │    │   OR projection fallback    │                          │    │
│  │    └──────────────┬──────────────┘                          │    │
│  │                   │                                          │    │
│  │                   ▼  (n, 384)                                │    │
│  │    ┌─────────────────────────────┐                          │    │
│  │    │   L2 Normalization          │                          │    │
│  │    │   ||embedding|| = 1.0       │                          │    │
│  │    └──────────────┬──────────────┘                          │    │
│  │                   │                                          │    │
│  │                   ▼                                          │    │
│  │    Output: Normalized embeddings (n, 384)                   │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎓 Key Takeaways

1. **Numeric features need special handling** - Transformers expect text, we project first

2. **JL projection preserves distances** - Random orthogonal matrices work surprisingly well

3. **Batch processing is production-essential** - Never assume unlimited memory

4. **Model selection is a trade-off** - Speed vs quality vs size

5. **L2 normalization enables similarity** - Required for cosine/dot product search

---

## 📖 Further Reading

- [Johnson-Lindenstrauss Lemma (Paper)](https://cseweb.ucsd.edu/~dasgupta/papers/jl.pdf)
- [Sentence-BERT: Sentence Embeddings (Paper)](https://arxiv.org/abs/1908.10084)
- [Efficient Estimation of Word Representations (Word2Vec)](https://arxiv.org/abs/1301.3781)
- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)

---

## ➡️ Next Lesson

**Lesson 04: Vector Storage with FAISS** - Build and query vector indices for similarity search at scale.

---

*Lesson created during Phase 3 build. Last updated: 2026-02-01*
