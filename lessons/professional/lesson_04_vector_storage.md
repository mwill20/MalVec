# Lesson 4: Vector Storage Architecture

> **Phase:** 4 - Vector Storage  
> **Track:** Professional  
> **Time:** 60-90 minutes

---

## Learning Objectives

By the end of this lesson, you will:

1. Understand FAISS index types and when to use each
2. Design production vector storage systems
3. Implement efficient batch operations
4. Handle index persistence and versioning

---

## 1. FAISS Index Types

### Overview

| Index Type | Use Case | Speed | Memory | Accuracy |
|------------|----------|-------|--------|----------|
| `Flat` | <1M vectors, exact k-NN | Slow | High | 100% |
| `IVF` | 1M-100M vectors, approximate | Fast | Medium | 95-99% |
| `HNSW` | 1M-10M vectors, high recall | Fast | High | 99%+ |
| `PQ` | 100M+ vectors, memory-limited | Very Fast | Very Low | 70-90% |

### MalVec Choice: Flat Index

For malware detection, we prioritize **accuracy over speed**:

```python
# We use IndexFlatIP (Inner Product) for cosine similarity
index = faiss.IndexFlatIP(embedding_dim)

# Or IndexFlatL2 for Euclidean distance
index = faiss.IndexFlatL2(embedding_dim)
```

**Why Flat:**

- Exact k-NN (no false negatives)
- Simple to implement
- Sufficient for <1M samples
- Easy to reason about

### When to Upgrade

| Condition | Recommendation |
|-----------|---------------|
| >1M vectors | Consider IVF |
| Query latency >100ms | Consider HNSW |
| Memory >8GB | Consider PQ |
| Need 99%+ recall | Stay with Flat or HNSW |

---

## 2. Cosine vs Inner Product vs L2

### The Relationship

For **L2-normalized vectors** (||v|| = 1):

```text
cosine_sim(a, b) = dot(a, b) / (||a|| × ||b||)
                 = dot(a, b) / (1 × 1)
                 = dot(a, b)
                 = inner_product(a, b)

L2_distance(a, b)² = ||a - b||²
                   = ||a||² + ||b||² - 2×dot(a, b)
                   = 1 + 1 - 2×dot(a, b)
                   = 2 - 2×inner_product(a, b)
                   = 2×(1 - cosine_sim(a, b))
```

**Key insight:** For normalized vectors, all three metrics are equivalent
(just different scales).

### Which to Use in FAISS?

| FAISS Index | What It Returns | Interpretation |
|-------------|-----------------|----------------|
| `IndexFlatIP` | Inner product | Higher = more similar |
| `IndexFlatL2` | L2 distance² | Lower = more similar |

We use `IndexFlatIP` because:

1. Results are in [0, 1] range (intuitive)
2. Matches sklearn's `cosine_similarity`
3. Higher = more similar (easier to interpret)

---

## 3. The MalVec Implementation

### Architecture

```text
┌──────────────────────────────────────────────────────┐
│                   VectorIndex                        │
├──────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │   Config    │  │ FAISS Index │  │  Metadata   │  │
│  │ (dataclass) │  │  (IndexIP)  │  │   (count)   │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
├──────────────────────────────────────────────────────┤
│  add()     │  search()     │  save()    │  load()   │
└──────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Library | FAISS | Industry standard, battle-tested |
| Index type | Flat | Exact k-NN, no false negatives |
| Metric | Inner Product | Equivalent to cosine for normalized |
| Persistence | index + config files | Separate metadata from binary |
| GPU | Optional default off | CPU sufficient for most use cases |

---

## 4. Batch Operations

### Why Batching Matters

```python
# BAD: Add one at a time
for embedding in embeddings:
    index.add(embedding.reshape(1, -1))  # N function calls

# GOOD: Add all at once
index.add(embeddings)  # 1 function call
```

FAISS is optimized for batch operations. Adding 1000 embeddings at once
is **orders of magnitude faster** than adding 1000 times.

### Implementation Details

```python
def add(self, embeddings: np.ndarray) -> None:
    # Ensure 2D
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    
    # FAISS REQUIREMENTS:
    # 1. Must be float32
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    
    # 2. Must be C-contiguous (row-major)
    if not embeddings.flags['C_CONTIGUOUS']:
        embeddings = np.ascontiguousarray(embeddings)
    
    # Now safe to add
    self._index.add(embeddings)
```

### Memory-Efficient Batch Processing

For very large datasets:

```python
def add_batched(index, embeddings, batch_size=10000):
    """Add embeddings in batches to control memory."""
    n_samples = len(embeddings)
    
    for i in range(0, n_samples, batch_size):
        batch = embeddings[i:i+batch_size]
        index.add(batch)
        
        if i % 100000 == 0:
            print(f"Added {i:,} / {n_samples:,}")
```

---

## 5. Persistence Strategy

### File Format

```text
my_index/
├── vectors.index    # FAISS binary format
└── vectors.config   # JSON metadata
```

**vectors.config:**

```json
{
    "embedding_dim": 384,
    "metric": "cosine",
    "use_gpu": false,
    "count": 500000,
    "version": "1.0.0",
    "created_at": "2026-02-01T20:15:30Z"
}
```

### Versioning Strategy

For production systems:

```python
def save(self, path: str, version: str = None):
    metadata = {
        'embedding_dim': self.config.embedding_dim,
        'metric': self.config.metric,
        'count': self._count,
        'version': version or '1.0.0',
        'created_at': datetime.now().isoformat(),
        'malvec_version': malvec.__version__,
    }
    # ... save files
```

### Index Updates

For adding new malware to existing index:

```python
# Option 1: Append to existing (fast, but no deletions)
index = VectorIndex.load('production_index')
index.add(new_embeddings)
index.save('production_index')

# Option 2: Rebuild (slower, but clean)
all_embeddings = load_all_embeddings()  # includes new
new_index = VectorIndex()
new_index.add(all_embeddings)
new_index.save('production_index_v2')
```

---

## 6. Production Considerations

### Memory Estimation

```text
Memory per vector = embedding_dim × 4 bytes (float32)
Memory for 1M vectors @ 384 dims = 384 × 4 × 1,000,000
                                = 1.536 GB
```

Plus overhead for FAISS internal structures (~10-20%).

### Query Performance

| Index Size | Query Time (Flat) | k |
|------------|-------------------|---|
| 10,000 | <1ms | 5 |
| 100,000 | ~5ms | 5 |
| 1,000,000 | ~50ms | 5 |
| 10,000,000 | ~500ms | 5 |

For malware detection (typically <1M samples), Flat is fine.

### GPU Acceleration

```python
config = VectorIndexConfig(use_gpu=True)
index = VectorIndex(config)

# 10-100x faster for large indexes
# Requires: pip install faiss-gpu
```

---

## 7. Hands-On Exercise

### Task: Build Production Pipeline

```python
from malvec.ember_loader import load_ember_features
from malvec.embedder import EmbeddingGenerator
from malvec.vector_store import VectorIndex, VectorIndexConfig
import numpy as np
import tempfile
import os

# 1. Build the pipeline
print("Loading features...")
X, y = load_ember_features(max_samples=1000)

print("Generating embeddings...")
generator = EmbeddingGenerator()
embeddings = generator.generate(X)

# 2. Build index with explicit config
config = VectorIndexConfig(
    embedding_dim=384,
    metric='cosine'
)
index = VectorIndex(config)
index.add(embeddings)

print(f"Index built: {index.size()} vectors")
print(f"Index info: {index.get_info()}")

# 3. Test persistence
with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, 'test_index')
    
    # Save
    index.save(path)
    print(f"\nSaved to {path}")
    
    # Check files exist
    assert os.path.exists(f"{path}.index")
    assert os.path.exists(f"{path}.config")
    
    # Load
    loaded = VectorIndex.load(path)
    print(f"Loaded: {loaded.size()} vectors")
    
    # Verify search works
    query = embeddings[0:1]
    d_original, i_original = index.search(query, k=5)
    d_loaded, i_loaded = loaded.search(query, k=5)
    
    assert np.allclose(d_original, d_loaded)
    assert np.array_equal(i_original, i_loaded)
    print("\nSearch results match after load!")

# 4. Benchmark search speed
import time

n_queries = 100
queries = embeddings[:n_queries]

start = time.time()
for _ in range(10):  # 10 iterations
    index.search(queries, k=5)
elapsed = time.time() - start

queries_per_second = (n_queries * 10) / elapsed
print(f"\nSearch performance: {queries_per_second:.0f} queries/second")
print(f"Latency: {(elapsed / 10 / n_queries) * 1000:.2f} ms/query")
```

---

## Checkpoint Questions

### Question 1

Why use `IndexFlatIP` instead of `IndexFlatL2` for normalized embeddings?

<details>
<summary>Answer</summary>

For L2-normalized vectors, inner product = cosine similarity:

- Results are in intuitive [0, 1] range
- Higher = more similar (easier to interpret than lower L2 distance)
- Mathematically equivalent (just scaled/shifted)

Both give same k-NN rankings. IP is more intuitive.

</details>

### Question 2

When would you upgrade from Flat to IVF index?

<details>
<summary>Answer</summary>

Upgrade when:

1. **>1M vectors** - Flat query time becomes noticeable (>100ms)
2. **Latency matters** - Need <10ms queries
3. **Can tolerate approximate** - 95% recall is acceptable

Don't upgrade if:

- Accuracy is critical (false negatives are dangerous)
- Dataset is <1M vectors
- Query speed isn't a bottleneck

</details>

### Question 3

What happens if you forget to normalize embeddings before using `IndexFlatIP`?

<details>
<summary>Answer</summary>

Inner product on unnormalized vectors favors vectors with larger magnitude,
not more similar direction.

Example:

- A = [0.1, 0.1] (small magnitude)
- B = [100, 100] (large magnitude)
- Query = [1, 0]

dot(Query, A) = 0.1
dot(Query, B) = 100

B wins even though A might be more "similar" in direction.

Always normalize for cosine similarity!

</details>

---

## Key Takeaways

1. **FAISS Flat** = exact k-NN, best for accuracy-critical applications
2. **Normalize vectors** before using inner product index
3. **Batch operations** are orders of magnitude faster
4. **Persistence** requires both index file and config
5. **Memory** ≈ embedding_dim × 4 bytes × n_vectors

---

## Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Unnormalized vectors with IP | Wrong rankings | Normalize before add/search |
| Adding one at a time | Very slow | Batch insertions |
| Forgetting C_CONTIGUOUS | FAISS crash | `np.ascontiguousarray()` |
| Wrong dtype (float64) | FAISS error | Convert to float32 |
| Not saving config | Can't reload properly | Save both files |

---

## Further Reading

- [FAISS Wiki: Index Types](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [Billion-Scale Approximate Nearest Neighbor Search](https://arxiv.org/abs/1702.08734)
- [Product Quantization for Nearest Neighbor Search](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_pq.pdf)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)
