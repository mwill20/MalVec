# Lesson 3: Embedding Generation

> **Phase:** 3 - Embedding Generation  
> **Track:** Novice  
> **Time:** 45-60 minutes

---

## Learning Objectives

By the end of this lesson, you will:

1. Understand what embeddings are and why we need them
2. Learn how random projection reduces dimensions
3. Understand L2 normalization for similarity search
4. Generate embeddings from EMBER features

---

## 1. What Are Embeddings?

### The Problem

EMBER features have 2381 dimensions. That's a lot of numbers to handle:

- **Storage:** Each sample takes 2381 × 4 bytes = ~9.5 KB
- **Search:** Comparing two samples requires 2381 operations
- **Visualization:** Impossible to visualize 2381 dimensions

### The Solution: Compress to Lower Dimensions

An **embedding** is a compact representation of data:

```text
Original: 2381 dimensions → Embedding: 384 dimensions

That's ~6x smaller!
```

### Why 384 Dimensions?

- Small enough to be fast
- Large enough to preserve relationships
- Works well with vector databases like FAISS

---

## 2. Johnson-Lindenstrauss Projection

### The Magic of Random Projection

Here's a surprising fact: **random projection preserves distances**.

The Johnson-Lindenstrauss lemma proves that if you project high-dimensional data to lower dimensions using a **random matrix**, the relative distances between points are preserved.

```text
Original space:
  Point A ---- 10 units ---- Point B
  Point A ---- 5 units ---- Point C

After random projection:
  Point A' ---- ~10 units ---- Point B'
  Point A' ---- ~5 units ---- Point C'
```

B is still further from A than C is. The **ordering** is preserved!

### Why This Matters

For malware detection, we care about finding **similar** samples:

- If malware A is similar to malware B in the original space
- They should still be similar after projection
- This lets k-NN search work correctly

### The Code

```python
from sklearn.random_projection import GaussianRandomProjection

# Create projector
projector = GaussianRandomProjection(
    n_components=384,  # Output dimensions
    random_state=42    # For reproducibility
)

# Fit and transform
embeddings = projector.fit_transform(features)
```

---

## 3. L2 Normalization

### What Is L2 Normalization?

L2 normalization scales each vector to have length 1:

```python
# Before: vector might have any length
vector = [3, 4]  # length = 5

# After L2 normalization: length = 1  
normalized = [0.6, 0.8]  # length = 1
```

### Why Normalize?

**Speed:** After normalization, cosine similarity = dot product:

```python
# Without normalization (slow):
cosine_sim = dot(a, b) / (norm(a) * norm(b))

# With normalization (fast):
cosine_sim = dot(a, b)  # That's it!
```

### How It Works

```python
def normalize(vector):
    length = sqrt(sum(x**2 for x in vector))
    return [x / length for x in vector]
```

---

## 4. Using the EmbeddingGenerator

### Basic Usage

```python
from malvec.embedder import EmbeddingGenerator
from malvec.ember_loader import load_ember_features

# Load features
X, y = load_ember_features(max_samples=1000)
print(f"Input shape: {X.shape}")  # (1000, 2381)

# Generate embeddings
generator = EmbeddingGenerator()
embeddings = generator.generate(X)
print(f"Output shape: {embeddings.shape}")  # (1000, 384)
```

### Custom Configuration

```python
from malvec.embedder import EmbeddingGenerator, EmbeddingConfig

config = EmbeddingConfig(
    embedding_dim=256,   # Smaller embeddings
    random_state=123,    # Different seed
    normalize=False,     # Skip normalization
)

generator = EmbeddingGenerator(config)
```

### Checking the Results

```python
import numpy as np

# Verify normalization
norms = np.linalg.norm(embeddings, axis=1)
print(f"All norms ≈ 1: {np.allclose(norms, 1.0)}")  # True

# Check embedding info
info = generator.get_model_info()
print(f"Model: {info['model_name']}")  # random_projection
print(f"Dim: {info['embedding_dim']}")  # 384
```

---

## 5. Hands-On Exercise

### Task: Generate and Analyze Embeddings

```python
from malvec.embedder import EmbeddingGenerator
from malvec.ember_loader import load_ember_features
import numpy as np

# 1. Load exactly 500 samples
X, y = load_ember_features(max_samples=500)
print(f"Loaded {len(X)} samples")

# 2. Create generator and generate embeddings
generator = EmbeddingGenerator()
embeddings = generator.generate(X)

# 3. Verify embedding properties
assert embeddings.shape == (500, 384), "Wrong shape!"
norms = np.linalg.norm(embeddings, axis=1)
assert np.allclose(norms, 1.0), "Not normalized!"

# 4. Compare two samples using cosine similarity
sample_a = embeddings[0]
sample_b = embeddings[1]
similarity = np.dot(sample_a, sample_b)  # Cosine sim = dot product for normalized vectors
print(f"Similarity between sample 0 and 1: {similarity:.4f}")

# 5. Find most similar pair
# (This is expensive for large datasets - just for learning!)
max_sim = -1
best_pair = (0, 0)
for i in range(min(50, len(embeddings))):  # Only check first 50
    for j in range(i+1, min(50, len(embeddings))):
        sim = np.dot(embeddings[i], embeddings[j])
        if sim > max_sim:
            max_sim = sim
            best_pair = (i, j)

print(f"Most similar pair: {best_pair} with similarity {max_sim:.4f}")
```

---

## Checkpoint Questions

### Question 1

What does L2 normalization do to a vector?

<details>
<summary>Answer</summary>

L2 normalization scales a vector so its length (L2 norm) equals 1.
This is done by dividing each component by the vector's length.

**Example:** [3, 4] has length 5, so normalized = [0.6, 0.8]

</details>

### Question 2

Why can we use dot product instead of cosine similarity after normalization?

<details>
<summary>Answer</summary>

Cosine similarity = dot(a, b) / (|a| × |b|)

After L2 normalization, |a| = 1 and |b| = 1, so:

Cosine similarity = dot(a, b) / (1 × 1) = dot(a, b)

The denominator becomes 1, so we can skip it!

</details>

### Question 3

Why use random projection instead of something like PCA?

<details>
<summary>Answer</summary>

1. **Speed:** Random projection is just matrix multiplication - no expensive computation
2. **Simplicity:** No need to compute covariance matrices
3. **Guaranteed:** JL lemma mathematically guarantees distance preservation
4. **No training data needed:** Works on any data without fitting

</details>

---

## Key Takeaways

1. **Embeddings** compress high-dimensional data into a compact form
2. **Random projection** (JL lemma) preserves distances - proven mathematically
3. **L2 normalization** makes cosine similarity = dot product (faster)
4. `EmbeddingGenerator` handles all this with a simple API
5. 2381 dims → 384 dims is ~6x compression while preserving relationships

---

## Next Steps

In Lesson 4, you'll learn about **vector databases** and how to efficiently
search through millions of embeddings to find similar malware samples.

---

## Further Reading

- [Johnson-Lindenstrauss Lemma (Wikipedia)](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma)
- [Cosine Similarity (Wikipedia)](https://en.wikipedia.org/wiki/Cosine_similarity)
- [scikit-learn Random Projection](https://scikit-learn.org/stable/modules/random_projection.html)
