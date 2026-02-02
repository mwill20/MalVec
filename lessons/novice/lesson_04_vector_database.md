# Lesson 4: Vector Database Basics

> **Phase:** 4 - Vector Storage  
> **Track:** Novice  
> **Time:** 45-60 minutes

---

## Learning Objectives

By the end of this lesson, you will:

1. Understand what vector databases are and why we need them
2. Learn how FAISS enables fast similarity search
3. Build and query a vector index
4. Save and load indexes for persistence

---

## 1. The Search Problem

### The Naive Approach

If you have 1 million malware embeddings and want to find the 5 most similar to a query:

```python
# Naive approach - compare with EVERY sample
for i in range(1_000_000):
    similarity = dot(query, all_embeddings[i])
    # track top 5...
```

**Time:** 1 million operations per query. Very slow!

### The Smart Approach: Vector Indexes

A vector index organizes embeddings so we can find similar ones **without** checking every single one:

```text
Query → Index → Top 5 results

Instead of 1,000,000 comparisons
We do maybe 1,000 comparisons
That's 1000x faster!
```

---

## 2. What Is FAISS?

**FAISS** = Facebook AI Similarity Search

- Library for efficient similarity search
- Handles billions of vectors
- Created by Facebook/Meta AI Research
- Industry standard (used by Spotify, Pinterest, etc.)

### How It Works (Simplified)

FAISS organizes vectors into regions:

```text
Original: [ * * * * * * * * * * ... 1 million stars ]

FAISS groups them:
[Region 1: ***]  [Region 2: ***]  [Region 3: ***] ...

When searching:
1. Find which regions are likely matches
2. Only search within those regions
3. Much faster!
```

---

## 3. Creating an Index

### Basic Usage

```python
from malvec.vector_store import VectorIndex
import numpy as np

# Create an empty index
index = VectorIndex()

# Create some fake embeddings (in real code, use EmbeddingGenerator)
n_samples = 1000
embeddings = np.random.randn(n_samples, 384).astype(np.float32)

# Normalize (required for cosine similarity)
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / norms

# Add to index
index.add(embeddings)

print(f"Index size: {index.size()}")  # 1000
```

### With Configuration

```python
from malvec.vector_store import VectorIndex, VectorIndexConfig

config = VectorIndexConfig(
    embedding_dim=384,  # Must match your embeddings
    metric='cosine',    # or 'l2' for Euclidean distance
    use_gpu=False,      # Set True if you have GPU
)

index = VectorIndex(config)
```

---

## 4. Searching the Index

### Find Similar Vectors

```python
# Query = first embedding (pretend it's new malware we want to classify)
query = embeddings[0:1]  # Shape (1, 384)

# Find 5 nearest neighbors
distances, indices = index.search(query, k=5)

print(f"Nearest indices: {indices[0]}")
print(f"Similarities: {distances[0]}")
```

**Output:**

```text
Nearest indices: [0 573 291 842 123]
Similarities: [1.0 0.89 0.87 0.85 0.84]
```

- Index 0 is the query itself (similarity = 1.0, perfect match)
- Next closest is index 573 (similarity = 0.89)

### Batch Search

```python
# Search for 10 queries at once
queries = embeddings[0:10]
distances, indices = index.search(queries, k=5)

# distances.shape = (10, 5)
# indices.shape = (10, 5)
```

---

## 5. Understanding Distance Metrics

### Cosine Similarity (Default)

- Range: -1 to 1 (for normalized vectors: 0 to 1)
- Higher = more similar
- 1.0 = identical

```text
A • B = cos(θ) when ||A|| = ||B|| = 1
```

### L2 (Euclidean) Distance

- Range: 0 to infinity
- Lower = more similar
- 0 = identical

```text
||A - B||² = sum((a_i - b_i)²)
```

### Which to Use?

| Situation | Use |
|-----------|-----|
| Text/embedding similarity | Cosine |
| Geometric distance | L2 |
| Default for MalVec | Cosine |

For normalized embeddings (which we use), cosine and inner product are equivalent.

---

## 6. Saving and Loading

### Save Your Index

```python
# After building index...
index.save('/path/to/my_index')

# Creates two files:
# - /path/to/my_index.index (FAISS data)
# - /path/to/my_index.config (metadata)
```

### Load Later

```python
# In a new session...
loaded_index = VectorIndex.load('/path/to/my_index')

# Ready to search immediately
distances, indices = loaded_index.search(query, k=5)
```

---

## 7. Hands-On Exercise

### Task: Build a Malware Similarity Search

```python
from malvec.ember_loader import load_ember_features
from malvec.embedder import EmbeddingGenerator
from malvec.vector_store import VectorIndex
import numpy as np

# 1. Load features and generate embeddings
X, y = load_ember_features(max_samples=500)
generator = EmbeddingGenerator()
embeddings = generator.generate(X)

print(f"Generated {len(embeddings)} embeddings of shape {embeddings.shape[1]}")

# 2. Build the index
index = VectorIndex()
index.add(embeddings)
print(f"Index contains {index.size()} vectors")

# 3. Search for similar samples
# Pick a random malware sample
query_idx = 42
query = embeddings[query_idx:query_idx+1]

distances, indices = index.search(query, k=5)

# 4. Display results
print(f"\nSamples most similar to sample {query_idx}:")
print("-" * 40)
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    label = "malware" if y[idx] == 1 else "benign"
    print(f"{i+1}. Sample {idx} ({label}) - similarity: {dist:.4f}")

# 5. Check if similar samples have same label
query_label = y[query_idx]
same_label_count = sum(1 for idx in indices[0] if y[idx] == query_label)
print(f"\n{same_label_count}/5 neighbors have same label as query")
```

**Expected output:**

```text
Generated 500 embeddings of shape 384
Index contains 500 vectors

Samples most similar to sample 42:
----------------------------------------
1. Sample 42 (malware) - similarity: 1.0000
2. Sample 127 (malware) - similarity: 0.8934
3. Sample 389 (malware) - similarity: 0.8756
4. Sample 201 (benign) - similarity: 0.8123
5. Sample 88 (malware) - similarity: 0.7989

4/5 neighbors have same label as query
```

---

## Checkpoint Questions

### Question 1

Why is FAISS faster than comparing every vector?

<details>
<summary>Answer</summary>

FAISS organizes vectors into groups/clusters. When searching:

1. It identifies which groups likely contain good matches
2. Only searches within those groups
3. Skips most of the database

This reduces comparisons from millions to thousands.

</details>

### Question 2

When should you use L2 distance vs cosine similarity?

<details>
<summary>Answer</summary>

- **Cosine similarity:** When you care about *direction* not *magnitude*
  - Best for text and semantic embeddings
  - Ignores how "strong" a feature is
  
- **L2 distance:** When you care about actual geometric distance
  - Best for coordinates, physical measurements
  - Sensitive to magnitude

For embeddings (like ours), cosine is usually better.

</details>

### Question 3

What two files does `index.save()` create?

<details>
<summary>Answer</summary>

1. `*.index` - The FAISS index data (binary, large)
2. `*.config` - Configuration metadata (JSON, small)

Both are needed to load the index later.

</details>

---

## Key Takeaways

1. **Vector databases** make similarity search fast (1000x+ speedup)
2. **FAISS** is the industry standard for vector search
3. **Cosine similarity** is best for embeddings (higher = more similar)
4. **Save/load** indexes to avoid rebuilding
5. For **malware detection**: similar embeddings often = same class

---

## Next Steps

In Lesson 5, you'll learn about **k-NN classification** - using the
neighbors we just found to actually classify malware!

---

## Further Reading

- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [FAISS Tutorial](https://github.com/facebookresearch/faiss/wiki/Getting-started)
- [Vector Databases Explained](https://www.pinecone.io/learn/vector-database/)
