# Lesson 5: Classification Algorithms

> **Phase:** 5 - Classification  
> **Track:** Professional  
> **Time:** 60-90 minutes

---

## Learning Objectives

By the end of this lesson, you will:

1. Understand K-NN design tradeoffs for malware detection
2. Implement voting strategies with appropriate weighting
3. Calibrate confidence thresholds for production
4. Design human-in-the-loop review workflows

---

## 1. K-NN Architecture for Malware

### Why K-NN over Neural Networks?

| Factor | K-NN | Neural Network |
|--------|------|----------------|
| Interpretability | High (show neighbors) | Low (black box) |
| Training time | None (just index) | Hours/days |
| New samples | Add to index | Retrain |
| Few-shot learning | Natural | Requires tricks |
| Compute (inference) | O(N) or O(log N) | O(1) |
| Memory | O(Nd) | O(params) |

For malware detection:

- **Interpretability** is critical (explain why flagged)
- **New families** appear daily (easy updates)
- **Few samples** of new variants (few-shot learning)

### Our Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                      KNNClassifier                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌─────────────┐  ┌───────────────────┐  │
│  │ VectorIndex  │  │   Labels    │  │     Config        │  │
│  │ (FAISS)      │  │ (parallel)  │  │ (k, threshold)    │  │
│  └──────────────┘  └─────────────┘  └───────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  fit()        │  predict()      │  predict_with_review()   │
└─────────────────────────────────────────────────────────────┘
```

**Key design decision:** Labels stored separately from VectorIndex.

This enables:

- VectorIndex for non-classification tasks (clustering, deduplication)
- Testing vector search independently
- Cleaner separation of concerns

---

## 2. Voting Strategies

### Majority Voting

```python
def _majority_vote(self, labels: np.ndarray) -> int:
    counts = np.bincount(labels.astype(int))
    return counts.argmax()
```

**When to use:**

- When all similarities are roughly equal
- When you want simplicity
- When k is small (5-10)

### Weighted Voting

```python
def _weighted_vote(self, labels: np.ndarray, weights: np.ndarray) -> int:
    weights = np.maximum(weights, 0)  # Ensure positive
    
    unique_labels = np.unique(labels)
    class_weights = {}
    
    for label in unique_labels:
        mask = labels == label
        class_weights[int(label)] = weights[mask].sum()
    
    return max(class_weights, key=class_weights.get)
```

**When to use:**

- When similarity scores vary widely
- When closest neighbors should dominate
- For boundary cases

### Distance-Weighted Voting (Alternative)

```python
def _distance_weighted_vote(self, labels, distances, epsilon=1e-8):
    """Weight inversely by distance (for L2 metric)."""
    weights = 1.0 / (distances + epsilon)
    # ... then weighted vote
```

---

## 3. Confidence Calibration

### What Confidence Means

Our confidence = agreement rate among k neighbors:

```python
confidence = (neighbors agreeing with prediction) / k
```

**Interpretation:**

- 100% = unanimous agreement
- 60% = 3/5 agree (narrow majority)
- 50% = tie (random guess)

### Confidence vs Probability

⚠️ **Important:** Our confidence is NOT a calibrated probability.

| Confidence | Our meaning | NOT necessarily |
|------------|-------------|-----------------|
| 80% | 4/5 neighbors agree | 80% chance correct |
| 60% | 3/5 neighbors agree | 60% chance correct |

For calibrated probabilities, you'd need:

1. Temperature scaling on held-out data
2. Platt scaling (logistic regression on scores)
3. Isotonic regression

### Threshold Selection

```text
                    Threshold
                        │
  ◄── AUTO-ACCEPT ─────┼───── MANUAL REVIEW ──►
                        │
    High confidence     │    Low confidence
         ✓              │         ?
```

**Factors:**

- **Lower threshold (0.5):** More automation, more misses
- **Higher threshold (0.9):** More reviews, fewer misses
- **Production default (0.6-0.7):** Balance

**Data-driven selection:**

```python
# Find threshold that flags 90% of misclassifications
from sklearn.metrics import roc_curve

# Get confidences and correctness on validation set
confidences = clf.predict(X_val, return_confidence=True)[1]
correct = (predictions == y_val).astype(int)

# Find threshold
fpr, tpr, thresholds = roc_curve(correct, confidences)
# Choose threshold where tpr >= 0.9
idx = np.argmax(tpr >= 0.9)
optimal_threshold = thresholds[idx]
```

---

## 4. Production Considerations

### Choosing k

| k | Pros | Cons |
|---|------|------|
| 1 | Fastest, simplest | Noisy, unstable |
| 5 | Good balance | Most common default |
| 10+ | Stable | Blurs decision boundary |

**Odd vs Even:**

- Odd k avoids ties in binary classification
- Even k can tie → need tiebreaker

**Our tiebreaker:** `np.bincount().argmax()` returns lower class on tie.

### Computational Complexity

| Operation | IndexFlat | IndexIVF |
|-----------|-----------|----------|
| Add | O(1) | O(1) |
| Search | O(Nd) | O(N/P × d) |
| Memory | O(Nd) | O(Nd + overhead) |

Where:

- N = number of samples
- d = embedding dimension
- P = number of partitions

**For malware detection:**

- <1M samples → IndexFlat (exact)
- 1M+ samples → Consider IndexIVF (approximate)

### Handling Class Imbalance

Real-world malware data is often imbalanced:

```text
Reality: 95% benign, 5% malware
Problem: K-NN biased toward majority class
```

**Solutions:**

1. **Stratified sampling** - Balance training set
2. **Class weights** - Weight malware votes higher
3. **Threshold adjustment** - Lower confidence threshold for malware

```python
def _weighted_vote_with_class_weights(self, labels, weights, class_weights):
    """Weight by similarity AND class importance."""
    for label in unique_labels:
        mask = labels == label
        class_weight = class_weights.get(int(label), 1.0)
        total = weights[mask].sum() * class_weight
```

---

## 5. Human-in-the-Loop Design

### The Review Workflow

```text
    ┌──── Sample ─────┐
    │                 │
    ▼                 │
┌───────────┐         │
│ Classify  │         │
└─────┬─────┘         │
      │               │
      ▼               │
┌───────────┐    No   │
│Confidence?├─────────┘
│  ≥ 0.7    │
└─────┬─────┘
      │ Yes
      ▼
┌───────────┐
│ Auto-Act  │
└───────────┘
```

For samples needing review:

1. Show prediction + confidence
2. Show k nearest neighbors with labels
3. Analyst makes final call
4. Feed decision back to system

### Feedback Loop

```python
# Store analyst decisions
def record_decision(self, sample_hash, analyst_label, predicted_label):
    """Record analyst override for future training."""
    decision = {
        'hash': sample_hash,
        'analyst_label': analyst_label,
        'predicted_label': predicted_label,
        'was_correct': analyst_label == predicted_label,
        'timestamp': datetime.now().isoformat(),
    }
    self._decisions.append(decision)
    
    # If override, consider adding to training set
    if analyst_label != predicted_label:
        self._overrides.append(decision)
```

---

## 6. Hands-On Exercise

### Task: Production Classifier Pipeline

```python
from malvec.ember_loader import load_ember_features
from malvec.embedder import EmbeddingGenerator
from malvec.classifier import KNNClassifier, ClassifierConfig
import numpy as np

# Load data
X, y = load_ember_features(max_samples=1000)

# Split into train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Generate embeddings
generator = EmbeddingGenerator()
train_emb = generator.generate(X_train)
test_emb = generator.generate(X_test)

# Train classifier
config = ClassifierConfig(k=5, confidence_threshold=0.6)
clf = KNNClassifier(config)
clf.fit(train_emb, y_train)

# Evaluate on test set
results = clf.predict_with_review(test_emb)
predictions = results['predictions']
confidences = results['confidences']
needs_review = results['needs_review']

# Metrics
accuracy = (predictions == y_test).mean()
review_rate = needs_review.mean()

print(f"Test Accuracy: {accuracy:.1%}")
print(f"Review Rate: {review_rate:.1%}")

# Analyze review effectiveness
auto_correct = ((~needs_review) & (predictions == y_test)).sum()
auto_total = (~needs_review).sum()
auto_accuracy = auto_correct / auto_total if auto_total > 0 else 0

review_correct = (needs_review & (predictions == y_test)).sum()
review_total = needs_review.sum()
review_accuracy = review_correct / review_total if review_total > 0 else 0

print(f"\nAuto-accepted samples:")
print(f"  Count: {auto_total} ({100*auto_total/len(y_test):.1f}%)")
print(f"  Accuracy: {auto_accuracy:.1%}")

print(f"\nReview-flagged samples:")
print(f"  Count: {review_total} ({100*review_total/len(y_test):.1f}%)")
print(f"  Would-be Accuracy: {review_accuracy:.1%}")

# Find optimal threshold
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
print(f"\nThreshold Analysis:")
print("-" * 50)
for t in thresholds:
    flagged = confidences < t
    flagged_rate = flagged.mean()
    auto = ~flagged
    auto_acc = (predictions[auto] == y_test[auto]).mean() if auto.sum() > 0 else 0
    print(f"Threshold {t}: Auto-rate={1-flagged_rate:.0%}, Auto-accuracy={auto_acc:.1%}")
```

---

## 7. Checkpoint Questions

### Question 1

Why store labels separately from VectorIndex?

<details>
<summary>Answer</summary>

**Separation of Concerns:**

1. VectorIndex can be reused for clustering, deduplication
2. Labels can be updated without rebuilding index
3. Testing is cleaner (test search and classification independently)
4. Different classifiers can share the same index

</details>

### Question 2

When would weighted voting give different results than majority voting?

<details>
<summary>Answer</summary>

When similarity scores vary widely:

```text
Neighbors:    [A,    A,    B,    B,    B]
Similarities: [0.95, 0.90, 0.10, 0.08, 0.05]

Majority: B wins (3 > 2)
Weighted: A wins (1.85 > 0.23)
```

The weighted vote recognizes that the B neighbors are distant and
should count less.

</details>

### Question 3

How would you handle a 95%/5% class imbalance?

<details>
<summary>Answer</summary>

Options:

1. **Stratified sampling** - Undersample benign or oversample malware
2. **Class weights** - Weight malware votes 19x higher
3. **Lower threshold for malware** - Flag malware predictions at 50%
4. **Precision/Recall tradeoff** - Tune for high malware recall

Best practice: Combine stratified sampling with threshold tuning.

</details>

---

## Key Takeaways

1. **K-NN is ideal** for malware - interpretable, updatable, few-shot capable
2. **Weighted voting** when similarity varies
3. **Confidence ≠ probability** - it's agreement rate
4. **Threshold selection** is data-driven (ROC analysis)
5. **Human-in-the-loop** for low confidence samples

---

## Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Even k | Ties possible | Use odd k |
| Treating confidence as probability | Miscalibrated decisions | Use calibrated thresholds |
| Ignoring class imbalance | Bias to majority | Stratify or weight |
| k too large | Slow, blurry boundaries | k=5-15 usually enough |
| k=1 | Noisy, unstable | Minimum k=3 |

---

## Further Reading

- [K-NN in High Dimensions](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
- [Approximate Nearest Neighbor Search](https://ann-benchmarks.com/)
- [Confidence Calibration for ML](https://arxiv.org/abs/1706.04599)
