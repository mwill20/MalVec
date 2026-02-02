# Lesson 5: K-NN Classification

> **Phase:** 5 - Classification  
> **Track:** Novice  
> **Time:** 45-60 minutes

---

## Learning Objectives

By the end of this lesson, you will:

1. Understand how K-Nearest Neighbors classification works
2. Use majority voting to classify malware
3. Interpret confidence scores
4. Know when samples need manual review

---

## 1. What Is K-NN Classification?

### The Core Idea

K-NN is simple: **Find the k most similar samples, then vote.**

```text
Query: "Is this malware?"

Step 1: Find 5 nearest neighbors in embedding space
Step 2: Check their labels (e.g., 3 malware, 2 benign)
Step 3: Majority vote → malware (3/5)
```

### Why K-NN for Malware?

- **No training required** - just store known samples
- **Explainable** - "Similar to these 5 samples"
- **Works with few samples** - new malware families
- **Updates easily** - add new samples without retraining

---

## 2. Using the Classifier

### Basic Usage

```python
from malvec.ember_loader import load_ember_features
from malvec.embedder import EmbeddingGenerator
from malvec.classifier import KNNClassifier

# Load and prepare data
X, y = load_ember_features(max_samples=1000)
generator = EmbeddingGenerator()
embeddings = generator.generate(X)

# Train classifier
clf = KNNClassifier()
clf.fit(embeddings, y)

# Predict on new sample
query = embeddings[0:1]  # In practice, this would be a new sample
prediction = clf.predict(query)
print(f"Prediction: {'Malware' if prediction[0] == 1 else 'Benign'}")
```

### With Configuration

```python
from malvec.classifier import KNNClassifier, ClassifierConfig

config = ClassifierConfig(
    k=5,                      # Number of neighbors
    vote_method='majority',   # or 'weighted'
    confidence_threshold=0.7  # Below this → manual review
)

clf = KNNClassifier(config)
```

---

## 3. Understanding Voting

### Majority Voting

Count how many neighbors have each label:

```text
Neighbors: [malware, malware, benign, malware, benign]
Count:     malware=3, benign=2
Winner:    malware (3 > 2)
```

### Weighted Voting

Weight each vote by similarity:

```text
Neighbors:    [malware, malware, benign, malware, benign]
Similarities: [0.95,    0.90,    0.85,   0.75,    0.70]

malware score = 0.95 + 0.90 + 0.75 = 2.60
benign score  = 0.85 + 0.70 = 1.55

Winner: malware (2.60 > 1.55)
```

Weighted voting gives more importance to closer neighbors.

---

## 4. Confidence Scoring

### What Is Confidence?

Confidence = fraction of neighbors agreeing with the prediction.

```python
# Get prediction with confidence
predictions, confidences = clf.predict(query, return_confidence=True)

print(f"Prediction: {predictions[0]}")
print(f"Confidence: {confidences[0]:.0%}")
```

**Example outputs:**

| Neighbors | Prediction | Confidence |
|-----------|------------|------------|
| 5/5 malware | malware | 100% |
| 3/5 malware | malware | 60% |
| 1/5 malware | benign | 80% |

### Interpreting Confidence

| Confidence | Meaning |
|------------|---------|
| 100% | All neighbors agree - high trust |
| 80%+ | Strong agreement - probably correct |
| 60-80% | Some disagreement - review if critical |
| 50-60% | Nearly split - needs human review |

---

## 5. Manual Review Flagging

### Why Flag for Review?

When confidence is low, we shouldn't trust the prediction blindly.

```python
# Get prediction with review flag
result = clf.predict_with_review(query)

print(f"Prediction: {result['predictions'][0]}")
print(f"Confidence: {result['confidences'][0]:.0%}")
print(f"Needs Review: {result['needs_review'][0]}")
```

### The Threshold

Samples with confidence below threshold get flagged:

```python
# Default threshold is 0.6 (60%)
config = ClassifierConfig(confidence_threshold=0.7)
```

| Confidence | Threshold=0.6 | Threshold=0.7 |
|------------|---------------|---------------|
| 50% | ⚠️ Review | ⚠️ Review |
| 65% | ✅ Auto | ⚠️ Review |
| 80% | ✅ Auto | ✅ Auto |

---

## 6. Hands-On Exercise

### Task: Classify and Analyze

```python
from malvec.ember_loader import load_ember_features
from malvec.embedder import EmbeddingGenerator
from malvec.classifier import KNNClassifier, ClassifierConfig
import numpy as np

# Load data
X, y = load_ember_features(max_samples=500)
generator = EmbeddingGenerator()
embeddings = generator.generate(X)

# Train with specific k
config = ClassifierConfig(k=5, confidence_threshold=0.7)
clf = KNNClassifier(config)
clf.fit(embeddings, y)

# Predict all samples
results = clf.predict_with_review(embeddings)

# Analyze results
predictions = results['predictions']
confidences = results['confidences']
needs_review = results['needs_review']

print("Classification Summary:")
print("-" * 40)

# Accuracy
accuracy = (predictions == y).mean()
print(f"Accuracy: {accuracy:.1%}")

# Review rate
review_rate = needs_review.mean()
print(f"Needs Review: {review_rate:.1%}")

# Confidence distribution
print(f"\nConfidence Distribution:")
print(f"  High (>80%): {(confidences > 0.8).sum()}")
print(f"  Medium (60-80%): {((confidences >= 0.6) & (confidences <= 0.8)).sum()}")
print(f"  Low (<60%): {(confidences < 0.6).sum()}")

# Find hardest sample
min_conf_idx = np.argmin(confidences)
print(f"\nHardest sample: #{min_conf_idx}")
print(f"  Confidence: {confidences[min_conf_idx]:.1%}")
print(f"  True label: {'malware' if y[min_conf_idx] == 1 else 'benign'}")
```

---

## 7. Checkpoint Questions

### Question 1

With k=5 and neighbors [1,1,0,0,1], what would majority voting predict?

<details>
<summary>Answer</summary>

**Malware (1).**

Count: 3 malware, 2 benign → majority is malware.

</details>

### Question 2

Why might you choose weighted voting over majority voting?

<details>
<summary>Answer</summary>

Weighted voting gives more importance to **closer neighbors**.

If a very similar sample (0.95) disagrees with a distant sample (0.50),
the close one matters more.

Use weighted when similarity scores vary widely.

</details>

### Question 3

What confidence would you get with 4/5 neighbors agreeing?

<details>
<summary>Answer</summary>

**80%** (4 ÷ 5 = 0.80)

</details>

---

## 8. Key Takeaways

1. **K-NN classification** = find k neighbors, majority vote
2. **Confidence** = fraction of neighbors agreeing
3. **Low confidence** → flag for human review
4. **Weighted voting** prioritizes closer neighbors
5. Choose **k** wisely (too small = noise, too large = blur)

---

## What's Next?

In the next lessons, you'll learn about:

- Building the command-line interface (CLI)
- Creating a training pipeline
- Deploying for production use

---

## Further Reading

- [K-NN Classification Explained](https://scikit-learn.org/stable/modules/neighbors.html)
- [Choosing k for K-NN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm#Parameter_selection)
- [Voting Methods](https://en.wikipedia.org/wiki/Weighted_voting)
