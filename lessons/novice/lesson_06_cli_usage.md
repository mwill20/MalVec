# Lesson 6: Command-Line Interface

**Phase:** 6 - CLI & Training Pipeline  
**Prerequisites:** Lessons 1-5 completed  
**Objective:** Learn to use MalVec's command-line interface for training and classification

---

## What You'll Learn

In this lesson, you'll learn how to:

- Train a malware detection model from the command line
- Classify samples and interpret results
- Evaluate model performance
- Export results for analysis

---

## The MalVec CLI

MalVec provides a complete command-line interface for malware classification:

| Command | Purpose |
|---------|---------|
| `train` | Build a model from EMBER data |
| `classify` | Classify a single sample |
| `evaluate` | Test model performance |
| `batch` | Classify many samples |
| `info` | Show model details |

---

## Exercise 1: Train Your First Model

Open your terminal and navigate to the MalVec project:

```bash
cd C:\Projects\MalVec
.\venv\Scripts\activate
```

Train a model with 500 samples:

```bash
python -m malvec.cli.train --output ./my_model --max-samples 500 --verbose
```

**Expected Output:**

```
MalVec Training
===============
Output: my_model
Max samples: 500
k: 5

Loading EMBER features...
  Loaded 500 samples
  Labels: 254 benign, 246 malware

Generating embeddings...
  Generated 500 embeddings
  Dimension: 384

Building classifier...
  k=5, threshold=0.6

Saving model...
  Saved to my_model

Training complete: 500 samples indexed
Model saved to: my_model
```

---

## Exercise 2: View Model Information

See what was created:

```bash
python -m malvec.cli.info --model ./my_model
```

**Expected Output:**

```
MalVec Model Info
=================
Path: C:\Projects\MalVec\my_model
Version: 1.0

Samples:
  Total: 500
  Benign: 254
  Malware: 246

Configuration:
  k: 5
  Threshold: 0.6
  Embedding Dim: 384
  Metric: cosine

Files:
  index: 767,245 bytes
  config: 74 bytes
  labels: 2,128 bytes
  meta: 138 bytes
  Total: 0.73 MB
```

**Understanding the Output:**

- **Samples**: How many malware/benign files the model knows
- **k**: Number of neighbors used for voting
- **Threshold**: Confidence level for auto-classification
- **Metric**: How similarity is measured (cosine = angle between vectors)

---

## Exercise 3: Classify a Sample

Classify sample #42 with full details:

```bash
python -m malvec.cli.classify --model ./my_model --sample-index 42 --show-confidence --show-neighbors
```

**Expected Output:**

```
Sample #42
Prediction: BENIGN
Confidence: 80%
Status: [OK] AUTO-CLASSIFIED
True Label: benign
Correct: YES

Nearest Neighbors (k=5):
  #1: idx=42, sim=1.0000, benign
  #2: idx=16, sim=0.1446, malware
  #3: idx=166, sim=0.1436, benign
  #4: idx=71, sim=0.1398, benign
  #5: idx=162, sim=0.1396, benign
```

**What This Means:**

- **Prediction**: Model's guess (BENIGN or MALWARE)
- **Confidence**: How sure the model is (80% = 4/5 neighbors agree)
- **Status**: Whether it needs human review
- **Neighbors**: The 5 most similar known samples

Notice:

- Neighbor #1 is itself (similarity = 1.0, perfect match)
- 4 of 5 neighbors are benign → 80% confidence → benign prediction

---

## Exercise 4: Evaluate Model Performance

Check how well your model performs:

```bash
python -m malvec.cli.evaluate --model ./my_model --max-samples 500
```

**Expected Output:**

```
MalVec Model Evaluation
=======================
Model: my_model
Samples: 500

Performance Metrics:
  Accuracy:  68.8%
  Precision: 68.8%
  Recall:    67.1%
  F1 Score:  67.9%

Confusion Matrix:
                Predicted
                Benign  Malware
  Actual Benign   179       75
  Actual Malware   81      165

Review Status:
  Needs Review: 302/500 (60.4%)
  Auto-classified: 198/500
  High-confidence Accuracy: 75.3%
```

**Understanding Metrics:**

| Metric | Meaning | Good Range |
|--------|---------|------------|
| Accuracy | Overall correct predictions | > 85% |
| Precision | "Of malware predictions, how many are correct?" | > 75% |
| Recall | "Of actual malware, how many did we find?" | > 70% |
| F1 Score | Balance of precision and recall | > 75% |

**Note:** Our 68.8% accuracy matches the expected result for synthetic data (random labels). With real EMBER data, expect 85-95%.

---

## Exercise 5: Batch Classification

Export classifications to a file:

```bash
python -m malvec.cli.batch --model ./my_model --max-samples 50 --output results.csv
```

Open `results.csv` - you'll see:

```
sample_index,prediction,confidence,needs_review,true_label,correct
0,malware,0.6,False,malware,True
1,benign,0.8,False,benign,True
2,malware,0.6,False,benign,False
...
```

For JSON output:

```bash
python -m malvec.cli.batch --model ./my_model --max-samples 10 --output results.json --format json
```

---

## Review Workflow

When samples need review, use this workflow:

```bash
# 1. Find samples needing review
python -m malvec.cli.batch --model ./my_model --max-samples 100 --output review.csv --review-only

# 2. For each flagged sample, show neighbors
python -m malvec.cli.classify --model ./my_model --sample-index 15 --show-neighbors

# 3. Human analyst makes final decision based on neighbors
```

---

## Checkpoint Questions

1. **What does k=5 mean?**
   - [ ] The model has 5 layers
   - [x] The model looks at 5 nearest neighbors to vote
   - [ ] The model has 5 types of features

2. **If confidence is 60%, what happened in voting?**
   - [x] 3 neighbors voted one way, 2 voted the other (3/5 = 60%)
   - [ ] The model is 60% accurate overall
   - [ ] The sample is 60% malware

3. **Why might a sample need review?**
   - [ ] The model crashed
   - [x] Confidence is below the threshold (neighbors disagree)
   - [ ] The file is corrupted

---

## Cleanup

Remove your test model:

```bash
Remove-Item -Recurse -Force ./my_model
Remove-Item results.csv
Remove-Item results.json
```

---

## Summary

| Command | When to Use |
|---------|-------------|
| `train` | Create a new model from labeled samples |
| `info` | Check what's in a model |
| `classify` | Investigate a single sample |
| `evaluate` | Measure overall model quality |
| `batch` | Process many samples at once |

**Next Lesson:** We'll dive into the code behind these commands and learn about model persistence.
