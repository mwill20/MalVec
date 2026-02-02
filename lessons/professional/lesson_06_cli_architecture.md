# Lesson 6: CLI Architecture & Model Persistence

**Phase:** 6 - CLI & Training Pipeline  
**Prerequisites:** Lessons 1-5, Python CLI patterns  
**Objective:** Understand CLI architecture, model serialization, and evaluation strategies

---

## Architecture Overview

MalVec's CLI follows a modular architecture:

```
malvec/cli/
├── __init__.py       # Module marker
├── train.py          # Model training
├── classify.py       # Single-sample classification
├── evaluate.py       # Performance metrics
├── batch.py          # Multi-sample processing
└── info.py           # Model introspection
```

Each command is an independent module runnable via `python -m`:

```bash
python -m malvec.cli.train --help
python -m malvec.cli.classify --help
```

---

## Model Persistence Strategy

### File Structure

A trained MalVec model consists of four files:

```
model_dir/
├── model.index       # FAISS vector index (binary)
├── model.config      # Index configuration (JSON)
├── model.labels.npy  # Label array (NumPy binary)
└── model.meta        # Training metadata (JSON)
```

### Why Multiple Files?

| File | Format | Reason |
|------|--------|--------|
| `.index` | Binary | FAISS native format, memory-mappable |
| `.config` | JSON | Human-readable, editable |
| `.labels.npy` | NumPy | Fast loading, native format |
| `.meta` | JSON | Training provenance, parameters |

This separation enables:

- **Index replacement** without retraining
- **Label updates** without re-embedding
- **Metadata inspection** without loading model
- **Parallel loading** for large models

### Serialization Code

```python
def save_model(clf, output_path: Path):
    """Save classifier to disk."""
    model_prefix = str(output_path / 'model')
    
    # 1. Save FAISS index
    clf._index.save(model_prefix)
    # Creates: model.index, model.config
    
    # 2. Save labels (NumPy format)
    np.save(f"{model_prefix}.labels", clf._labels)
    # Creates: model.labels.npy
    
    # 3. Save metadata (JSON)
    meta = {
        'k': clf.config.k,
        'threshold': clf.config.confidence_threshold,
        'embedding_dim': clf.config.embedding_dim,
        'n_samples': len(clf._labels),
        'version': '1.0',
    }
    with open(f"{model_prefix}.meta", 'w') as f:
        json.dump(meta, f, indent=2)
```

### Deserialization Code

```python
def load_model(model_path: Path):
    """Load classifier from disk."""
    model_prefix = str(model_path / 'model')
    
    # 1. Load metadata first (fast, gives config)
    with open(f"{model_prefix}.meta") as f:
        meta = json.load(f)
    
    # 2. Load FAISS index
    index = VectorIndex.load(model_prefix)
    
    # 3. Load labels
    labels = np.load(f"{model_prefix}.labels.npy")
    
    # 4. Reconstruct classifier
    config = ClassifierConfig(
        k=meta['k'],
        confidence_threshold=meta['threshold'],
        embedding_dim=meta['embedding_dim']
    )
    clf = KNNClassifier(config)
    clf._index = index
    clf._labels = labels
    
    return clf, meta
```

---

## Evaluation Metrics Deep Dive

### Confusion Matrix

For binary classification (malware vs benign):

```
                 Predicted
                 Benign    Malware
Actual Benign      TN        FP
Actual Malware     FN        TP
```

| Metric | Formula | Meaning |
|--------|---------|---------|
| **True Positive (TP)** | Malware correctly flagged | Success |
| **True Negative (TN)** | Benign correctly cleared | Success |
| **False Positive (FP)** | Benign flagged as malware | Alert fatigue |
| **False Negative (FN)** | Malware missed | Security risk |

### Derived Metrics

```python
def compute_metrics(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    # Accuracy: Overall correctness
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Precision: Of predicted malware, how many are actually malware?
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Recall: Of actual malware, how many did we detect?
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1: Harmonic mean of precision and recall
    f1 = 2 * precision * recall / (precision + recall)
    
    return accuracy, precision, recall, f1
```

### Metric Selection for Security

| Goal | Prioritize | Why |
|------|------------|-----|
| **Security-first** | Recall | Don't miss malware (FN is dangerous) |
| **SOC efficiency** | Precision | Reduce false alarms (FP causes fatigue) |
| **Balanced** | F1 | Trade-off between both |

For malware detection, typically:

- **Recall > 95%** is critical (miss nothing)
- **Precision > 80%** is acceptable (some false positives OK)
- **Review mechanism** catches edge cases

---

## Confidence-Based Triage

### The Review Workflow

```python
def predict_with_review(self, query):
    predictions, confidences = self.predict(query, return_confidence=True)
    needs_review = confidences < self.config.confidence_threshold
    
    return {
        'predictions': predictions,
        'confidences': confidences,
        'needs_review': needs_review,
    }
```

### Threshold Selection

| Threshold | Review Rate | Auto-classify Rate | Use Case |
|-----------|-------------|-------------------|----------|
| 0.9 | ~80% | ~20% | High-security, human-heavy |
| 0.7 | ~40% | ~60% | Balanced |
| 0.6 | ~20% | ~80% | High throughput |
| 0.5 | ~5% | ~95% | Automated pipeline |

### High-Confidence vs. Low-Confidence Accuracy

```python
# Accuracy on samples we auto-classify
high_conf_mask = ~needs_review
high_conf_accuracy = (y_true[high_conf_mask] == y_pred[high_conf_mask]).mean()

# Accuracy on samples we flag for review
low_conf_mask = needs_review
low_conf_accuracy = (y_true[low_conf_mask] == y_pred[low_conf_mask]).mean()
```

**Expectation:**

- High-confidence accuracy: 85-95%
- Low-confidence accuracy: 55-65%

This validates that the confidence score is meaningful.

---

## CLI Design Patterns

### Argument Parsing

```python
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train MalVec classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m malvec.cli.train --output ./model --max-samples 1000
        """
    )
    
    # Required arguments
    parser.add_argument('--output', '-o', required=True, help='...')
    
    # Optional with defaults
    parser.add_argument('--k', type=int, default=5, help='...')
    
    # Flags
    parser.add_argument('--verbose', '-v', action='store_true')
    
    return parser.parse_args()
```

### Error Handling

```python
def main():
    args = parse_args()
    
    try:
        clf, meta = load_model(Path(args.model))
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1  # Non-zero exit code
    
    # ... rest of logic ...
    return 0  # Success
```

### Output Formats

Support both human-readable and machine-parseable output:

```python
if args.json:
    print(json.dumps(result, indent=2))
else:
    print(f"Accuracy: {result['accuracy']:.1%}")
```

---

## Performance Considerations

### Lazy Loading

```python
# Only load EMBER when needed
from malvec.ember_loader import load_ember_features

# Not at module level - avoids startup cost
```

### Batch Processing

```python
# Good: Single embedding generation call
embeddings = generator.generate(X)  # All at once

# Bad: Loop over samples
for sample in X:
    embedding = generator.generate(sample)  # N calls
```

### Memory Mapping

For very large indices:

```python
# FAISS supports memory-mapped indices
index = faiss.read_index(path, faiss.IO_FLAG_MMAP)
```

This loads index metadata but reads vectors on-demand.

---

## Testing CLI Commands

### Subprocess-Based Testing

```python
def test_train_creates_model(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [sys.executable, '-m', 'malvec.cli.train',
             '--output', tmpdir, '--max-samples', '100'],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert os.path.exists(os.path.join(tmpdir, 'model.index'))
```

### Why Subprocess?

1. **Isolation**: Tests actual CLI invocation
2. **Coverage**: Includes argument parsing
3. **Realism**: Matches user experience
4. **Failure modes**: Catches import errors

---

## Production Considerations

### Model Versioning

```json
{
  "version": "1.0",
  "created_at": "2026-02-01T21:00:00Z",
  "ember_version": "2018",
  "malvec_version": "0.1.0",
  "training_samples": 600000
}
```

### Model Registry

```python
class ModelRegistry:
    def __init__(self, base_path: Path):
        self.base = base_path
    
    def save(self, clf, name: str, version: str):
        path = self.base / name / version
        # ...
    
    def load(self, name: str, version: str = "latest"):
        # ...
    
    def list_versions(self, name: str):
        # ...
```

### A/B Testing

```python
# Load multiple models
model_a = load_model("model_v1")
model_b = load_model("model_v2")

# Compare predictions
pred_a = model_a.predict(embeddings)
pred_b = model_b.predict(embeddings)

# Measure agreement
agreement = (pred_a == pred_b).mean()
```

---

## Summary

### CLI Command Structure

| Command | Input | Output |
|---------|-------|--------|
| `train` | EMBER data | Model directory |
| `classify` | Model + sample ID | Prediction + confidence |
| `evaluate` | Model + test data | Metrics report |
| `batch` | Model + samples | CSV/JSON results |
| `info` | Model | Statistics |

### Key Design Decisions

1. **Multi-file models**: Separation of concerns
2. **JSON metadata**: Human-readable provenance
3. **NumPy labels**: Fast loading, native format
4. **Confidence-based triage**: Human-in-the-loop
5. **Subprocess testing**: Realistic CLI validation

### What's Next

Phase 7 will add:

- Input validation for binary files
- Static feature extraction
- End-to-end binary → classification pipeline
