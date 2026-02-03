# MalVec User Guide

This guide covers end-to-end workflows for using MalVec to classify malware.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Training a Model](#training-a-model)
4. [Classifying Files](#classifying-files)
5. [Batch Processing](#batch-processing)
6. [Model Management](#model-management)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

---

## Installation

### Prerequisites

- Python 3.11 or higher
- 8GB RAM minimum (16GB recommended)
- 10GB disk space for EMBER dataset

### Install from Source

```bash
# Clone repository
git clone https://github.com/yourusername/malvec
cd malvec

# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install MalVec in development mode
pip install -e .
```

### Optional Dependencies

```bash
# Beautiful terminal output
pip install rich

# YAML configuration support
pip install pyyaml

# GPU acceleration (if CUDA available)
pip install faiss-gpu
```

### Verify Installation

```bash
# Check import works
python -c "import malvec; print(malvec.__version__)"

# Run tests
pytest tests/unit/ -v
```

---

## Quick Start

### 1. Train a Model

```bash
# Train on 10,000 samples (quick demo)
python -m malvec.cli.train \
  --output ./model \
  --max-samples 10000 \
  --verbose
```

### 2. Classify a File

```bash
# Classify a PE binary
python -m malvec.cli.classify \
  --model ./model \
  --file suspicious.exe \
  --show-confidence
```

### 3. View Results

```
Sample: suspicious.exe
Prediction: MALWARE
Confidence: 87%
Status: [OK] AUTO-CLASSIFIED
```

---

## Training a Model

### Using EMBER Dataset

MalVec uses the EMBER dataset for training. First, download and prepare it:

```bash
# Download EMBER (requires ember package)
pip install ember

# Create vectorized features
python -c "
import ember
ember.create_vectorized_features('/data/ember2018/')
"
```

### Training Commands

```bash
# Basic training
python -m malvec.cli.train --output ./model --max-samples 10000

# Full training with all options
python -m malvec.cli.train \
  --output ./production_model \
  --max-samples 500000 \
  --k 5 \
  --threshold 0.7 \
  --embedding-dim 384 \
  --verbose
```

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | required | Output directory |
| `--max-samples, -n` | 10000 | Maximum samples to load |
| `--k` | 5 | K-NN neighbors |
| `--threshold` | 0.6 | Confidence threshold |
| `--embedding-dim` | 384 | Embedding dimension |
| `--verbose, -v` | false | Show progress |
| `--config, -c` | none | YAML config file |

### Training Output

Training creates these files in the output directory:

```
model/
├── model.index     # FAISS vector index
├── model.config    # Index configuration
├── model.labels.npy # Training labels
└── model.meta      # Model metadata
```

---

## Classifying Files

### Single File Classification

```bash
# Basic classification
python -m malvec.cli.classify \
  --model ./model \
  --file sample.exe

# With confidence score
python -m malvec.cli.classify \
  --model ./model \
  --file sample.exe \
  --show-confidence

# With nearest neighbors
python -m malvec.cli.classify \
  --model ./model \
  --file sample.exe \
  --show-neighbors

# JSON output
python -m malvec.cli.classify \
  --model ./model \
  --file sample.exe \
  --json
```

### Classification Options

| Option | Description |
|--------|-------------|
| `--model, -m` | Model directory or .malvec file |
| `--file, -f` | PE binary to classify |
| `--show-confidence, -c` | Show confidence percentage |
| `--show-neighbors, -n` | Show K nearest neighbors |
| `--k` | Override model's k value |
| `--json` | Output as JSON |

### Understanding Results

```
Sample: suspicious.exe
Prediction: MALWARE
Confidence: 87%
Status: [OK] AUTO-CLASSIFIED

Nearest Neighbors (k=5):
  #1: idx=42351, sim=0.9234, malware
  #2: idx=18923, sim=0.8912, malware
  #3: idx=55123, sim=0.8756, malware
  #4: idx=33421, sim=0.8521, malware
  #5: idx=12345, sim=0.8234, benign
```

- **Prediction**: MALWARE or BENIGN
- **Confidence**: Agreement among K neighbors (0-100%)
- **Status**: AUTO-CLASSIFIED (confident) or NEEDS REVIEW (uncertain)
- **Neighbors**: Similar samples from training data

### Handling "Needs Review" Results

When confidence is below threshold, manual review is recommended:

```
Status: [!] NEEDS REVIEW
```

This means:
1. The K neighbors don't strongly agree
2. The sample may be novel or ambiguous
3. Human analyst should examine the file

---

## Batch Processing

### Process Directory

```bash
# Process all PE files in directory
python -m malvec.cli.batch \
  --model ./model \
  --input-dir ./samples \
  --output results.csv

# Recursive search
python -m malvec.cli.batch \
  --model ./model \
  --input-dir ./samples \
  --output results.csv \
  --recursive
```

### Output Format

CSV output:

```csv
file,sha256,prediction,confidence,needs_review,processing_time_ms
sample1.exe,abc123...,MALWARE,0.95,false,234
sample2.exe,def456...,BENIGN,0.72,true,189
```

### Batch Options

| Option | Description |
|--------|-------------|
| `--input-dir` | Directory with samples |
| `--output` | Output CSV file |
| `--recursive` | Search subdirectories |
| `--parallel` | Number of workers |
| `--timeout` | Per-file timeout |

---

## Model Management

### Create Distributable Archive

```bash
# Create .malvec archive
python -m malvec.cli.archive create \
  --model ./model \
  --output malvec_v1.malvec \
  --compression gz
```

### Inspect Archive

```bash
# Show metadata without extracting
python -m malvec.cli.archive inspect malvec_v1.malvec
```

Output:
```
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Property       ┃ Value      ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ k              │ 5          │
│ threshold      │ 0.6        │
│ embedding_dim  │ 384        │
│ n_samples      │ 100000     │
│ n_benign       │ 50000      │
│ n_malware      │ 50000      │
│ version        │ 1.0        │
└────────────────┴────────────┘
```

### Extract Archive

```bash
# Extract to directory
python -m malvec.cli.archive extract \
  malvec_v1.malvec \
  --output ./extracted_model
```

### Using Archives Directly

```bash
# Classify with archive (auto-extracts)
python -m malvec.cli.classify \
  --model malvec_v1.malvec \
  --file sample.exe
```

---

## Configuration

### YAML Configuration File

Create `malvec.yaml`:

```yaml
# Model settings
model_path: ./production_model
embedding_dim: 384
k: 5
confidence_threshold: 0.7

# Security settings
sandbox_enabled: true
timeout: 30
max_memory: 536870912  # 512MB
max_filesize: 52428800  # 50MB

# Logging
audit_log: /var/log/malvec/audit.log
verbose: false
```

### Use Configuration File

```bash
python -m malvec.cli.train --config malvec.yaml --output ./model
```

### Environment Variables

All settings can be overridden via environment:

```bash
export MALVEC_MODEL_PATH=./model
export MALVEC_K=7
export MALVEC_TIMEOUT=60
export MALVEC_VERBOSE=true
```

### Configuration Priority

1. Command-line arguments (highest)
2. Environment variables
3. Configuration file
4. Default values (lowest)

---

## Troubleshooting

### Common Issues

#### "EMBER features not found"

```
Error: EMBER dataset not found at /data/ember2018/
Hint: Download EMBER first: pip install ember
```

**Solution**: Download and prepare EMBER dataset:
```bash
pip install ember
python -c "import ember; ember.create_vectorized_features('/data/ember2018/')"
```

#### "Model not found"

```
Error: Model not found at ./model
Hint: Train a model first: python -m malvec.cli.train --output ./model
```

**Solution**: Train or specify correct model path.

#### "Feature extraction timed out"

```
Error: Feature extraction exceeded 30s timeout
Hint: File may be malformed or excessively large
```

**Solution**: Increase timeout or check file:
```bash
python -m malvec.cli.classify --model ./model --file sample.exe --timeout 60
```

#### "Invalid file format"

```
Error: Invalid file suspicious.txt: Not a PE file (bad magic bytes)
Hint: Ensure file is a valid PE binary and < 50MB
```

**Solution**: MalVec only supports PE files with MZ header.

### Viewing Audit Logs

```bash
# View recent classifications
tail -f /var/log/malvec/audit.log | jq .

# Find all malware detections
grep '"prediction": "MALWARE"' /var/log/malvec/audit.log

# Find sandbox violations
grep "sandbox_violation" /var/log/malvec/audit.log
```

### Debug Mode

```bash
# Enable verbose output
python -m malvec.cli.classify \
  --model ./model \
  --file sample.exe \
  --verbose

# Python logging
export MALVEC_LOG_LEVEL=DEBUG
python -m malvec.cli.classify ...
```

---

## FAQ

### What file types are supported?

MalVec supports PE (Portable Executable) files:
- `.exe` - Windows executables
- `.dll` - Dynamic link libraries
- `.sys` - Windows drivers

ELF support is planned for future releases.

### How accurate is MalVec?

On the EMBER 2018 dataset:
- Accuracy: ~92%
- Precision: ~91%
- Recall: ~89%
- F1 Score: ~90%

Real-world accuracy depends on training data quality and similarity to test samples.

### Can MalVec detect zero-day malware?

MalVec excels at detecting variants of known malware families. For truly novel malware, it may flag for manual review. The embedding approach catches polymorphic variants better than signature-based detection.

### Is it safe to analyze malware?

MalVec uses static analysis only - malware is never executed. Additional security layers:
- Process isolation (separate subprocess)
- Sandboxing (timeout, memory limits)
- Audit logging (all operations tracked)

For maximum safety, run in an isolated VM.

### How do I improve accuracy?

1. **More training data**: Use larger EMBER subsets
2. **Tune k**: Try k=3, 5, 7, 9
3. **Adjust threshold**: Higher threshold = more "needs review"
4. **Fine-tune embeddings**: Experiment with embedding dimensions

### Can I use GPU acceleration?

Yes, install faiss-gpu:
```bash
pip install faiss-gpu
```

This accelerates the K-NN search for large models.

### How do I contribute training data?

Contact the maintainers. Community contributions should:
- Be properly labeled
- Not contain PII
- Come from legal sources

---

## Next Steps

- [Security Guide](SECURITY.md) - Understand security controls
- [Deployment Guide](DEPLOYMENT.md) - Production deployment
- [API Reference](API.md) - Python API documentation
