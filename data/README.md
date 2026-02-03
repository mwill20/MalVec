# MalVec Data Directory

This directory holds data artifacts used by MalVec. **No data is distributed with the repository** — you generate or download it locally.

## Directory Structure

```
data/
├── embeddings/    # Generated embedding vectors
├── models/        # Trained model files (.malvec archives or directories)
├── samples/       # PE files for analysis (DO NOT commit malware samples)
└── README.md      # This file
```

## Data Sources

### EMBER Dataset (Development Mode)

MalVec uses the [EMBER 2018 dataset](https://github.com/elastic/ember) for training and evaluation in development mode.

- **Format:** Pre-extracted feature vectors (2381 dimensions per sample)
- **Size:** ~4 GB (train + test sets)
- **Samples:** 1.1 million labeled PE files (600K train, 200K test, 300K unlabeled)
- **Labels:** 0 = benign, 1 = malware

**Download and setup:**

```bash
pip install git+https://github.com/elastic/ember.git
python -c "import ember; ember.create_vectorized_features('path/to/ember2018')"
```

See the [User Guide](../docs/USER_GUIDE.md) for detailed setup instructions.

### Binary Mode (Production)

In production mode, MalVec extracts features directly from PE binaries using LIEF and pefile. No pre-extracted dataset is required — just provide PE files in `data/samples/`.

## Generated Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| FAISS index | `models/model.index` | Trained vector search index |
| Labels | `models/model.labels.npy` | NumPy array of sample labels |
| Metadata | `models/model.meta` | JSON model metadata (version, params, creation date) |
| Config | `models/model.config` | JSON classification parameters (k, threshold) |
| Archive | `models/*.malvec` | Packaged model (tarball of the above) |

## Security Notes

- **Never commit malware samples** to this repository
- Sample files in `data/samples/` are gitignored by default
- Model archives (`.malvec` files) contain no malware — only mathematical representations
- All file paths in generated artifacts use SHA256 hashes, not filenames

## Synthetic Data for Testing

Tests use synthetic data generated with fixed random seeds (`numpy.random.RandomState(42)`), so no external data download is required to run the test suite.
