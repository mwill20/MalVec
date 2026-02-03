"""Shared test fixtures for MalVec test suite."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a clean temporary directory for each test."""
    return tmp_path


@pytest.fixture
def sample_features():
    """Generate synthetic EMBER-like feature vectors.

    Returns a (10, 2381) array simulating EMBER feature extraction output.
    Uses a fixed random seed for reproducibility.
    """
    rng = np.random.RandomState(42)
    return rng.rand(10, 2381).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Generate synthetic labels matching sample_features.

    Returns 10 labels: 6 malware (1), 4 benign (0).
    """
    return np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)


@pytest.fixture
def sample_embeddings():
    """Generate synthetic 384-dimensional embeddings.

    Returns a (10, 384) array simulating embedding generator output.
    Uses a fixed random seed for reproducibility.
    """
    rng = np.random.RandomState(42)
    return rng.rand(10, 384).astype(np.float32)


@pytest.fixture
def minimal_pe_header():
    """Generate minimal valid PE header bytes for testing.

    This is the smallest valid PE signature (MZ header + PE offset)
    that will pass magic byte validation without being a real executable.
    """
    # MZ header
    header = bytearray(256)
    header[0:2] = b'MZ'
    # PE offset at 0x3C pointing to 0x80
    header[0x3C] = 0x80
    # PE signature at 0x80
    header[0x80:0x84] = b'PE\x00\x00'
    return bytes(header)


@pytest.fixture
def model_dir(tmp_path, sample_embeddings, sample_labels):
    """Create a minimal model directory with index, labels, and metadata.

    Provides a ready-to-use model directory for testing classification
    and archive operations.
    """
    import json

    model_path = tmp_path / "model"
    model_path.mkdir()

    # Save labels
    np.save(model_path / "model.labels.npy", sample_labels)

    # Create a simple FAISS index
    try:
        import faiss
        dim = sample_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(sample_embeddings)
        faiss.write_index(index, str(model_path / "model.index"))
    except ImportError:
        # If FAISS not available, create a placeholder
        (model_path / "model.index").write_bytes(b"FAISS_PLACEHOLDER")

    # Create metadata
    meta = {
        "version": "0.1.0",
        "n_samples": len(sample_labels),
        "embedding_dim": 384,
        "k": 5,
        "mode": "ember",
        "created": "2026-01-01T00:00:00",
    }
    (model_path / "model.meta").write_text(json.dumps(meta, indent=2))

    # Create config
    config = {
        "k": 5,
        "confidence_threshold": 0.6,
        "embedding_dim": 384,
    }
    (model_path / "model.config").write_text(json.dumps(config, indent=2))

    return model_path
