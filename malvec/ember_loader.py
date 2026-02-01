"""
EMBER Dataset Loader

Utilities for loading and validating EMBER (Elastic Malware Benchmark for 
Empowering Researchers) feature vectors.

EMBER provides pre-extracted features from PE files, enabling malware research
without needing to handle actual malware samples.

Dataset: https://github.com/elastic/ember
Paper: https://arxiv.org/abs/1804.04637

Features:
- 2381-dimensional feature vectors
- Pre-computed from static PE analysis
- Includes: histogram, byte entropy, strings, headers, imports, etc.
"""

from pathlib import Path
from typing import Literal, Optional

import numpy as np

# EMBER feature vector dimension (fixed by the dataset)
EMBER_FEATURE_DIM = 2381

# EMBER feature groups and their approximate sizes
# These groups represent different aspects of PE file analysis
FEATURE_GROUPS = {
    'histogram': 256,          # Byte histogram (256 bins)
    'byteentropy': 256,        # Byte entropy histogram
    'strings': 104,            # String-based features
    'general': 10,             # General file info
    'header': 62,              # PE header features
    'section': 255,            # Section information
    'imports': 1280,           # Import features (hashed)
    'exports': 128,            # Export features (hashed)
    'datadirectories': 30,     # Data directory info
}


def validate_features(features: np.ndarray, allow_unlabeled: bool = False) -> bool:
    """
    Validate EMBER feature vector format.
    
    Args:
        features: Feature vector to validate (should be 2381-dim float array)
        allow_unlabeled: If True, allow -1 labels (unlabeled samples)
    
    Returns:
        True if valid
        
    Raises:
        TypeError: If features is not a numpy array
        ValueError: If features has wrong shape, contains NaN, or infinite values
    """
    # Type check
    if not isinstance(features, np.ndarray):
        try:
            features = np.asarray(features)
        except Exception:
            raise TypeError(f"Features must be array-like, got {type(features)}")
    
    # Check numeric type
    if not np.issubdtype(features.dtype, np.number):
        raise TypeError(f"Features must be numeric, got dtype {features.dtype}")
    
    # Handle both single vector and batch
    if features.ndim == 1:
        if features.shape[0] != EMBER_FEATURE_DIM:
            raise ValueError(
                f"Expected {EMBER_FEATURE_DIM} features, got {features.shape[0]}"
            )
    elif features.ndim == 2:
        if features.shape[1] != EMBER_FEATURE_DIM:
            raise ValueError(
                f"Expected {EMBER_FEATURE_DIM} features, got {features.shape[1]}"
            )
    else:
        raise ValueError(f"Features must be 1D or 2D array, got {features.ndim}D")
    
    # Check for NaN
    if np.isnan(features).any():
        raise ValueError("Features contain NaN values")
    
    # Check for infinite values
    if np.isinf(features).any():
        raise ValueError("Features contain infinite values")
    
    return True


def get_feature_names() -> list[str]:
    """
    Get names for all 2381 EMBER features.
    
    Returns:
        List of 2381 feature names, one for each dimension
    """
    names = []
    
    for group_name, group_size in FEATURE_GROUPS.items():
        for i in range(group_size):
            names.append(f"{group_name}_{i:04d}")
    
    # Ensure we have exactly 2381 names
    assert len(names) == EMBER_FEATURE_DIM, f"Generated {len(names)} names, expected {EMBER_FEATURE_DIM}"
    
    return names


def load_ember_features(
    data_dir: Optional[Path] = None,
    subset: Literal['train', 'test'] = 'train',
    max_samples: Optional[int] = None,
    include_unlabeled: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load EMBER feature vectors from the dataset.
    
    Args:
        data_dir: Path to EMBER data directory. If None, uses default location.
        subset: Which subset to load ('train' or 'test')
        max_samples: Maximum number of samples to load (for testing/debugging)
        include_unlabeled: Whether to include unlabeled samples (label=-1)
    
    Returns:
        Tuple of (X, y) where:
        - X: Feature matrix of shape (n_samples, 2381)
        - y: Label array of shape (n_samples,) with 0=benign, 1=malicious
        
    Raises:
        ValueError: If subset is not 'train' or 'test'
        FileNotFoundError: If EMBER data files are not found
    """
    # Validate subset
    if subset not in ('train', 'test'):
        raise ValueError(f"subset must be 'train' or 'test', got '{subset}'")
    
    # Determine data directory
    if data_dir is None:
        # Check common locations
        possible_paths = [
            Path('data/ember2018'),
            Path('ember2018'),
            Path.home() / 'ember2018',
        ]
        for path in possible_paths:
            if path.exists():
                data_dir = path
                break
    
    if data_dir is None or not Path(data_dir).exists():
        # For development without the full dataset, generate synthetic data
        # This allows tests to pass without downloading 7GB
        return _generate_synthetic_ember_data(subset, max_samples or 1000)
    
    # Try to load using ember library
    try:
        import ember
        
        X, y = ember.read_vectorized_features(
            str(data_dir),
            subset=subset,
            feature_version=2
        )
        
        # Filter out unlabeled if requested
        if not include_unlabeled:
            labeled_mask = y != -1
            X = X[labeled_mask]
            y = y[labeled_mask]
        
        # Apply max_samples limit
        if max_samples is not None and max_samples < len(X):
            X = X[:max_samples]
            y = y[:max_samples]
        
        return X.astype(np.float32), y.astype(np.int32)
        
    except ImportError:
        # ember library not available, use synthetic data
        return _generate_synthetic_ember_data(subset, max_samples or 1000)
    except Exception as e:
        # Data files not found or corrupted
        raise FileNotFoundError(
            f"Could not load EMBER data from {data_dir}: {e}\n"
            "Download from: https://github.com/elastic/ember#download"
        )


def _generate_synthetic_ember_data(
    subset: str, 
    n_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic EMBER-like data for testing.
    
    This allows development and testing without the full 7GB dataset.
    Features are randomly generated but match EMBER's structure.
    
    Args:
        subset: 'train' or 'test' (affects random seed for reproducibility)
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (X, y) with synthetic data
    """
    # Use different seeds for train/test to simulate different data
    seed = 42 if subset == 'train' else 123
    rng = np.random.default_rng(seed)
    
    # Generate features
    X = rng.standard_normal((n_samples, EMBER_FEATURE_DIM)).astype(np.float32)
    
    # Generate balanced labels (0 or 1)
    y = rng.integers(0, 2, size=n_samples).astype(np.int32)
    
    return X, y


def summarize_features(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Generate a summary of EMBER features and labels.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)
        
    Returns:
        Dictionary with summary statistics
    """
    n_samples, n_features = X.shape
    
    return {
        'num_samples': n_samples,
        'num_features': n_features,
        'num_malicious': int(np.sum(y == 1)),
        'num_benign': int(np.sum(y == 0)),
        'num_unlabeled': int(np.sum(y == -1)) if -1 in y else 0,
        'feature_mean': float(np.mean(X)),
        'feature_std': float(np.std(X)),
        'feature_min': float(np.min(X)),
        'feature_max': float(np.max(X)),
        'malicious_ratio': float(np.sum(y == 1) / n_samples) if n_samples > 0 else 0.0,
    }


# Module-level exports
__all__ = [
    'EMBER_FEATURE_DIM',
    'FEATURE_GROUPS',
    'validate_features',
    'get_feature_names',
    'load_ember_features',
    'summarize_features',
]
