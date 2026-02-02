"""
Feature Extraction Utilities for MalVec.

This module provides a wrapper around the EMBER feature extraction logic.
It handles the dependency issues (specifically lightgbm) gracefully by mocking it if missing.
"""

import sys
import types
from pathlib import Path
import numpy as np

# HACK: Mock lightgbm to allow importing ember without it being installed
# EMBER only needs lightgbm for its own pre-trained models, not for feature extraction.
try:
    import lightgbm
except ImportError:
    m = types.ModuleType('lightgbm')
    sys.modules['lightgbm'] = m

try:
    from ember.features import PEFeatureExtractor
    _EMBER_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as e:
    # If ember is not installed at all or has other issues
    PEFeatureExtractor = None
    _EMBER_AVAILABLE = False
    _IMPORT_ERROR = str(e)


class FeatureExtractionError(Exception):
    """Raised when feature extraction fails."""
    pass


class FeatureExtractor:
    """Extracts EMBERv2 features from PE files."""
    
    def __init__(self, version: int = 2):
        """
        Initialize the feature extractor.
        
        Args:
            version: EMBER feature version (default: 2)
        """
        if not _EMBER_AVAILABLE:
            raise ImportError(
                f"EMBER library not found or failed to load: {_IMPORT_ERROR}. "
                "Feature extraction requires 'ember' package."
            )
            
        # Suppress warnings during initialization
        try:
            self.extractor = PEFeatureExtractor(version)
        except Exception as e:
             raise FeatureExtractionError(f"Failed to initialize EMBER extractor: {e}")

        self.dim = self.extractor.dim  # Usually 2381 for v2

    def extract(self, file_path: Path) -> np.ndarray:
        """
        Extract features from a PE file.
        
        Args:
            file_path: Path to the PE file
            
        Returns:
            Numpy array of features (float32)
            
        Raises:
            FeatureExtractionError on failure
        """
        try:
            with open(file_path, "rb") as f:
                bytez = f.read()
                
            # extract features
            # Note: feature_vector usually returns a numpy array, but we enforce float32
            features = self.extractor.feature_vector(bytez)
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            raise FeatureExtractionError(f"Failed to extract features from {file_path}: {e}")
