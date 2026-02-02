"""
Tests for malvec.extractor (Native Feature Extraction).

This suite tests the native FeatureExtractor against real binaries (system files)
to ensure it produces valid EMBER-compatible feature vectors.
"""

import sys
import pytest
import numpy as np
from pathlib import Path
from malvec.extractor import FeatureExtractor

class TestFeatureExtractor:
    """Test the native FeatureExtractor with real system binaries."""
    
    @pytest.fixture
    def real_pe_file(self):
        """Use the python executable itself as a test PE file."""
        path = Path(sys.executable)
        if not path.exists():
            pytest.skip("Python executable not found (strange environment?)")
        return path

    def test_initialization(self):
        """Should initialize with correct dimension."""
        extractor = FeatureExtractor()
        assert extractor.dim == 2381

    def test_extract_real_binary(self, real_pe_file):
        """Should extract features from a real system binary."""
        extractor = FeatureExtractor()
        features = extractor.extract(real_pe_file)
        
        # Verify shape
        assert isinstance(features, np.ndarray)
        assert features.shape == (2381,)
        assert features.dtype == np.float32
        
        # Verify content is not empty/zero (real files have content)
        assert np.any(features != 0)
        
        # Verify specific feature ranges roughly
        # ByteHistogram (first 256) should sum to 1.0 (normalized)
        hist = features[0:256]
        assert np.isclose(hist.sum(), 1.0, atol=1e-4)
        
        # ByteEntropyHistogram (next 256)
        entropy = features[256:512]
        assert np.all(entropy >= 0)
        assert np.all(entropy <= 1.0) # normalized
        
        # General info (size should match file size)
        # Feature index for "size" depends on order, let's just check non-zero
        
    def test_extract_failure(self, tmp_path):
        """Should handle invalid files gracefully."""
        # Create a file that is NOT a PE (just text)
        f = tmp_path / "text.txt"
        f.write_text("This is not a PE file.")
        
        extractor = FeatureExtractor()
        
        # LIEF might return None or partial, or raise error.
        # Our implementation catches LIEF errors and prints, returns None potentially?
        # Let's check implementation behavior:
        # try: lief_binary = ... except ...: lief_binary = None
        # Then features calculate using raw bytez + None binary.
        # This is valid EMBER behavior (fallback to byte-only features).
        
        features = extractor.extract(f)
        assert isinstance(features, np.ndarray)
        assert features.shape == (2381,)
        
        # Most structural features should be 0 since lief_binary is None
        # But byte-histogram should be populated
        assert np.any(features[0:256] != 0)

    def test_nonexistent_file(self):
        """Should raise error for missing file."""
        extractor = FeatureExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract("nonexistent_ghost_file.exe")
