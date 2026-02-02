"""
Tests for malvec.features (Feature Extraction).
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path

# We need to mock ember before importing malvec.features in case ember is missing
# or to control its behavior completely.
# However, malvec.features has conditional import logic.

from malvec.features import FeatureExtractor, FeatureExtractionError

class TestFeatureExtractor:
    """Test the FeatureExtractor wrapper."""
    
    @pytest.fixture
    def mock_extractor_cls(self):
        """Mock the underlying PEFeatureExtractor class."""
        with patch('malvec.features.PEFeatureExtractor') as mock:
            # Setup the instance returned by constructor
            instance = mock.return_value
            instance.dim = 2381
            instance.feature_vector.return_value = np.zeros(2381, dtype=np.float32)
            yield mock

    def test_initialization(self, mock_extractor_cls):
        """Should initialize correctly with defaults."""
        extractor = FeatureExtractor()
        assert extractor.dim == 2381
        mock_extractor_cls.assert_called_with(2)  # Check default version

    def test_extract_success(self, mock_extractor_cls, tmp_path):
        """Should extract features from a file."""
        # Create dummy file
        f = tmp_path / "test.exe"
        f.write_bytes(b"MZ" + b"\x00"*100)
        
        extractor = FeatureExtractor()
        features = extractor.extract(f)
        
        # Verify result
        assert isinstance(features, np.ndarray)
        assert features.shape == (2381,)
        assert features.dtype == np.float32
        
        # Verify call arguments
        # The extractor reads the file content and passes bytes
        mock_extractor_cls.return_value.feature_vector.assert_called_once()
        args = mock_extractor_cls.return_value.feature_vector.call_args
        assert args[0][0].startswith(b"MZ")

    def test_extract_failure(self, mock_extractor_cls, tmp_path):
        """Should raise FeatureExtractionError on underlying failure."""
        f = tmp_path / "bad.exe"
        f.write_bytes(b"bad")
        
        # Simulate underlying library error
        mock_extractor_cls.return_value.feature_vector.side_effect = Exception("Parsing error")
        
        extractor = FeatureExtractor()
        
        with pytest.raises(FeatureExtractionError, match="Failed to extract"):
            extractor.extract(f)

    def test_file_read_error(self, mock_extractor_cls):
        """Should raise error if file read fails."""
        extractor = FeatureExtractor()
        
        # Non-existent file
        with pytest.raises(FeatureExtractionError):
            extractor.extract(Path("nonexistent_file_123.exe"))
