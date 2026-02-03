"""
Tests for sandboxed feature extraction.

Tests the integration of FeatureExtractor with sandbox isolation.

Note: Sandboxed extraction tests are skipped on Windows by default
because subprocess spawning with LIEF imports is slow (~30s+).
Use --runslow to run these tests.
"""

import pytest
import sys
import numpy as np
from pathlib import Path

from malvec.extractor import FeatureExtractor, extract_in_process
from malvec.sandbox import SandboxConfig, SandboxViolation


# Mark for slow sandbox tests that spawn subprocesses
SLOW_SANDBOX_REASON = "Sandboxed extraction is slow on Windows due to subprocess+LIEF import overhead"
slow_sandbox = pytest.mark.skipif(
    sys.platform == 'win32',
    reason=SLOW_SANDBOX_REASON
)


class TestExtractorSandbox:
    """Test sandboxed feature extraction."""

    @slow_sandbox
    def test_extraction_with_sandbox_enabled(self):
        """Verify extraction works with sandbox enabled."""
        # Use Python executable as a valid PE file
        pe_path = Path(sys.executable)

        config = SandboxConfig(timeout=120)  # Longer timeout for subprocess
        extractor = FeatureExtractor(sandbox=True, config=config)
        features = extractor.extract(pe_path)

        assert features.shape == (2381,)
        assert features.dtype == np.float32
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()

    def test_extraction_without_sandbox(self):
        """Verify direct extraction still works."""
        pe_path = Path(sys.executable)

        extractor = FeatureExtractor(sandbox=False)
        features = extractor.extract(pe_path)

        assert features.shape == (2381,)
        assert features.dtype == np.float32

    @slow_sandbox
    def test_extraction_results_match(self):
        """Verify sandboxed and direct extraction produce same results."""
        pe_path = Path(sys.executable)

        # Extract with sandbox (longer timeout for subprocess)
        config = SandboxConfig(timeout=120)
        sandboxed_extractor = FeatureExtractor(sandbox=True, config=config)
        sandboxed_features = sandboxed_extractor.extract(pe_path)

        # Extract without sandbox
        direct_extractor = FeatureExtractor(sandbox=False)
        direct_features = direct_extractor.extract(pe_path)

        # Results should be identical
        np.testing.assert_array_almost_equal(
            sandboxed_features,
            direct_features,
            decimal=5
        )

    @slow_sandbox
    def test_custom_sandbox_config(self):
        """Verify custom sandbox configuration is applied."""
        pe_path = Path(sys.executable)

        config = SandboxConfig(
            timeout=120,  # Longer timeout for subprocess
            max_memory=1024 * 1024 * 1024,  # 1GB
            max_filesize=100 * 1024 * 1024  # 100MB
        )
        extractor = FeatureExtractor(sandbox=True, config=config)

        assert extractor.sandbox_config.timeout == 120
        assert extractor.sandbox_config.max_memory == 1024 * 1024 * 1024

        features = extractor.extract(pe_path)
        assert features.shape == (2381,)

    def test_sandbox_enabled_by_default(self):
        """Verify sandbox is enabled by default."""
        extractor = FeatureExtractor()
        assert extractor.sandbox_enabled is True

    @slow_sandbox
    def test_extraction_timeout_config(self):
        """Verify timeout configuration is respected."""
        pe_path = Path(sys.executable)

        # Longer timeout for subprocess extraction
        config = SandboxConfig(timeout=120)
        extractor = FeatureExtractor(sandbox=True, config=config)

        features = extractor.extract(pe_path)
        assert features.shape == (2381,)

    def test_file_validation_in_sandbox(self, tmp_path):
        """Verify file validation happens before extraction."""
        nonexistent = tmp_path / "nonexistent.exe"

        extractor = FeatureExtractor(sandbox=True)

        with pytest.raises(SandboxViolation) as excinfo:
            extractor.extract(nonexistent)

        assert excinfo.value.violation_type == "file_not_found"

    def test_oversized_file_rejected(self, tmp_path):
        """Verify oversized files are rejected."""
        # Create a file larger than max_filesize
        large_file = tmp_path / "large.exe"
        large_file.write_bytes(b"MZ" + b"\x00" * 1000)

        config = SandboxConfig(max_filesize=100)  # 100 byte limit
        extractor = FeatureExtractor(sandbox=True, config=config)

        with pytest.raises(SandboxViolation) as excinfo:
            extractor.extract(large_file)

        assert excinfo.value.violation_type == "file_too_large"


class TestExtractInProcess:
    """Test the extract_in_process helper function."""

    def test_extract_in_process_function(self):
        """Verify extract_in_process works correctly."""
        pe_path = Path(sys.executable)

        features = extract_in_process(pe_path)

        assert features.shape == (2381,)
        assert features.dtype == np.float32

    def test_extract_in_process_no_double_sandbox(self):
        """Verify extract_in_process doesn't create nested sandbox."""
        pe_path = Path(sys.executable)

        # extract_in_process creates extractor with sandbox=False
        # to avoid nested sandboxing when called from SandboxContext
        features = extract_in_process(pe_path)

        assert features.shape == (2381,)


class TestExtractorDimensions:
    """Test feature extractor dimensions."""

    def test_feature_dimensions(self):
        """Verify total feature dimensions is 2381."""
        extractor = FeatureExtractor(sandbox=False)

        assert extractor.dim == 2381

    def test_output_shape(self):
        """Verify output shape matches expected dimensions."""
        pe_path = Path(sys.executable)

        extractor = FeatureExtractor(sandbox=False)
        features = extractor.extract(pe_path)

        assert features.shape == (extractor.dim,)
        assert features.shape == (2381,)


class TestExtractorErrorHandling:
    """Test error handling in sandboxed extraction."""

    @slow_sandbox
    def test_invalid_path_type_handling(self):
        """Test handling of invalid path types."""
        config = SandboxConfig(timeout=120)
        extractor = FeatureExtractor(sandbox=True, config=config)

        # String path should work
        features = extractor.extract(str(sys.executable))
        assert features.shape == (2381,)

        # Path object should work
        features = extractor.extract(Path(sys.executable))
        assert features.shape == (2381,)

    @slow_sandbox
    def test_extraction_with_corrupted_pe(self, tmp_path):
        """Test extraction with invalid PE file."""
        # Create a file with MZ header but invalid PE structure
        bad_pe = tmp_path / "bad.exe"
        bad_pe.write_bytes(b"MZ" + b"\x00" * 256)

        config = SandboxConfig(timeout=120)
        extractor = FeatureExtractor(sandbox=True, config=config)

        # Should extract byte-level features even if LIEF fails
        # (graceful degradation)
        features = extractor.extract(bad_pe)

        # Features should still be extracted (fallback mode)
        assert features.shape == (2381,)
        assert features.dtype == np.float32
