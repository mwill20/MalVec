"""
Tests for malvec.validator.
"""

import pytest
import os
import tempfile
from pathlib import Path
from malvec.validator import InputValidator, ValidationError

class TestValidator:
    """Test functionality of InputValidator."""
    
    @pytest.fixture
    def valid_file(self):
        """Create a valid PE file for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.exe') as f:
            # Create a mock PE file
            f.write(b'MZ' + b'A' * 300)
            path = Path(f.name)
            
        yield path
        
        # Cleanup
        if path.exists():
            path.unlink()
            
    @pytest.fixture
    def invalid_magic(self):
        """Create a file with invalid magic bytes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as f:
            # Must be > 256 bytes to pass size check
            f.write(b'This is not a PE file' + b'A' * 300)
            path = Path(f.name)
            
        yield path
        
        if path.exists():
            path.unlink()
            
    # ... (other fixtures unchanged) ...
            
    def test_max_size(self):
        """Should verify max size limit logic."""
        from unittest.mock import patch, PropertyMock
        
        path = "dummy_path.exe"
        
        # Mock Path.stat().st_size
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 60 * 1024 * 1024  # 60MB
            
            # Use PropertyMock to mock .st_size directly if needed,
            # but usually stat() returns an object with st_size.
            
            # We also need to mock exists() and is_file()
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.is_file', return_value=True):
                    with pytest.raises(ValidationError, match="File too large"):
                        InputValidator.validate(path)
