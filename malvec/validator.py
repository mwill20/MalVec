"""
MalVec Input Validator.

Validates input files before processing to ensure they are safe and
likely to be valid PE (Portable Executable) files.

Features:
- Invalidates non-existent files
- Checks file size limits (DoS protection)
- Verifies 'MZ' magic bytes
- Checks for minimum file size
"""

import os
from pathlib import Path
from typing import Optional, Union


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class InputValidator:
    """Validates binary input files."""
    
    # Maximum file size (50 MB)
    MAX_SIZE_MB = 50
    MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024
    
    # Minimum PE file size (MZ header + PE header + minimal section)
    MIN_SIZE_BYTES = 256
    
    # PE Magic Bytes
    MAGIC_MZ = b'MZ'
    
    @classmethod
    def validate(cls, file_path: Union[str, Path]) -> Path:
        """
        Validate a file for processing.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Path object if valid
            
        Throws:
            ValidationError if invalid
        """
        path = Path(file_path)
        
        # 1. Existence Check
        if not path.exists():
            raise ValidationError(f"File not found: {path}")
            
        if not path.is_file():
            raise ValidationError(f"Not a file: {path}")
            
        # 2. Size Check
        size = path.stat().st_size
        
        if size > cls.MAX_SIZE_BYTES:
            raise ValidationError(
                f"File too large: {size / 1024 / 1024:.2f}MB "
                f"(max {cls.MAX_SIZE_MB}MB)"
            )
            
        if size < cls.MIN_SIZE_BYTES:
            raise ValidationError(
                f"File too small: {size} bytes (min {cls.MIN_SIZE_BYTES} bytes)"
            )
            
        # 3. Magic Bytes Check (PE header)
        try:
            with open(path, 'rb') as f:
                header = f.read(2)
                if header != cls.MAGIC_MZ:
                    raise ValidationError(
                        f"Invalid file format: Magic bytes {header!r} (expected b'MZ')"
                    )
        except OSError as e:
            raise ValidationError(f"Read error: {e}")
            
        return path
