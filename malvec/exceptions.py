"""
MalVec Exception Classes.

Provides user-friendly exception classes with helpful hints
for common error scenarios.

Each exception includes:
- Clear error message describing what went wrong
- Optional hint suggesting how to fix the problem
- Proper inheritance for catching by type

Usage:
    from malvec.exceptions import ModelNotFoundError

    raise ModelNotFoundError(Path("./missing_model"))
    # Output: Error: Model not found at ./missing_model
    #         Hint: Train a model first: python -m malvec.cli.train --output ./model
"""

from pathlib import Path
from typing import Optional


class MalVecError(Exception):
    """
    Base exception for all MalVec errors.

    Provides consistent formatting with optional hints.

    Attributes:
        message: Primary error message.
        hint: Optional suggestion for resolution.
    """

    def __init__(self, message: str, hint: Optional[str] = None):
        """
        Initialize MalVec error.

        Args:
            message: Primary error description.
            hint: Optional suggestion for fixing the error.
        """
        self.message = message
        self.hint = hint
        super().__init__(self.format_message())

    def format_message(self) -> str:
        """Format error message with hint."""
        msg = f"Error: {self.message}"
        if self.hint:
            msg += f"\nHint: {self.hint}"
        return msg


class ModelNotFoundError(MalVecError):
    """Model file or directory not found."""

    def __init__(self, path: Path):
        """
        Initialize model not found error.

        Args:
            path: Path where model was expected.
        """
        self.path = path
        super().__init__(
            f"Model not found at {path}",
            hint="Train a model first: python -m malvec.cli.train --output ./model"
        )


class InvalidModelError(MalVecError):
    """Model is invalid or corrupted."""

    def __init__(self, path: Path, reason: str):
        """
        Initialize invalid model error.

        Args:
            path: Path to the invalid model.
            reason: Description of what's wrong.
        """
        self.path = path
        self.reason = reason
        super().__init__(
            f"Invalid model at {path}: {reason}",
            hint="Model files may be corrupted. Re-train or re-download the model."
        )


class InvalidFileError(MalVecError):
    """Input file is invalid or unsupported."""

    def __init__(self, path: Path, reason: str):
        """
        Initialize invalid file error.

        Args:
            path: Path to the invalid file.
            reason: Why the file is invalid.
        """
        self.path = path
        self.reason = reason
        super().__init__(
            f"Invalid file {path}: {reason}",
            hint="Ensure file is a valid PE binary and < 50MB"
        )


class FileNotFoundError(MalVecError):
    """Required file not found."""

    def __init__(self, path: Path, description: str = "file"):
        """
        Initialize file not found error.

        Args:
            path: Path that was not found.
            description: Type of file (e.g., "input file", "model").
        """
        self.path = path
        super().__init__(
            f"{description.capitalize()} not found: {path}",
            hint=f"Check that the path exists and is accessible."
        )


class ExtractionTimeoutError(MalVecError):
    """Feature extraction exceeded timeout."""

    def __init__(self, timeout: int, file_path: Optional[Path] = None):
        """
        Initialize timeout error.

        Args:
            timeout: Timeout that was exceeded (seconds).
            file_path: Path to file being processed (optional).
        """
        self.timeout = timeout
        self.file_path = file_path

        msg = f"Feature extraction exceeded {timeout}s timeout"
        if file_path:
            msg += f" for {file_path.name}"

        super().__init__(
            msg,
            hint="File may be malformed or excessively large. "
                 "Try increasing timeout with --timeout option."
        )


class ExtractionError(MalVecError):
    """Feature extraction failed."""

    def __init__(self, reason: str, file_path: Optional[Path] = None):
        """
        Initialize extraction error.

        Args:
            reason: Why extraction failed.
            file_path: Path to file being processed (optional).
        """
        self.reason = reason
        self.file_path = file_path

        msg = f"Feature extraction failed: {reason}"
        if file_path:
            msg = f"Feature extraction failed for {file_path.name}: {reason}"

        super().__init__(
            msg,
            hint="The file may be corrupted or in an unsupported format."
        )


class ValidationError(MalVecError):
    """Input validation failed."""

    def __init__(self, reason: str, file_path: Optional[Path] = None):
        """
        Initialize validation error.

        Args:
            reason: Why validation failed.
            file_path: Path to file that failed validation (optional).
        """
        self.reason = reason
        self.file_path = file_path

        if file_path:
            msg = f"Validation failed for {file_path}: {reason}"
        else:
            msg = f"Validation failed: {reason}"

        super().__init__(msg)


class SandboxViolationError(MalVecError):
    """Sandbox security constraint violated."""

    def __init__(
        self,
        violation: str,
        violation_type: str = "unknown",
        file_path: Optional[Path] = None
    ):
        """
        Initialize sandbox violation error.

        Args:
            violation: Description of the violation.
            violation_type: Type (timeout, memory, crash).
            file_path: Path to file being processed (optional).
        """
        self.violation = violation
        self.violation_type = violation_type
        self.file_path = file_path

        hints = {
            "timeout": "Increase timeout with --timeout option.",
            "memory": "Increase memory limit or process smaller files.",
            "crash": "The file may be maliciously crafted to crash parsers.",
            "unknown": "Check audit logs for more details.",
        }

        super().__init__(
            violation,
            hint=hints.get(violation_type, hints["unknown"])
        )


class ConfigurationError(MalVecError):
    """Configuration is invalid."""

    def __init__(self, reason: str, config_path: Optional[Path] = None):
        """
        Initialize configuration error.

        Args:
            reason: What's wrong with configuration.
            config_path: Path to config file (optional).
        """
        self.reason = reason
        self.config_path = config_path

        if config_path:
            msg = f"Invalid configuration in {config_path}: {reason}"
        else:
            msg = f"Invalid configuration: {reason}"

        super().__init__(
            msg,
            hint="Check configuration file syntax and values."
        )


class DatasetError(MalVecError):
    """Dataset loading or processing error."""

    def __init__(self, reason: str, dataset_path: Optional[Path] = None):
        """
        Initialize dataset error.

        Args:
            reason: What went wrong with dataset.
            dataset_path: Path to dataset (optional).
        """
        self.reason = reason
        self.dataset_path = dataset_path

        if dataset_path:
            msg = f"Dataset error at {dataset_path}: {reason}"
        else:
            msg = f"Dataset error: {reason}"

        super().__init__(
            msg,
            hint="Ensure dataset is downloaded and properly formatted. "
                 "Run: python -c \"import ember; ember.create_vectorized_features('/data/ember2018/')\""
        )


class ClassificationError(MalVecError):
    """Classification failed."""

    def __init__(self, reason: str):
        """
        Initialize classification error.

        Args:
            reason: Why classification failed.
        """
        self.reason = reason
        super().__init__(
            f"Classification failed: {reason}",
            hint="Check model compatibility and input file format."
        )


def format_exception_for_cli(exc: Exception) -> str:
    """
    Format exception for CLI display.

    Provides clean output without Python traceback for known errors.

    Args:
        exc: Exception to format.

    Returns:
        User-friendly error string.
    """
    if isinstance(exc, MalVecError):
        return str(exc)
    else:
        return f"Error: {type(exc).__name__}: {exc}"
