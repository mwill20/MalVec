"""
MalVec Audit Logging Module.

Provides security audit logging for all malware analysis operations.
Logs are structured JSON for easy parsing and analysis.

Logged Events:
- File processing attempts
- Validation failures
- Sandbox violations
- Classification results
- Errors and exceptions

Security:
- File paths are NOT logged directly (use SHA256 hashes)
- Timestamps in UTC ISO format
- JSON structured for SIEM integration
"""

import os
import sys
import json
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field


# Default log directory based on platform
if sys.platform == 'win32':
    DEFAULT_LOG_DIR = Path(os.environ.get('LOCALAPPDATA', '.')) / 'malvec' / 'logs'
else:
    DEFAULT_LOG_DIR = Path('/var/log/malvec')


@dataclass
class AuditConfig:
    """
    Configuration for audit logging.

    Attributes:
        log_dir: Directory for audit log files.
        log_filename: Name of the audit log file.
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
        include_file_names: Whether to include file names (False for privacy).
        max_log_size: Maximum log file size before rotation (bytes).
        backup_count: Number of backup log files to keep.
    """
    log_dir: Path = field(default_factory=lambda: DEFAULT_LOG_DIR)
    log_filename: str = "audit.log"
    log_level: int = logging.INFO
    include_file_names: bool = False  # Privacy: default to hash-only
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

    def __post_init__(self):
        if isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)

    @property
    def log_path(self) -> Path:
        """Full path to the audit log file."""
        return self.log_dir / self.log_filename


class AuditLogger:
    """
    Security audit logger for MalVec operations.

    Logs all security-relevant events in structured JSON format.
    Designed for integration with SIEM systems and security monitoring.

    Events logged:
    - classification: Sample classification results
    - validation_failure: Input validation failures
    - sandbox_violation: Sandbox constraint violations
    - processing_error: Unexpected errors during processing
    - system_event: System-level events (startup, shutdown)

    Security considerations:
    - File paths are hashed by default (SHA256)
    - Timestamps in UTC for consistency
    - Structured JSON for easy parsing
    - File rotation to prevent disk exhaustion

    Usage:
        audit = AuditLogger()
        audit.log_classification(file_path, "MALWARE", 0.95, 1.23)
    """

    def __init__(self, config: AuditConfig = None, log_path: Path = None):
        """
        Initialize audit logger.

        Args:
            config: Audit configuration. Uses defaults if not provided.
            log_path: Direct path to log file (overrides config).
        """
        self.config = config or AuditConfig()

        # Allow direct log_path override
        if log_path:
            self.config.log_dir = log_path.parent
            self.config.log_filename = log_path.name

        self._logger: Optional[logging.Logger] = None
        self._setup_logger()

    def _setup_logger(self):
        """Configure the audit logger with file handler."""
        # Create log directory if needed
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

        # Use unique logger name based on log path to avoid handler conflicts
        # This ensures each AuditLogger instance gets its own logger
        logger_name = f"malvec.audit.{id(self)}"
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(self.config.log_level)

        # Clear any existing handlers to avoid duplicates
        self._logger.handlers.clear()

        try:
            # Use rotating file handler to prevent disk exhaustion
            from logging.handlers import RotatingFileHandler
            handler = RotatingFileHandler(
                self.config.log_path,
                maxBytes=self.config.max_log_size,
                backupCount=self.config.backup_count
            )
        except Exception:
            # Fallback to basic file handler if rotation fails
            handler = logging.FileHandler(self.config.log_path)

        # Simple format - JSON content is self-describing
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        self._logger.addHandler(handler)

        # Prevent propagation to root logger
        self._logger.propagate = False

    def _hash_file(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of a file.

        Args:
            file_path: Path to file.

        Returns:
            Hex-encoded SHA256 hash.
        """
        sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception:
            return "HASH_ERROR"

    def _get_file_identifier(self, file_path: Path) -> Dict[str, str]:
        """
        Get file identifier for logging.

        Returns hash always, filename only if configured.
        """
        identifier = {
            "sha256": self._hash_file(file_path)
        }
        if self.config.include_file_names:
            identifier["filename"] = file_path.name
        return identifier

    def _timestamp(self) -> str:
        """Get current UTC timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    def _log_event(self, level: int, event_type: str, data: Dict[str, Any]):
        """
        Log a structured event.

        Args:
            level: Logging level.
            event_type: Type of event (classification, violation, etc.).
            data: Event data dictionary.
        """
        event = {
            "event": event_type,
            "timestamp": self._timestamp(),
            **data
        }
        self._logger.log(level, json.dumps(event))
        # Flush handlers to ensure immediate write
        for handler in self._logger.handlers:
            handler.flush()

    def log_classification(
        self,
        file_path: Union[Path, str],
        prediction: str,
        confidence: float,
        processing_time: float,
        needs_review: bool = False,
        extra: Dict[str, Any] = None
    ):
        """
        Log a classification event.

        Args:
            file_path: Path to classified file.
            prediction: Classification result ("MALWARE" or "BENIGN").
            confidence: Confidence score (0-1).
            processing_time: Processing time in seconds.
            needs_review: Whether manual review is recommended.
            extra: Additional data to include.
        """
        file_path = Path(file_path)
        data = {
            "file": self._get_file_identifier(file_path),
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "processing_time_ms": int(processing_time * 1000),
            "needs_review": needs_review
        }
        if extra:
            data["extra"] = extra
        self._log_event(logging.INFO, "classification", data)

    def log_validation_failure(
        self,
        file_path: Union[Path, str],
        reason: str,
        error_type: str = "validation_error"
    ):
        """
        Log a validation failure event.

        Args:
            file_path: Path to file that failed validation.
            reason: Human-readable reason for failure.
            error_type: Type of validation error.
        """
        file_path = Path(file_path)
        data = {
            "file": self._get_file_identifier(file_path),
            "reason": reason,
            "error_type": error_type
        }
        self._log_event(logging.WARNING, "validation_failure", data)

    def log_sandbox_violation(
        self,
        file_path: Union[Path, str],
        violation: str,
        violation_type: str = "unknown"
    ):
        """
        Log a sandbox violation event.

        Args:
            file_path: Path to file that caused violation.
            violation: Description of the violation.
            violation_type: Type of violation (timeout, memory, crash).
        """
        file_path = Path(file_path)
        data = {
            "file": self._get_file_identifier(file_path),
            "violation": violation,
            "violation_type": violation_type
        }
        self._log_event(logging.ERROR, "sandbox_violation", data)

    def log_processing_error(
        self,
        file_path: Union[Path, str],
        error: str,
        error_type: str = "processing_error",
        stage: str = "unknown"
    ):
        """
        Log a processing error event.

        Args:
            file_path: Path to file being processed.
            error: Error message.
            error_type: Type of error.
            stage: Processing stage where error occurred.
        """
        file_path = Path(file_path)
        data = {
            "file": self._get_file_identifier(file_path),
            "error": error,
            "error_type": error_type,
            "stage": stage
        }
        self._log_event(logging.ERROR, "processing_error", data)

    def log_system_event(
        self,
        event_name: str,
        details: Dict[str, Any] = None
    ):
        """
        Log a system-level event.

        Args:
            event_name: Name of the event (startup, shutdown, config_change).
            details: Additional event details.
        """
        data = {
            "name": event_name
        }
        if details:
            data["details"] = details
        self._log_event(logging.INFO, "system_event", data)

    def log_batch_summary(
        self,
        total_files: int,
        successful: int,
        failed: int,
        malware_count: int,
        benign_count: int,
        processing_time: float
    ):
        """
        Log a batch processing summary.

        Args:
            total_files: Total files processed.
            successful: Number successfully processed.
            failed: Number that failed.
            malware_count: Number classified as malware.
            benign_count: Number classified as benign.
            processing_time: Total processing time in seconds.
        """
        data = {
            "total_files": total_files,
            "successful": successful,
            "failed": failed,
            "malware_count": malware_count,
            "benign_count": benign_count,
            "processing_time_ms": int(processing_time * 1000)
        }
        self._log_event(logging.INFO, "batch_summary", data)

    @property
    def log_path(self) -> Path:
        """Get the audit log file path."""
        return self.config.log_path


# Module-level singleton for convenience
_default_logger: Optional[AuditLogger] = None


def get_audit_logger(config: AuditConfig = None) -> AuditLogger:
    """
    Get or create the default audit logger.

    Args:
        config: Configuration to use if creating new logger.

    Returns:
        AuditLogger instance.
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = AuditLogger(config)
    return _default_logger


def log_classification(
    file_path: Union[Path, str],
    prediction: str,
    confidence: float,
    processing_time: float,
    needs_review: bool = False
):
    """Convenience function to log classification with default logger."""
    get_audit_logger().log_classification(
        file_path, prediction, confidence, processing_time, needs_review
    )


def log_validation_failure(file_path: Union[Path, str], reason: str):
    """Convenience function to log validation failure with default logger."""
    get_audit_logger().log_validation_failure(file_path, reason)


def log_sandbox_violation(
    file_path: Union[Path, str],
    violation: str,
    violation_type: str = "unknown"
):
    """Convenience function to log sandbox violation with default logger."""
    get_audit_logger().log_sandbox_violation(file_path, violation, violation_type)
