"""
Tests for malvec.audit module.

Tests audit logging functionality for security events.
"""

import pytest
import json
import sys
from pathlib import Path
from datetime import datetime

from malvec.audit import (
    AuditLogger,
    AuditConfig,
    get_audit_logger,
    log_classification,
    log_validation_failure,
    log_sandbox_violation,
)


class TestAuditConfig:
    """Test AuditConfig dataclass."""

    def test_default_values(self):
        """Verify default configuration values."""
        config = AuditConfig()

        assert config.log_filename == "audit.log"
        assert config.log_level >= 0
        assert config.include_file_names is False
        assert config.max_log_size == 10 * 1024 * 1024
        assert config.backup_count == 5

    def test_custom_values(self, tmp_path):
        """Verify custom configuration values."""
        config = AuditConfig(
            log_dir=tmp_path,
            log_filename="custom.log",
            include_file_names=True,
            max_log_size=5 * 1024 * 1024,
            backup_count=3,
        )

        assert config.log_dir == tmp_path
        assert config.log_filename == "custom.log"
        assert config.include_file_names is True
        assert config.max_log_size == 5 * 1024 * 1024
        assert config.backup_count == 3

    def test_log_path_property(self, tmp_path):
        """Verify log_path property."""
        config = AuditConfig(log_dir=tmp_path, log_filename="test.log")
        assert config.log_path == tmp_path / "test.log"


class TestAuditLogger:
    """Test AuditLogger class."""

    def test_logger_creates_log_directory(self, tmp_path):
        """Verify logger creates log directory."""
        log_dir = tmp_path / "logs" / "audit"
        config = AuditConfig(log_dir=log_dir)

        logger = AuditLogger(config)

        assert log_dir.exists()

    def test_logger_creates_log_file(self, tmp_path):
        """Verify logger creates log file on first write."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(log_path=log_path)

        # Write a log entry
        logger.log_system_event("test_event")

        assert log_path.exists()

    def test_log_path_property(self, tmp_path):
        """Verify log_path property returns correct path."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(log_path=log_path)

        assert logger.log_path == log_path


class TestClassificationLogging:
    """Test classification event logging."""

    def test_log_classification(self, tmp_path):
        """Verify classification logging."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(log_path=log_path)

        # Create a test file
        test_file = tmp_path / "test.exe"
        test_file.write_bytes(b"MZ" + b"\x00" * 100)

        logger.log_classification(
            test_file,
            "MALWARE",
            0.85,
            0.123,
            needs_review=False
        )

        # Verify log file created
        assert log_path.exists()

        # Verify log content
        with open(log_path) as f:
            log_content = f.read()

        assert "classification" in log_content
        assert "MALWARE" in log_content
        assert "0.85" in log_content

    def test_classification_log_structure(self, tmp_path):
        """Verify classification log has correct JSON structure."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(log_path=log_path)

        test_file = tmp_path / "test.exe"
        test_file.write_bytes(b"MZ" + b"\x00" * 100)

        logger.log_classification(
            test_file,
            "BENIGN",
            0.92,
            1.5,
            needs_review=True
        )

        with open(log_path) as f:
            line = f.readline()

        # Extract JSON from log line (after timestamp and level)
        json_start = line.find("{")
        json_data = json.loads(line[json_start:])

        assert json_data["event"] == "classification"
        assert json_data["prediction"] == "BENIGN"
        assert json_data["confidence"] == 0.92
        assert json_data["processing_time_ms"] == 1500
        assert json_data["needs_review"] is True
        assert "timestamp" in json_data
        assert "sha256" in json_data["file"]

    def test_classification_with_extra_data(self, tmp_path):
        """Test classification logging with extra data."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(log_path=log_path)

        test_file = tmp_path / "test.exe"
        test_file.write_bytes(b"MZ" + b"\x00" * 100)

        logger.log_classification(
            test_file,
            "MALWARE",
            0.95,
            0.5,
            extra={"model_version": "1.0", "k": 5}
        )

        with open(log_path) as f:
            line = f.readline()

        json_start = line.find("{")
        json_data = json.loads(line[json_start:])

        assert "extra" in json_data
        assert json_data["extra"]["model_version"] == "1.0"


class TestValidationFailureLogging:
    """Test validation failure event logging."""

    def test_log_validation_failure(self, tmp_path):
        """Verify validation failure logging."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(log_path=log_path)

        test_file = tmp_path / "invalid.exe"
        test_file.write_bytes(b"NOTMZ" + b"\x00" * 100)

        logger.log_validation_failure(
            test_file,
            "Invalid magic bytes"
        )

        with open(log_path) as f:
            log_content = f.read()

        assert "validation_failure" in log_content
        assert "Invalid magic bytes" in log_content

    def test_validation_failure_structure(self, tmp_path):
        """Verify validation failure log structure."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(log_path=log_path)

        test_file = tmp_path / "large.exe"
        test_file.write_bytes(b"MZ" + b"\x00" * 100)

        logger.log_validation_failure(
            test_file,
            "File too large",
            error_type="size_exceeded"
        )

        with open(log_path) as f:
            line = f.readline()

        json_start = line.find("{")
        json_data = json.loads(line[json_start:])

        assert json_data["event"] == "validation_failure"
        assert json_data["reason"] == "File too large"
        assert json_data["error_type"] == "size_exceeded"


class TestSandboxViolationLogging:
    """Test sandbox violation event logging."""

    def test_log_sandbox_violation(self, tmp_path):
        """Verify sandbox violation logging."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(log_path=log_path)

        test_file = tmp_path / "malicious.exe"
        test_file.write_bytes(b"MZ" + b"\x00" * 100)

        logger.log_sandbox_violation(
            test_file,
            "Execution exceeded 30s timeout",
            violation_type="timeout"
        )

        with open(log_path) as f:
            log_content = f.read()

        assert "sandbox_violation" in log_content
        assert "timeout" in log_content

    def test_sandbox_violation_structure(self, tmp_path):
        """Verify sandbox violation log structure."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(log_path=log_path)

        test_file = tmp_path / "crasher.exe"
        test_file.write_bytes(b"MZ" + b"\x00" * 100)

        logger.log_sandbox_violation(
            test_file,
            "Memory limit exceeded",
            violation_type="memory_exceeded"
        )

        with open(log_path) as f:
            line = f.readline()

        json_start = line.find("{")
        json_data = json.loads(line[json_start:])

        assert json_data["event"] == "sandbox_violation"
        assert json_data["violation"] == "Memory limit exceeded"
        assert json_data["violation_type"] == "memory_exceeded"


class TestProcessingErrorLogging:
    """Test processing error event logging."""

    def test_log_processing_error(self, tmp_path):
        """Verify processing error logging."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(log_path=log_path)

        test_file = tmp_path / "error.exe"
        test_file.write_bytes(b"MZ" + b"\x00" * 100)

        logger.log_processing_error(
            test_file,
            "LIEF parsing failed",
            error_type="ParseError",
            stage="feature_extraction"
        )

        with open(log_path) as f:
            log_content = f.read()

        assert "processing_error" in log_content
        assert "LIEF parsing failed" in log_content
        assert "feature_extraction" in log_content


class TestSystemEventLogging:
    """Test system event logging."""

    def test_log_system_event(self, tmp_path):
        """Verify system event logging."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(log_path=log_path)

        logger.log_system_event(
            "startup",
            {"version": "1.0.0", "model": "knn_v1"}
        )

        with open(log_path) as f:
            line = f.readline()

        json_start = line.find("{")
        json_data = json.loads(line[json_start:])

        assert json_data["event"] == "system_event"
        assert json_data["name"] == "startup"
        assert json_data["details"]["version"] == "1.0.0"


class TestBatchSummaryLogging:
    """Test batch summary logging."""

    def test_log_batch_summary(self, tmp_path):
        """Verify batch summary logging."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(log_path=log_path)

        logger.log_batch_summary(
            total_files=100,
            successful=95,
            failed=5,
            malware_count=30,
            benign_count=65,
            processing_time=120.5
        )

        with open(log_path) as f:
            line = f.readline()

        json_start = line.find("{")
        json_data = json.loads(line[json_start:])

        assert json_data["event"] == "batch_summary"
        assert json_data["total_files"] == 100
        assert json_data["successful"] == 95
        assert json_data["failed"] == 5
        assert json_data["malware_count"] == 30
        assert json_data["benign_count"] == 65
        assert json_data["processing_time_ms"] == 120500


class TestPrivacyFeatures:
    """Test privacy-related features."""

    def test_file_hash_logged_by_default(self, tmp_path):
        """Verify file hash is always logged."""
        log_path = tmp_path / "audit.log"
        config = AuditConfig(log_dir=tmp_path, include_file_names=False)
        logger = AuditLogger(config)

        test_file = tmp_path / "test.exe"
        test_file.write_bytes(b"MZ" + b"\x00" * 100)

        logger.log_classification(test_file, "BENIGN", 0.9, 1.0)

        with open(log_path) as f:
            line = f.readline()

        json_start = line.find("{")
        json_data = json.loads(line[json_start:])

        assert "sha256" in json_data["file"]
        assert "filename" not in json_data["file"]

    def test_filename_included_when_configured(self, tmp_path):
        """Verify filename included when configured."""
        log_path = tmp_path / "audit.log"
        config = AuditConfig(log_dir=tmp_path, include_file_names=True)
        logger = AuditLogger(config)

        test_file = tmp_path / "myfile.exe"
        test_file.write_bytes(b"MZ" + b"\x00" * 100)

        logger.log_classification(test_file, "BENIGN", 0.9, 1.0)

        with open(log_path) as f:
            line = f.readline()

        json_start = line.find("{")
        json_data = json.loads(line[json_start:])

        assert "sha256" in json_data["file"]
        assert json_data["file"]["filename"] == "myfile.exe"


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_get_audit_logger_singleton(self, tmp_path, monkeypatch):
        """Verify get_audit_logger returns singleton."""
        # Reset the global logger
        import malvec.audit as audit_module
        monkeypatch.setattr(audit_module, '_default_logger', None)

        config = AuditConfig(log_dir=tmp_path)
        logger1 = get_audit_logger(config)
        logger2 = get_audit_logger()

        # Should be same instance
        assert logger1 is logger2


class TestTimestampFormat:
    """Test timestamp formatting."""

    def test_timestamp_is_iso_format(self, tmp_path):
        """Verify timestamps are in ISO format."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(log_path=log_path)

        logger.log_system_event("test")

        with open(log_path) as f:
            line = f.readline()

        json_start = line.find("{")
        json_data = json.loads(line[json_start:])

        timestamp = json_data["timestamp"]

        # Should be parseable as ISO format
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        assert dt is not None
