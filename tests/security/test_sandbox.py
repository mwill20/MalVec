"""
Tests for malvec.sandbox module.

Tests sandbox isolation, timeout enforcement, and memory limits.
"""

import pytest
import sys
import time
from pathlib import Path

from malvec.sandbox import (
    SandboxConfig,
    SandboxContext,
    SandboxViolation,
    sandboxed_execution,
    run_sandboxed,
)
from malvec.isolation import run_isolated, is_memory_limit_supported


# Helper functions must be module-level to be picklable on Windows
def _simple_add(x, y):
    """Simple function for testing basic execution."""
    return x + y


def _sleep_forever():
    """Function that never returns - for timeout testing."""
    while True:
        time.sleep(0.1)


def _sleep_short(seconds):
    """Function that sleeps for specified time."""
    time.sleep(seconds)
    return "completed"


def _allocate_memory_mb(mb):
    """Attempt to allocate specified MB of memory."""
    # Allocate memory in chunks to avoid Python optimizations
    data = []
    chunk_size = 1024 * 1024  # 1MB chunks
    for _ in range(mb):
        data.append(bytearray(chunk_size))
    return len(data)


def _raise_value_error():
    """Function that raises an exception."""
    raise ValueError("Test error message")


def _crash_process():
    """Function that crashes the process."""
    sys.exit(99)


def _return_file_content(file_path):
    """Read and return file content."""
    with open(file_path, 'rb') as f:
        return f.read()[:100]  # First 100 bytes


class TestSandboxConfig:
    """Test SandboxConfig dataclass."""

    def test_default_values(self):
        """Verify default configuration values."""
        config = SandboxConfig()

        assert config.timeout == 30
        assert config.max_memory == 512 * 1024 * 1024
        assert config.max_filesize == 50 * 1024 * 1024
        assert config.allow_network is False
        assert isinstance(config.temp_dir, Path)

    def test_custom_values(self):
        """Verify custom configuration values."""
        config = SandboxConfig(
            timeout=10,
            max_memory=256 * 1024 * 1024,
            max_filesize=10 * 1024 * 1024,
            allow_network=True,
        )

        assert config.timeout == 10
        assert config.max_memory == 256 * 1024 * 1024
        assert config.max_filesize == 10 * 1024 * 1024
        assert config.allow_network is True

    def test_invalid_timeout_raises(self):
        """Verify invalid timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            SandboxConfig(timeout=0)

        with pytest.raises(ValueError, match="timeout must be positive"):
            SandboxConfig(timeout=-1)

    def test_invalid_memory_raises(self):
        """Verify invalid max_memory raises ValueError."""
        with pytest.raises(ValueError, match="max_memory must be positive"):
            SandboxConfig(max_memory=0)

    def test_invalid_filesize_raises(self):
        """Verify invalid max_filesize raises ValueError."""
        with pytest.raises(ValueError, match="max_filesize must be positive"):
            SandboxConfig(max_filesize=-1)


class TestSandboxContext:
    """Test SandboxContext context manager."""

    def test_context_manager_lifecycle(self):
        """Verify context manager enters and exits correctly."""
        config = SandboxConfig(timeout=5)
        sandbox = SandboxContext(config)

        assert sandbox.is_active is False

        with sandbox:
            assert sandbox.is_active is True
            assert sandbox.temp_path is not None
            assert sandbox.temp_path.exists()

        assert sandbox.is_active is False

    def test_temp_directory_cleanup(self):
        """Verify temp directory is cleaned up on exit."""
        config = SandboxConfig(timeout=5)

        with SandboxContext(config) as sandbox:
            temp_path = sandbox.temp_path
            assert temp_path.exists()

        # After exit, temp directory should be removed
        assert not temp_path.exists()

    def test_successful_execution(self):
        """Verify sandbox allows valid operations."""
        config = SandboxConfig(timeout=5)

        with SandboxContext(config) as sandbox:
            result = sandbox.run(_simple_add, 2, 3)

        assert result == 5

    def test_run_outside_context_raises(self):
        """Verify running outside context raises RuntimeError."""
        sandbox = SandboxContext()

        with pytest.raises(RuntimeError, match="Sandbox not active"):
            sandbox.run(_simple_add, 1, 2)


class TestTimeoutEnforcement:
    """Test timeout enforcement in sandbox."""

    def test_timeout_kills_long_running_process(self):
        """Verify timeout kills process that runs too long."""
        config = SandboxConfig(timeout=1)

        start = time.time()
        with pytest.raises(SandboxViolation) as excinfo:
            with SandboxContext(config) as sandbox:
                sandbox.run(_sleep_forever)

        duration = time.time() - start

        # Should timeout around 1 second, give some slack
        assert duration < 3.0
        assert excinfo.value.violation_type == "timeout"
        assert "timeout" in str(excinfo.value).lower()

    def test_fast_function_completes_before_timeout(self):
        """Verify fast functions complete successfully."""
        config = SandboxConfig(timeout=5)

        with SandboxContext(config) as sandbox:
            result = sandbox.run(_sleep_short, 0.1)

        assert result == "completed"

    def test_run_isolated_timeout(self):
        """Test run_isolated timeout directly."""
        start = time.time()
        with pytest.raises(TimeoutError):
            run_isolated(_sleep_forever, timeout=0.5)
        duration = time.time() - start

        assert duration < 2.0


class TestMemoryLimits:
    """Test memory limit enforcement."""

    @pytest.mark.skipif(
        not is_memory_limit_supported(),
        reason="Memory limits not supported on this platform"
    )
    def test_memory_limit_prevents_exhaustion(self):
        """Verify memory limit prevents large allocations."""
        # Try to allocate 1GB with 100MB limit
        with pytest.raises((MemoryError, SandboxViolation)):
            run_isolated(
                _allocate_memory_mb,
                1024,  # 1GB
                timeout=30,
                max_memory=100 * 1024 * 1024  # 100MB limit
            )

    def test_small_allocation_succeeds(self):
        """Verify small allocations succeed."""
        # Allocate 10MB with 512MB limit
        result = run_isolated(
            _allocate_memory_mb,
            10,  # 10MB
            timeout=30,
            max_memory=512 * 1024 * 1024
        )
        assert result == 10


class TestExceptionPropagation:
    """Test exception propagation from sandbox."""

    def test_exception_propagates(self):
        """Verify exceptions are propagated from sandboxed code."""
        config = SandboxConfig(timeout=5)

        with pytest.raises(ValueError, match="Test error message"):
            with SandboxContext(config) as sandbox:
                sandbox.run(_raise_value_error)

    def test_run_isolated_exception(self):
        """Test run_isolated exception propagation."""
        with pytest.raises(ValueError, match="Test error message"):
            run_isolated(_raise_value_error, timeout=5)


class TestProcessCrashHandling:
    """Test handling of process crashes."""

    def test_crash_detected_as_violation(self):
        """Verify process crash is detected."""
        config = SandboxConfig(timeout=5)

        with pytest.raises((SandboxViolation, RuntimeError)):
            with SandboxContext(config) as sandbox:
                sandbox.run(_crash_process)

    def test_run_isolated_crash(self):
        """Test run_isolated crash detection."""
        with pytest.raises(RuntimeError) as excinfo:
            run_isolated(_crash_process, timeout=5)
        assert "Exit Code: 99" in str(excinfo.value)


class TestFileValidation:
    """Test file validation before sandbox execution."""

    def test_validate_existing_file(self, tmp_path):
        """Verify existing file passes validation."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"MZ" + b"\x00" * 100)

        config = SandboxConfig(max_filesize=1024)
        with SandboxContext(config) as sandbox:
            sandbox.validate_file(test_file)  # Should not raise

    def test_validate_nonexistent_file_raises(self, tmp_path):
        """Verify nonexistent file raises SandboxViolation."""
        nonexistent = tmp_path / "does_not_exist.exe"

        config = SandboxConfig()
        with SandboxContext(config) as sandbox:
            with pytest.raises(SandboxViolation) as excinfo:
                sandbox.validate_file(nonexistent)
            assert excinfo.value.violation_type == "file_not_found"

    def test_validate_oversized_file_raises(self, tmp_path):
        """Verify oversized file raises SandboxViolation."""
        large_file = tmp_path / "large.bin"
        large_file.write_bytes(b"\x00" * 1000)  # 1000 bytes

        config = SandboxConfig(max_filesize=100)  # 100 byte limit
        with SandboxContext(config) as sandbox:
            with pytest.raises(SandboxViolation) as excinfo:
                sandbox.validate_file(large_file)
            assert excinfo.value.violation_type == "file_too_large"


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_sandboxed_execution_context(self):
        """Test sandboxed_execution context manager."""
        with sandboxed_execution() as sandbox:
            result = sandbox.run(_simple_add, 10, 20)
        assert result == 30

    def test_run_sandboxed_function(self):
        """Test run_sandboxed convenience function."""
        result = run_sandboxed(_simple_add, 5, 7, timeout=5)
        assert result == 12

    def test_run_sandboxed_with_timeout(self):
        """Test run_sandboxed respects timeout."""
        start = time.time()
        with pytest.raises(SandboxViolation):
            run_sandboxed(_sleep_forever, timeout=1)
        duration = time.time() - start
        assert duration < 3.0


class TestFileOperations:
    """Test file operations within sandbox."""

    def test_read_file_in_sandbox(self, tmp_path):
        """Verify file reading works in sandbox."""
        test_file = tmp_path / "test.txt"
        test_content = b"Hello, sandbox!"
        test_file.write_bytes(test_content)

        with SandboxContext() as sandbox:
            result = sandbox.run(_return_file_content, str(test_file))

        assert result == test_content
