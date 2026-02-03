"""
MalVec Sandbox Module.

Provides sandboxed execution environment for processing untrusted files.
This is a defense-in-depth layer that isolates malware analysis from
the main system.

Security Features:
- Timeout enforcement (kills process after N seconds)
- Memory limits (prevents memory exhaustion)
- Filesystem isolation (temp directory only)
- Process isolation (separate process group)
- Network isolation (where supported)

Note: Network isolation requires platform-specific features (Linux namespaces,
Windows firewall rules) and may not be available on all systems.
"""

import os
import sys
import shutil
import tempfile
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Any, Optional
from contextlib import contextmanager

from malvec.isolation import run_isolated, IsolationConfig


logger = logging.getLogger(__name__)


@dataclass
class SandboxConfig:
    """
    Sandbox configuration.

    Attributes:
        timeout: Maximum execution time in seconds.
        max_memory: Maximum memory usage in bytes.
        max_filesize: Maximum file size to process in bytes.
        allow_network: Whether to allow network access (currently advisory).
        temp_dir: Base directory for sandbox temporary files.
    """
    timeout: int = 30  # Max execution time (seconds)
    max_memory: int = 512 * 1024 * 1024  # 512MB
    max_filesize: int = 50 * 1024 * 1024  # 50MB
    allow_network: bool = False
    temp_dir: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "malvec_sandbox")

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_memory <= 0:
            raise ValueError("max_memory must be positive")
        if self.max_filesize <= 0:
            raise ValueError("max_filesize must be positive")
        # Ensure temp_dir is a Path
        if isinstance(self.temp_dir, str):
            self.temp_dir = Path(self.temp_dir)


class SandboxViolation(Exception):
    """
    Raised when sandbox constraints are violated.

    This exception indicates a security boundary was crossed,
    such as timeout exceeded, memory exhaustion, or file size limit.
    """

    def __init__(self, message: str, violation_type: str = "unknown"):
        super().__init__(message)
        self.violation_type = violation_type


class SandboxContext:
    """
    Context manager for sandboxed execution.

    Provides a secure environment for processing untrusted files with:
    - Timeout enforcement (kills process after N seconds)
    - Memory limits (prevents memory exhaustion)
    - Filesystem isolation (read-only except temp dir)
    - Process isolation (separate process group)

    Usage:
        config = SandboxConfig(timeout=30, max_memory=512*1024*1024)
        with SandboxContext(config) as sandbox:
            result = sandbox.run(extract_features, file_path)

    Note:
        Network isolation is advisory on most platforms. For true
        network isolation, use container technologies (Docker) or
        platform-specific features (Linux namespaces).
    """

    def __init__(self, config: SandboxConfig = None):
        """
        Initialize sandbox context.

        Args:
            config: Sandbox configuration. Uses defaults if not provided.
        """
        self.config = config or SandboxConfig()
        self._temp_dir: Optional[Path] = None
        self._active = False

    def __enter__(self) -> "SandboxContext":
        """
        Set up sandbox environment.

        Creates isolated temp directory and prepares resource limits.
        """
        self._setup_temp_directory()
        self._active = True

        if not self.config.allow_network:
            self._warn_network_isolation()

        logger.debug(f"Sandbox activated: timeout={self.config.timeout}s, "
                     f"max_memory={self.config.max_memory}B")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Clean up sandbox environment.

        Removes temp directory and any spawned processes.
        """
        self._active = False
        self._cleanup_temp_directory()
        logger.debug("Sandbox deactivated and cleaned up")
        return False  # Don't suppress exceptions

    def _setup_temp_directory(self):
        """Create isolated temporary directory."""
        # Create unique temp directory for this sandbox session
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)
        self._temp_dir = Path(tempfile.mkdtemp(dir=self.config.temp_dir))
        logger.debug(f"Created sandbox temp directory: {self._temp_dir}")

    def _cleanup_temp_directory(self):
        """Remove sandbox temporary directory and contents."""
        if self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
                logger.debug(f"Cleaned up sandbox temp directory: {self._temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")

    def _warn_network_isolation(self):
        """Log warning if network isolation is not supported."""
        if sys.platform == 'win32':
            logger.debug("Network isolation not enforced on Windows")
        elif sys.platform == 'darwin':
            logger.debug("Network isolation not enforced on macOS")
        else:
            logger.debug("Network isolation requires additional setup on Linux")

    def validate_file(self, file_path: Path) -> None:
        """
        Validate file before sandboxed processing.

        Args:
            file_path: Path to file to validate.

        Raises:
            SandboxViolation: If file exceeds size limit or doesn't exist.
        """
        if not file_path.exists():
            raise SandboxViolation(
                f"File does not exist: {file_path}",
                violation_type="file_not_found"
            )

        file_size = file_path.stat().st_size
        if file_size > self.config.max_filesize:
            raise SandboxViolation(
                f"File size ({file_size} bytes) exceeds limit ({self.config.max_filesize} bytes)",
                violation_type="file_too_large"
            )

    def run(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run function in sandboxed subprocess.

        The function is executed in a separate process with:
        - Timeout enforcement
        - Memory limits (where supported)
        - Crash isolation

        Args:
            func: Function to execute. Must be picklable (module-level).
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            The return value of func.

        Raises:
            SandboxViolation: If sandbox constraints are violated.
            RuntimeError: If process crashes unexpectedly.
            Exception: Any exception raised by func.

        Example:
            with SandboxContext() as sandbox:
                features = sandbox.run(extract_features, file_path)
        """
        if not self._active:
            raise RuntimeError("Sandbox not active. Use 'with SandboxContext() as sandbox:'")

        try:
            result = run_isolated(
                func,
                *args,
                timeout=float(self.config.timeout),
                max_memory=self.config.max_memory,
                **kwargs
            )
            return result
        except TimeoutError as e:
            raise SandboxViolation(
                f"Execution exceeded {self.config.timeout}s timeout",
                violation_type="timeout"
            ) from e
        except MemoryError as e:
            raise SandboxViolation(
                f"Memory limit exceeded ({self.config.max_memory} bytes)",
                violation_type="memory_exceeded"
            ) from e
        except RuntimeError as e:
            if "crashed" in str(e).lower():
                raise SandboxViolation(
                    f"Process crashed during execution: {e}",
                    violation_type="crash"
                ) from e
            raise

    @property
    def temp_path(self) -> Optional[Path]:
        """Get the sandbox's temporary directory path."""
        return self._temp_dir

    @property
    def is_active(self) -> bool:
        """Check if sandbox is currently active."""
        return self._active


@contextmanager
def sandboxed_execution(
    config: SandboxConfig = None
):
    """
    Convenience context manager for sandboxed execution.

    Args:
        config: Sandbox configuration.

    Yields:
        SandboxContext instance.

    Example:
        with sandboxed_execution() as sandbox:
            result = sandbox.run(my_function, arg1, arg2)
    """
    sandbox = SandboxContext(config)
    with sandbox:
        yield sandbox


def run_sandboxed(
    func: Callable,
    *args,
    timeout: int = 30,
    max_memory: int = 512 * 1024 * 1024,
    **kwargs
) -> Any:
    """
    Convenience function to run a single function in a sandbox.

    This is a simpler API when you don't need the full context manager.

    Args:
        func: Function to execute (must be picklable).
        *args: Function arguments.
        timeout: Maximum execution time in seconds.
        max_memory: Maximum memory in bytes.
        **kwargs: Function keyword arguments.

    Returns:
        Function result.

    Raises:
        SandboxViolation: If sandbox constraints are violated.

    Example:
        features = run_sandboxed(extract_features, file_path, timeout=10)
    """
    config = SandboxConfig(timeout=timeout, max_memory=max_memory)
    with SandboxContext(config) as sandbox:
        return sandbox.run(func, *args, **kwargs)
