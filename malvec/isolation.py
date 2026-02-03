"""
Process isolation utilities for MalVec.

Provides capabilities to run dangerous code (like binary parsing) in
isolated processes with enforced timeouts and resource limits. This
protects the main application from segfaults (e.g. in LIEF/C++ bindings),
hangs, and resource exhaustion attacks.

Security Features:
- Timeout enforcement (kills process after N seconds)
- Memory limits (prevents memory exhaustion attacks)
- Process isolation (separate process for crash protection)
- Exception propagation (errors returned safely)
"""

import multiprocessing
import sys
import time
from typing import Any, Callable, Optional
from dataclasses import dataclass


@dataclass
class IsolationConfig:
    """Configuration for process isolation."""
    timeout: float = 30.0  # Max execution time (seconds)
    max_memory: int = 512 * 1024 * 1024  # 512MB default

    def __post_init__(self):
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_memory <= 0:
            raise ValueError("max_memory must be positive")


def _set_memory_limit(max_bytes: int) -> bool:
    """
    Set memory limit for the current process.

    Args:
        max_bytes: Maximum memory in bytes.

    Returns:
        True if limit was set, False if not supported on this platform.
    """
    if sys.platform == 'linux':
        try:
            import resource
            # Set soft and hard limits for virtual memory
            resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
            return True
        except (ImportError, ValueError, resource.error):
            return False
    elif sys.platform == 'darwin':
        # macOS: resource limits available but less effective
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
            return True
        except (ImportError, ValueError, resource.error):
            return False
    else:
        # Windows: memory limits require win32api/job objects
        # Best effort - monitoring instead of hard limits
        return False


def _worker(
    q: multiprocessing.Queue,
    func: Callable,
    args: tuple,
    kwargs: dict,
    max_memory: int
):
    """
    Worker function to execute the target function and report result.

    Sets resource limits before execution if supported.
    """
    # Attempt to set memory limit (best effort)
    _set_memory_limit(max_memory)

    try:
        result = func(*args, **kwargs)
        q.put(("OK", result))
    except MemoryError as e:
        q.put(("MEMORY_ERROR", str(e)))
    except Exception as e:
        # We wrap the exception to ensure it survives pickling across processes
        q.put(("ERR", e))


def run_isolated(
    func: Callable,
    *args,
    timeout: float = 30.0,
    max_memory: int = 512 * 1024 * 1024,
    **kwargs
) -> Any:
    """
    Run a function in a separate process with timeout and resource limits.

    This provides defense-in-depth by:
    1. Running in separate process (crash isolation)
    2. Enforcing timeout (prevents infinite loops)
    3. Setting memory limits where supported (prevents exhaustion)

    Args:
        func: The function to run. Must be picklable (module-level functions).
        *args: Positional arguments for func.
        timeout: Maximum execution time in seconds (default 30).
        max_memory: Maximum memory usage in bytes (default 512MB).
        **kwargs: Keyword arguments for func.

    Returns:
        The return value of func.

    Raises:
        TimeoutError: If execution exceeds timeout.
        MemoryError: If memory limit exceeded.
        RuntimeError: If the process crashes (segfault) or exits without result.
        Exception: Any exception raised by func is re-raised here.

    Example:
        >>> def expensive_parse(path):
        ...     return parse_binary(path)
        >>> result = run_isolated(expensive_parse, "/path/to/file", timeout=10)
    """
    # Validate inputs
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if max_memory <= 0:
        raise ValueError("max_memory must be positive")

    # Use 'spawn' or 'fork' depending on OS, but default is usually safe
    # On Windows, 'spawn' is default and required.
    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_worker,
        args=(q, func, args, kwargs, max_memory)
    )

    start_time = time.time()
    p.start()
    p.join(timeout)

    if p.is_alive():
        # Process still running - kill it
        p.terminate()
        time.sleep(0.1)  # Give it a moment
        if p.is_alive():
            # Force kill if still alive
            try:
                p.kill()
            except AttributeError:
                pass  # Python < 3.7
        p.join()  # cleanup
        raise TimeoutError(f"Execution exceeded {timeout}s timeout")

    # Check if we got a result
    if not q.empty():
        status, payload = q.get()
        if status == "OK":
            return payload
        elif status == "MEMORY_ERROR":
            raise MemoryError(f"Memory limit exceeded: {payload}")
        else:
            # Re-raise the exception from the worker
            raise payload
    else:
        # Queue is empty but process finished.
        # This implies a crash (segfault) or hard exit.
        exit_code = p.exitcode
        raise RuntimeError(
            f"Worker process crashed or exited unexpectedly (Exit Code: {exit_code})"
        )


class IsolatedExecutor:
    """
    Reusable executor for isolated function calls.

    Maintains consistent configuration across multiple calls.

    Example:
        >>> executor = IsolatedExecutor(timeout=10, max_memory=256*1024*1024)
        >>> result1 = executor.run(func1, arg1)
        >>> result2 = executor.run(func2, arg2)
    """

    def __init__(self, config: IsolationConfig = None):
        """
        Initialize executor with configuration.

        Args:
            config: Isolation configuration. Uses defaults if not provided.
        """
        self.config = config or IsolationConfig()

    def run(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run a function with configured isolation settings.

        Args:
            func: Function to execute (must be picklable).
            *args: Function arguments.
            **kwargs: Function keyword arguments.

        Returns:
            Function result.

        Raises:
            Same exceptions as run_isolated.
        """
        return run_isolated(
            func,
            *args,
            timeout=self.config.timeout,
            max_memory=self.config.max_memory,
            **kwargs
        )


def is_memory_limit_supported() -> bool:
    """
    Check if memory limits are supported on this platform.

    Returns:
        True if memory limits can be enforced, False otherwise.
    """
    return sys.platform in ('linux', 'darwin')
