"""
Tests for malvec.isolation.
"""

import pytest
import sys
import time
from malvec.isolation import run_isolated

# Helper functions must be module-level to be picklable on Windows
def _echo(x):
    return x

def _sleep(seconds):
    time.sleep(seconds)
    return "slept"

def _crash():
    sys.exit(99)

def _raise_error():
    raise ValueError("Expected error")

class TestIsolation:
    
    def test_success(self):
        result = run_isolated(_echo, "hello", timeout=1)
        assert result == "hello"

    def test_timeout(self):
        start = time.time()
        with pytest.raises(TimeoutError):
            # Run for 2s, timeout at 0.5s
            run_isolated(_sleep, 2, timeout=0.5)
        duration = time.time() - start
        assert duration < 1.0 # Should be close to 0.5

    def test_crash_handling(self):
        """Verify that process exit/crash is caught."""
        with pytest.raises(RuntimeError) as excinfo:
            run_isolated(_crash, timeout=1)
        assert "Exit Code: 99" in str(excinfo.value)

    def test_exception_propagation(self):
        with pytest.raises(ValueError) as excinfo:
            run_isolated(_raise_error, timeout=1)
        assert "Expected error" in str(excinfo.value)
