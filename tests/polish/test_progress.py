"""
Tests for ProgressReporter module.

Tests progress indicators, status messages, and fallback behavior.
"""

import sys
import pytest
from io import StringIO
from unittest.mock import patch, MagicMock


class TestProgressReporter:
    """Tests for ProgressReporter class."""

    def test_init_verbose_true(self):
        """Test initialization with verbose=True."""
        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=True)
        assert progress.verbose is True

    def test_init_verbose_false(self):
        """Test initialization with verbose=False."""
        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=False)
        assert progress.verbose is False

    def test_task_silent_mode(self):
        """Test task() in silent mode does nothing."""
        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=False)
        results = []

        with progress.task("Test task", total=10) as update:
            for i in range(10):
                update(i + 1)
                results.append(i)

        assert len(results) == 10

    def test_task_with_total(self):
        """Test task() with determinate progress."""
        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=True)
        progress._has_rich = False  # Force fallback mode

        results = []
        with progress.task("Test task", total=10) as update:
            for i in range(10):
                update(i + 1)
                results.append(i)

        assert len(results) == 10

    def test_task_without_total(self):
        """Test task() with indeterminate progress (spinner)."""
        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=True)
        progress._has_rich = False  # Force fallback

        with progress.task("Indeterminate task") as update:
            for i in range(5):
                update(i)

    def test_status_message(self, capsys):
        """Test status() prints success message."""
        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=True)
        progress._has_rich = False  # Force fallback

        progress.status("Operation succeeded")

        captured = capsys.readouterr()
        assert "Operation succeeded" in captured.out
        assert "[OK]" in captured.out

    def test_status_silent_mode(self, capsys):
        """Test status() suppressed in silent mode."""
        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=False)
        progress.status("Should not appear")

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_error_message(self, capsys):
        """Test error() prints error message."""
        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=True)
        progress._has_rich = False  # Force fallback

        progress.error("Something failed")

        captured = capsys.readouterr()
        assert "Something failed" in captured.err
        assert "[ERROR]" in captured.err

    def test_error_always_shown(self, capsys):
        """Test error() shown even in silent mode."""
        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=False)
        progress._has_rich = False

        progress.error("Critical error")

        captured = capsys.readouterr()
        assert "Critical error" in captured.err

    def test_warning_message(self, capsys):
        """Test warning() prints warning message."""
        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=True)
        progress._has_rich = False

        progress.warning("Be careful")

        captured = capsys.readouterr()
        assert "Be careful" in captured.out
        assert "[WARN]" in captured.out

    def test_warning_silent_mode(self, capsys):
        """Test warning() suppressed in silent mode."""
        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=False)
        progress.warning("Should not appear")

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_info_message(self, capsys):
        """Test info() prints info message."""
        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=True)
        progress._has_rich = False

        progress.info("Information")

        captured = capsys.readouterr()
        assert "Information" in captured.out
        assert "[INFO]" in captured.out

    def test_print_header(self, capsys):
        """Test print_header() shows ASCII art."""
        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=True)
        progress._has_rich = False

        progress.print_header()

        captured = capsys.readouterr()
        # Header has "Malware Vector Classification System" text
        assert "Malware" in captured.out or "Classification" in captured.out

    def test_print_header_silent_mode(self, capsys):
        """Test print_header() suppressed in silent mode."""
        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=False)
        progress.print_header()

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_print_table(self, capsys):
        """Test print_table() formats data."""
        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=True)
        progress._has_rich = False

        progress.print_table("Test Summary", {
            "Key1": "Value1",
            "Key2": 42,
            "Key3": 3.14
        })

        captured = capsys.readouterr()
        assert "Test Summary" in captured.out
        assert "Key1" in captured.out
        assert "Value1" in captured.out
        assert "42" in captured.out

    def test_print_table_silent_mode(self, capsys):
        """Test print_table() suppressed in silent mode."""
        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=False)
        progress.print_table("Summary", {"key": "value"})

        captured = capsys.readouterr()
        assert captured.out == ""


class TestProgressReporterWithRich:
    """Tests for ProgressReporter with rich library."""

    @pytest.fixture
    def mock_rich(self):
        """Mock rich library availability."""
        with patch('malvec.progress.HAS_RICH', True):
            yield

    def test_rich_detection(self):
        """Test rich library detection."""
        from malvec.progress import _check_rich_available

        # This will return True if rich is installed, False otherwise
        result = _check_rich_available()
        assert isinstance(result, bool)

    def test_task_with_rich_available(self, mock_rich):
        """Test task() uses rich when available."""
        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=True)

        # Should not raise even if rich progress is complex
        with progress.task("Test", total=5) as update:
            for i in range(5):
                update(i + 1)


class TestProgressReporterFallback:
    """Tests for fallback behavior without rich."""

    def test_fallback_without_rich(self, monkeypatch):
        """Test fallback mode when rich not available."""
        # Simulate rich not available
        monkeypatch.setattr('malvec.progress.HAS_RICH', False)

        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=True)
        assert not progress._has_rich

    def test_simple_progress_output(self, capsys, monkeypatch):
        """Test simple progress prints percentage milestones."""
        monkeypatch.setattr('malvec.progress.HAS_RICH', False)

        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=True)

        with progress.task("Processing", total=100) as update:
            for i in range(100):
                update(i + 1)

        captured = capsys.readouterr()
        assert "Processing" in captured.out
        # Should print at 10% intervals
        assert "10%" in captured.out or "20%" in captured.out or "Done" in captured.out


class TestGetProgressReporter:
    """Tests for get_progress_reporter helper."""

    def test_get_progress_reporter(self):
        """Test getting default progress reporter."""
        from malvec.progress import get_progress_reporter

        reporter = get_progress_reporter(verbose=True)

        assert reporter is not None
        assert reporter.verbose is True

    def test_get_progress_reporter_singleton(self):
        """Test that default reporter is reused."""
        from malvec import progress

        # Reset singleton
        progress._default_reporter = None

        reporter1 = progress.get_progress_reporter(verbose=True)
        reporter2 = progress.get_progress_reporter(verbose=True)

        assert reporter1 is reporter2


class TestProgressIntegration:
    """Integration tests for progress reporting."""

    def test_complete_workflow(self, capsys):
        """Test complete progress workflow."""
        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=True)
        progress._has_rich = False  # Force fallback for predictable output

        # Simulate a workflow
        progress.print_header()
        progress.status("Starting operation")

        with progress.task("Processing items", total=5) as update:
            for i in range(5):
                update(i + 1)

        progress.warning("Minor issue detected")
        progress.status("Operation complete")

        progress.print_table("Results", {
            "Items processed": 5,
            "Success rate": "100%"
        })

        captured = capsys.readouterr()
        assert "Starting operation" in captured.out
        assert "Operation complete" in captured.out

    def test_nested_tasks_not_supported(self):
        """Test that nested tasks don't break."""
        from malvec.progress import ProgressReporter

        progress = ProgressReporter(verbose=True)
        progress._has_rich = False

        # Nested tasks should work (not truly nested, sequential)
        with progress.task("Outer", total=2) as outer_update:
            outer_update(1)
            with progress.task("Inner", total=3) as inner_update:
                for i in range(3):
                    inner_update(i + 1)
            outer_update(2)
