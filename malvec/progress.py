"""
MalVec Progress Reporting Module.

Provides user-friendly progress indicators for long-running operations.
Uses 'rich' library for beautiful terminal output when available,
falls back to simple print statements otherwise.

Usage:
    progress = ProgressReporter(verbose=True)

    with progress.task("Processing files", total=100) as update:
        for i in range(100):
            # Do work
            update(i + 1)

    progress.status("Operation complete")
"""

import sys
from contextlib import contextmanager
from typing import Iterator, Callable, Optional


def _check_rich_available() -> bool:
    """Check if rich library is available."""
    try:
        import rich
        return True
    except ImportError:
        return False


HAS_RICH = _check_rich_available()


class ProgressReporter:
    """
    User-friendly progress reporting.

    Uses 'rich' library for beautiful terminal output with progress bars,
    spinners, and color. Falls back to simple print statements if rich
    is not installed.

    Attributes:
        verbose: Whether to show progress output.

    Example:
        progress = ProgressReporter(verbose=True)

        with progress.task("Loading data", total=1000) as update:
            for i, item in enumerate(data):
                process(item)
                update(i + 1)

        progress.status("Loading complete!")
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize progress reporter.

        Args:
            verbose: Whether to show progress output. If False, all
                    output is suppressed (useful for scripting).
        """
        self.verbose = verbose
        self._has_rich = HAS_RICH
        self._console = None

        if self._has_rich and self.verbose:
            from rich.console import Console
            self._console = Console()

    @contextmanager
    def task(
        self,
        description: str,
        total: Optional[int] = None
    ) -> Iterator[Callable[[int], None]]:
        """
        Context manager for a progress task.

        Args:
            description: Task description shown to user.
            total: Total number of steps (for percentage). If None,
                  shows indeterminate spinner.

        Yields:
            Update function: call with current progress (1-indexed).

        Example:
            with progress.task("Processing", total=100) as update:
                for i in range(100):
                    do_work()
                    update(i + 1)
        """
        if not self.verbose:
            # Silent mode - no output
            yield lambda _: None
            return

        if self._has_rich and total is not None:
            # Rich progress bar
            yield from self._rich_progress(description, total)
        elif self._has_rich:
            # Rich spinner (indeterminate)
            yield from self._rich_spinner(description)
        else:
            # Fallback simple progress
            yield from self._simple_progress(description, total)

    def _rich_progress(
        self,
        description: str,
        total: int
    ) -> Iterator[Callable[[int], None]]:
        """Rich library progress bar."""
        from rich.progress import (
            Progress, SpinnerColumn, TextColumn,
            BarColumn, TaskProgressColumn, TimeRemainingColumn
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self._console,
        ) as progress:
            task_id = progress.add_task(description, total=total)

            def update(current: int):
                progress.update(task_id, completed=current)

            yield update

    def _rich_spinner(self, description: str) -> Iterator[Callable[[int], None]]:
        """Rich library spinner for indeterminate progress."""
        from rich.progress import Progress, SpinnerColumn, TextColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            console=self._console,
        ) as progress:
            progress.add_task(description, total=None)

            # Update function does nothing for spinner
            yield lambda _: None

    def _simple_progress(
        self,
        description: str,
        total: Optional[int]
    ) -> Iterator[Callable[[int], None]]:
        """Simple fallback progress without rich."""
        print(f"{description}...", flush=True)

        last_pct = -1

        def update(current: int):
            nonlocal last_pct
            if total and total > 0:
                pct = int(100 * current / total)
                # Only print at 10% intervals to avoid spam
                if pct >= last_pct + 10:
                    print(f"  {pct}% complete", flush=True)
                    last_pct = pct

        yield update
        print(f"  Done: {description}", flush=True)

    def status(self, message: str):
        """
        Print a success status message.

        Args:
            message: Message to display.
        """
        if not self.verbose:
            return

        if self._has_rich:
            self._console.print(f"[bold green]\u2713[/bold green] {message}")
        else:
            print(f"[OK] {message}")

    def error(self, message: str):
        """
        Print an error message.

        Args:
            message: Error message to display.
        """
        if self._has_rich:
            from rich.console import Console
            console = Console(stderr=True)
            console.print(f"[bold red]\u2717[/bold red] {message}", style="red")
        else:
            print(f"[ERROR] {message}", file=sys.stderr)

    def warning(self, message: str):
        """
        Print a warning message.

        Args:
            message: Warning message to display.
        """
        if not self.verbose:
            return

        if self._has_rich:
            self._console.print(f"[bold yellow]\u26a0[/bold yellow] {message}")
        else:
            print(f"[WARN] {message}")

    def info(self, message: str):
        """
        Print an informational message.

        Args:
            message: Info message to display.
        """
        if not self.verbose:
            return

        if self._has_rich:
            self._console.print(f"[bold blue]\u2139[/bold blue] {message}")
        else:
            print(f"[INFO] {message}")

    def print_header(self, title: str = "MalVec"):
        """
        Print ASCII art header.

        Args:
            title: Title text (unused, for future customization).
        """
        if not self.verbose:
            return

        header = """
    \u2588\u2588\u2588\u2557   \u2588\u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2557    \u2588\u2588\u2557   \u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2588\u2557
    \u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2551    \u2588\u2588\u2551   \u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d\u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d
    \u2588\u2588\u2554\u2588\u2588\u2588\u2588\u2554\u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2551\u2588\u2588\u2551    \u2588\u2588\u2551   \u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2557  \u2588\u2588\u2551
    \u2588\u2588\u2551\u255a\u2588\u2588\u2554\u255d\u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2551\u2588\u2588\u2551    \u255a\u2588\u2588\u2557 \u2588\u2588\u2554\u255d\u2588\u2588\u2554\u2550\u2550\u255d  \u2588\u2588\u2551
    \u2588\u2588\u2551 \u255a\u2550\u255d \u2588\u2588\u2551\u2588\u2588\u2551  \u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u255a\u2588\u2588\u2588\u2588\u2554\u255d \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u255a\u2588\u2588\u2588\u2588\u2588\u2588\u2557
    \u255a\u2550\u255d     \u255a\u2550\u255d\u255a\u2550\u255d  \u255a\u2550\u255d\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d \u255a\u2550\u2550\u2550\u255d  \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d \u255a\u2550\u2550\u2550\u2550\u2550\u255d

    Malware Vector Classification System
"""
        if self._has_rich:
            self._console.print(header, style="bold cyan")
        else:
            print(header)

    def print_table(self, title: str, data: dict):
        """
        Print a summary table.

        Args:
            title: Table title.
            data: Dictionary of key-value pairs.
        """
        if not self.verbose:
            return

        if self._has_rich:
            from rich.table import Table

            table = Table(title=title)
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="green")

            for key, value in data.items():
                table.add_row(str(key), str(value))

            self._console.print(table)
        else:
            print(f"\n{'=' * 60}")
            print(title)
            print('=' * 60)
            for key, value in data.items():
                print(f"  {key}: {value}")
            print('=' * 60)


# Module-level convenience instance
_default_reporter: Optional[ProgressReporter] = None


def get_progress_reporter(verbose: bool = True) -> ProgressReporter:
    """
    Get or create the default progress reporter.

    Args:
        verbose: Whether to show output.

    Returns:
        ProgressReporter instance.
    """
    global _default_reporter
    if _default_reporter is None:
        _default_reporter = ProgressReporter(verbose=verbose)
    return _default_reporter
