#!/usr/bin/env python
"""
MalVec Archive CLI.

Create and manage .malvec model archives for easy distribution.

Commands:
    archive: Create .malvec archive from model directory
    extract: Extract .malvec archive to directory
    inspect: Show model metadata without extracting

Usage:
    python -m malvec.cli.archive create --model ./model --output model.malvec
    python -m malvec.cli.archive inspect model.malvec
    python -m malvec.cli.archive extract model.malvec --output ./extracted
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Manage MalVec model archives',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
    create   Create .malvec archive from model directory
    inspect  Show model metadata without extracting
    extract  Extract .malvec archive to directory
    list     List archive contents

Examples:
    # Create archive
    python -m malvec.cli.archive create --model ./model --output model.malvec

    # Inspect archive
    python -m malvec.cli.archive inspect model.malvec

    # Extract archive
    python -m malvec.cli.archive extract model.malvec --output ./model_dir

    # List contents
    python -m malvec.cli.archive list model.malvec
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Create command
    create_parser = subparsers.add_parser('create', help='Create archive')
    create_parser.add_argument(
        '--model', '-m',
        required=True,
        type=str,
        help='Model directory to archive'
    )
    create_parser.add_argument(
        '--output', '-o',
        required=True,
        type=str,
        help='Output .malvec file'
    )
    create_parser.add_argument(
        '--compression', '-c',
        default='gz',
        choices=['gz', 'bz2', 'xz', 'none'],
        help='Compression type (default: gz)'
    )

    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect archive')
    inspect_parser.add_argument(
        'archive',
        type=str,
        help='.malvec archive file'
    )

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract archive')
    extract_parser.add_argument(
        'archive',
        type=str,
        help='.malvec archive file'
    )
    extract_parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: current directory)'
    )

    # List command
    list_parser = subparsers.add_parser('list', help='List archive contents')
    list_parser.add_argument(
        'archive',
        type=str,
        help='.malvec archive file'
    )

    return parser.parse_args()


def cmd_create(args):
    """Create archive from model directory."""
    from malvec.model import MalVecModel
    from malvec.progress import ProgressReporter

    progress = ProgressReporter(verbose=True)

    model_dir = Path(args.model)
    output_path = Path(args.output)

    if not model_dir.exists():
        progress.error(f"Model directory not found: {model_dir}")
        return 1

    # Handle 'none' compression
    compression = '' if args.compression == 'none' else args.compression

    try:
        MalVecModel.save_archive(model_dir, output_path, compression=compression)
        progress.status(f"Archive created: {output_path}")
        return 0
    except Exception as e:
        progress.error(f"Failed to create archive: {e}")
        return 1


def cmd_inspect(args):
    """Inspect archive and show metadata."""
    from malvec.model import MalVecModel
    from malvec.progress import ProgressReporter, HAS_RICH

    progress = ProgressReporter(verbose=True)

    archive_path = Path(args.archive)

    if not archive_path.exists():
        progress.error(f"Archive not found: {archive_path}")
        return 1

    try:
        meta = MalVecModel.inspect(archive_path)

        if HAS_RICH:
            from rich.table import Table
            from rich.console import Console

            table = Table(title=f"Model: {archive_path.name}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            for key, value in meta.items():
                table.add_row(str(key), str(value))

            Console().print(table)
        else:
            print(f"\nModel: {archive_path.name}")
            print("=" * 50)
            for key, value in meta.items():
                print(f"  {key}: {value}")
            print("=" * 50)

        return 0
    except Exception as e:
        progress.error(f"Failed to inspect archive: {e}")
        return 1


def cmd_extract(args):
    """Extract archive to directory."""
    from malvec.model import MalVecModel
    from malvec.progress import ProgressReporter

    progress = ProgressReporter(verbose=True)

    archive_path = Path(args.archive)
    output_dir = Path(args.output) if args.output else Path.cwd()

    if not archive_path.exists():
        progress.error(f"Archive not found: {archive_path}")
        return 1

    try:
        progress.status(f"Extracting {archive_path}...")
        extract_dir = MalVecModel.load_archive(archive_path, output_dir)
        progress.status(f"Extracted to: {extract_dir}")
        return 0
    except Exception as e:
        progress.error(f"Failed to extract archive: {e}")
        return 1


def cmd_list(args):
    """List archive contents."""
    from malvec.model import MalVecModel
    from malvec.progress import ProgressReporter, HAS_RICH

    progress = ProgressReporter(verbose=True)

    archive_path = Path(args.archive)

    if not archive_path.exists():
        progress.error(f"Archive not found: {archive_path}")
        return 1

    try:
        contents = MalVecModel.list_contents(archive_path)

        if HAS_RICH:
            from rich.table import Table
            from rich.console import Console

            table = Table(title=f"Contents: {archive_path.name}")
            table.add_column("File", style="cyan")
            table.add_column("Size", style="green", justify="right")

            for item in contents:
                if item['is_file']:
                    size_str = f"{item['size']:,} bytes"
                    table.add_row(item['name'], size_str)

            Console().print(table)
        else:
            print(f"\nContents: {archive_path.name}")
            print("-" * 50)
            for item in contents:
                if item['is_file']:
                    print(f"  {item['name']}: {item['size']:,} bytes")
            print("-" * 50)

        return 0
    except Exception as e:
        progress.error(f"Failed to list archive: {e}")
        return 1


def main():
    """Main entry point."""
    args = parse_args()

    if args.command is None:
        print("Error: No command specified. Use --help for usage.")
        return 1

    commands = {
        'create': cmd_create,
        'inspect': cmd_inspect,
        'extract': cmd_extract,
        'list': cmd_list,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Error: Unknown command '{args.command}'")
        return 1


if __name__ == '__main__':
    sys.exit(main())
