#!/usr/bin/env python
"""
MalVec Model Info CLI.

Display information about a trained MalVec model.

Usage:
    python -m malvec.cli.info --model ./model
"""

import argparse
import sys
import os
import json
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Show MalVec model information',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show model info
    python -m malvec.cli.info --model ./model
    
    # JSON output
    python -m malvec.cli.info --model ./model --json
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to model directory'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    model_path = Path(args.model)
    model_prefix = str(model_path / 'model')
    
    # Check model exists
    meta_path = f"{model_prefix}.meta"
    if not os.path.exists(meta_path):
        print(f"Error: Model not found at {model_path}", file=sys.stderr)
        print(f"Expected file: {meta_path}", file=sys.stderr)
        return 1
    
    # Load metadata
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Load index config
    config_path = f"{model_prefix}.config"
    with open(config_path, 'r') as f:
        index_config = json.load(f)
    
    # Compute file sizes
    files = {
        'index': f"{model_prefix}.index",
        'config': f"{model_prefix}.config",
        'labels': f"{model_prefix}.labels.npy",
        'meta': f"{model_prefix}.meta",
    }
    
    file_sizes = {}
    total_size = 0
    for name, path in files.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            file_sizes[name] = size
            total_size += size
    
    # Build info dict
    info = {
        'path': str(model_path.absolute()),
        'version': meta.get('version', 'unknown'),
        'samples': {
            'total': meta['n_samples'],
            'benign': meta['n_benign'],
            'malware': meta['n_malware'],
        },
        'config': {
            'k': meta['k'],
            'threshold': meta['threshold'],
            'embedding_dim': meta['embedding_dim'],
            'metric': index_config.get('metric', 'cosine'),
        },
        'files': file_sizes,
        'total_size_bytes': total_size,
        'total_size_mb': round(total_size / 1024 / 1024, 2),
    }
    
    # Output
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"MalVec Model Info")
        print(f"=================")
        print(f"Path: {info['path']}")
        print(f"Version: {info['version']}")
        print()
        print(f"Samples:")
        print(f"  Total: {info['samples']['total']}")
        print(f"  Benign: {info['samples']['benign']}")
        print(f"  Malware: {info['samples']['malware']}")
        print()
        print(f"Configuration:")
        print(f"  k: {info['config']['k']}")
        print(f"  Threshold: {info['config']['threshold']}")
        print(f"  Embedding Dim: {info['config']['embedding_dim']}")
        print(f"  Metric: {info['config']['metric']}")
        print()
        print(f"Files:")
        for name, size in info['files'].items():
            print(f"  {name}: {size:,} bytes")
        print(f"  Total: {info['total_size_mb']} MB")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
