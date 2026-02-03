#!/usr/bin/env python
"""
MalVec Training CLI.

Train a malware classification model from EMBER features.

Usage:
    python -m malvec.cli.train --output ./model --max-samples 1000

This creates:
    - model.index: FAISS vector index
    - model.config: Index configuration
    - model.labels.npy: Label array
    - model.meta: Training metadata
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train MalVec malware classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train on 1000 samples
    python -m malvec.cli.train --output ./model --max-samples 1000

    # Train with custom k
    python -m malvec.cli.train --output ./model --k 7

    # Train with progress output
    python -m malvec.cli.train --output ./model --verbose
        """
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for model files'
    )

    parser.add_argument(
        '--max-samples', '-n',
        type=int,
        default=10000,
        help='Maximum samples to load (default: 10000)'
    )

    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of neighbors for K-NN (default: 5)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.6,
        help='Confidence threshold for review (default: 0.6)'
    )

    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=384,
        help='Embedding dimension (default: 384)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output with progress bars'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to configuration file (YAML)'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Import progress reporter
    from malvec.progress import ProgressReporter
    progress = ProgressReporter(verbose=args.verbose)

    # Print header if verbose
    if args.verbose:
        progress.print_header()

    # Load config file if specified
    if args.config:
        from malvec.config import MalVecConfig
        try:
            config = MalVecConfig.from_file(Path(args.config))
            progress.status(f"Loaded config from {args.config}")
            # Override args with config values (command line takes precedence)
            if args.k == 5:  # default value
                args.k = config.k
            if args.threshold == 0.6:  # default value
                args.threshold = config.confidence_threshold
            if args.embedding_dim == 384:  # default value
                args.embedding_dim = config.embedding_dim
        except Exception as e:
            progress.error(f"Failed to load config: {e}")
            return 1

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    progress.info(f"Training MalVec model")
    progress.info(f"Output: {output_path}")
    progress.info(f"Max samples: {args.max_samples}")

    # Load EMBER features
    progress.status("Loading EMBER features...")

    from malvec.ember_loader import load_ember_features

    try:
        X, y = load_ember_features(max_samples=args.max_samples)
    except Exception as e:
        progress.error(f"Failed to load EMBER features: {e}")
        progress.warning("Ensure EMBER dataset is available. See docs/USER_GUIDE.md")
        return 1

    n_benign = int(sum(y == 0))
    n_malware = int(sum(y == 1))
    progress.status(f"Loaded {len(X)} samples ({n_benign} benign, {n_malware} malware)")

    # Generate embeddings with progress
    progress.status("Generating embeddings...")

    from malvec.embedder import EmbeddingGenerator, EmbeddingConfig

    embed_config = EmbeddingConfig(
        embedding_dim=args.embedding_dim,
        normalize=True
    )
    generator = EmbeddingGenerator(embed_config)

    # Process in batches with progress
    batch_size = 1000
    embeddings_list = []

    with progress.task("Embedding samples", total=len(X)) as update:
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            emb = generator.generate(batch)
            embeddings_list.append(emb)
            update(min(i + batch_size, len(X)))

    embeddings = np.vstack(embeddings_list)
    progress.status(f"Generated {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]})")

    # Create and fit classifier
    progress.status("Building classifier...")

    from malvec.classifier import KNNClassifier, ClassifierConfig

    clf_config = ClassifierConfig(
        k=args.k,
        confidence_threshold=args.threshold,
        embedding_dim=args.embedding_dim
    )
    clf = KNNClassifier(clf_config)
    clf.fit(embeddings, y)

    progress.status(f"Classifier ready (k={args.k}, threshold={args.threshold})")

    # Save model
    progress.status("Saving model...")

    model_prefix = str(output_path / 'model')

    # Save index
    clf._index.save(model_prefix)

    # Save labels
    labels_path = f"{model_prefix}.labels"
    np.save(labels_path, clf._labels)

    # Save metadata
    meta_path = f"{model_prefix}.meta"
    meta = {
        'k': args.k,
        'threshold': args.threshold,
        'embedding_dim': args.embedding_dim,
        'n_samples': len(y),
        'n_benign': n_benign,
        'n_malware': n_malware,
        'version': '1.0',
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    progress.status(f"Model saved to {output_path}")

    # Print summary table
    progress.print_table("Training Summary", {
        "Samples": f"{len(y):,}",
        "Benign": f"{n_benign:,}",
        "Malware": f"{n_malware:,}",
        "Embedding Dim": args.embedding_dim,
        "K-NN k": args.k,
        "Confidence Threshold": f"{args.threshold:.2f}",
        "Model Path": str(output_path),
    })

    return 0


if __name__ == '__main__':
    sys.exit(main())
