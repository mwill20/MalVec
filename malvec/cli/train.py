#!/usr/bin/env python
"""
MalVec Training CLI.

Train a malware classification model from EMBER features.

Usage:
    python -m malvec.cli.train --output ./model --max-samples 1000

This creates:
    - model.index: FAISS vector index
    - model.config: Index configuration
    - model.labels: Label array
    - model.meta: Training metadata
"""

import argparse
import sys
import os
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
        help='Verbose output'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.verbose:
        print(f"MalVec Training")
        print(f"===============")
        print(f"Output: {output_path}")
        print(f"Max samples: {args.max_samples}")
        print(f"k: {args.k}")
        print()
    
    # Load EMBER features
    if args.verbose:
        print("Loading EMBER features...")
    
    from malvec.ember_loader import load_ember_features
    X, y = load_ember_features(max_samples=args.max_samples)
    
    if args.verbose:
        print(f"  Loaded {len(X)} samples")
        print(f"  Labels: {sum(y==0)} benign, {sum(y==1)} malware")
        print()
    
    # Generate embeddings
    if args.verbose:
        print("Generating embeddings...")
    
    from malvec.embedder import EmbeddingGenerator, EmbeddingConfig
    
    config = EmbeddingConfig(
        embedding_dim=args.embedding_dim,
        normalize=True
    )
    generator = EmbeddingGenerator(config)
    embeddings = generator.generate(X)
    
    if args.verbose:
        print(f"  Generated {embeddings.shape[0]} embeddings")
        print(f"  Dimension: {embeddings.shape[1]}")
        print()
    
    # Create and fit classifier
    if args.verbose:
        print("Building classifier...")
    
    from malvec.classifier import KNNClassifier, ClassifierConfig
    
    clf_config = ClassifierConfig(
        k=args.k,
        confidence_threshold=args.threshold,
        embedding_dim=args.embedding_dim
    )
    clf = KNNClassifier(clf_config)
    clf.fit(embeddings, y)
    
    if args.verbose:
        print(f"  k={args.k}, threshold={args.threshold}")
        print()
    
    # Save model
    if args.verbose:
        print("Saving model...")
    
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
        'n_benign': int(sum(y == 0)),
        'n_malware': int(sum(y == 1)),
        'version': '1.0',
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    if args.verbose:
        print(f"  Saved to {output_path}")
        print(f"  Files: model.index, model.config, model.labels, model.meta")
        print()
    
    print(f"Training complete: {len(y)} samples indexed")
    print(f"Model saved to: {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
