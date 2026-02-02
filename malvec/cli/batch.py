#!/usr/bin/env python
"""
MalVec Batch Classification CLI.

Classify multiple samples and output results to CSV or JSON.

Usage:
    python -m malvec.cli.batch --model ./model --output results.csv
"""

import argparse
import sys
import os
import json
import csv
import numpy as np
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch classify samples with MalVec',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Classify 100 samples, output to CSV
    python -m malvec.cli.batch --model ./model --max-samples 100 --output results.csv
    
    # Output to JSON
    python -m malvec.cli.batch --model ./model --output results.json --format json
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to model directory'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output file (CSV or JSON)'
    )
    
    parser.add_argument(
        '--max-samples', '-n',
        type=int,
        default=100,
        help='Maximum samples to classify (default: 100)'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['csv', 'json'],
        default='csv',
        help='Output format (default: csv)'
    )
    
    parser.add_argument(
        '--review-only',
        action='store_true',
        help='Only output samples that need review'
    )
    
    return parser.parse_args()


def load_model(model_path: Path):
    """Load a trained model from disk."""
    from malvec.vector_store import VectorIndex
    from malvec.classifier import KNNClassifier, ClassifierConfig
    
    model_prefix = str(model_path / 'model')
    
    with open(f"{model_prefix}.meta", 'r') as f:
        meta = json.load(f)
    
    index = VectorIndex.load(model_prefix)
    labels = np.load(f"{model_prefix}.labels.npy")
    
    config = ClassifierConfig(
        k=meta['k'],
        confidence_threshold=meta['threshold'],
        embedding_dim=meta['embedding_dim']
    )
    clf = KNNClassifier(config)
    clf._index = index
    clf._labels = labels
    
    return clf, meta


def main():
    """Main batch classification function."""
    args = parse_args()
    
    model_path = Path(args.model)
    output_path = Path(args.output)
    
    # Check model exists
    if not (model_path / 'model.meta').exists():
        print(f"Error: Model not found at {model_path}", file=sys.stderr)
        return 1
    
    # Load model
    clf, meta = load_model(model_path)
    
    # Load samples
    from malvec.ember_loader import load_ember_features
    from malvec.embedder import EmbeddingGenerator, EmbeddingConfig
    
    X, y = load_ember_features(max_samples=args.max_samples)
    
    # Generate embeddings
    config = EmbeddingConfig(
        embedding_dim=meta['embedding_dim'],
        normalize=True
    )
    generator = EmbeddingGenerator(config)
    embeddings = generator.generate(X)
    
    # Predict
    result = clf.predict_with_review(embeddings)
    predictions = result['predictions']
    confidences = result['confidences']
    needs_review = result['needs_review']
    
    # Build results
    results = []
    for i in range(len(X)):
        record = {
            'sample_index': i,
            'prediction': 'malware' if predictions[i] == 1 else 'benign',
            'confidence': float(confidences[i]),
            'needs_review': bool(needs_review[i]),
            'true_label': 'malware' if y[i] == 1 else 'benign',
            'correct': predictions[i] == y[i],
        }
        
        if args.review_only and not needs_review[i]:
            continue
        
        results.append(record)
    
    # Write output
    if args.format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        with open(output_path, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
    
    # Summary
    n_correct = sum(1 for r in results if r['correct'])
    n_review = sum(1 for r in results if r['needs_review'])
    
    print(f"Batch classification complete")
    print(f"  Samples: {len(results)}")
    print(f"  Correct: {n_correct}/{len(results)}")
    print(f"  Need Review: {n_review}/{len(results)}")
    print(f"  Output: {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
