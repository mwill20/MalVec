#!/usr/bin/env python
"""
MalVec Evaluation CLI.

Evaluate a trained model's performance on test data.

Usage:
    python -m malvec.cli.evaluate --model ./model --max-samples 1000

Output:
    Accuracy, precision, recall, F1, and confusion matrix.
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
        description='Evaluate MalVec model performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate on 1000 samples
    python -m malvec.cli.evaluate --model ./model --max-samples 1000
    
    # JSON output
    python -m malvec.cli.evaluate --model ./model --json
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to model directory'
    )
    
    parser.add_argument(
        '--max-samples', '-n',
        type=int,
        default=500,
        help='Maximum samples to evaluate (default: 500)'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    
    return parser.parse_args()


def load_model(model_path: Path):
    """Load a trained model from disk."""
    from malvec.vector_store import VectorIndex
    from malvec.classifier import KNNClassifier, ClassifierConfig
    
    model_prefix = str(model_path / 'model')
    
    # Load metadata
    with open(f"{model_prefix}.meta", 'r') as f:
        meta = json.load(f)
    
    # Load index
    index = VectorIndex.load(model_prefix)
    
    # Load labels
    labels = np.load(f"{model_prefix}.labels.npy")
    
    # Create classifier
    config = ClassifierConfig(
        k=meta['k'],
        confidence_threshold=meta['threshold'],
        embedding_dim=meta['embedding_dim']
    )
    clf = KNNClassifier(config)
    clf._index = index
    clf._labels = labels
    
    return clf, meta


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, confidences: np.ndarray, threshold: float):
    """Compute classification metrics."""
    # Basic counts
    n_samples = len(y_true)
    n_correct = (y_true == y_pred).sum()
    
    # Confusion matrix (binary: 0=benign, 1=malware)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    # Metrics
    accuracy = n_correct / n_samples if n_samples > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Review metrics
    needs_review = confidences < threshold
    review_rate = needs_review.sum() / n_samples if n_samples > 0 else 0
    
    # Accuracy on high-confidence samples
    high_conf_mask = ~needs_review
    if high_conf_mask.sum() > 0:
        high_conf_accuracy = (y_true[high_conf_mask] == y_pred[high_conf_mask]).sum() / high_conf_mask.sum()
    else:
        high_conf_accuracy = 0
    
    return {
        'n_samples': int(n_samples),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': {
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
        },
        'review_rate': float(review_rate),
        'n_review': int(needs_review.sum()),
        'high_confidence_accuracy': float(high_conf_accuracy),
        'n_high_confidence': int(high_conf_mask.sum()),
    }


def main():
    """Main evaluation function."""
    args = parse_args()
    
    model_path = Path(args.model)
    
    # Check model exists
    if not (model_path / 'model.meta').exists():
        print(f"Error: Model not found at {model_path}", file=sys.stderr)
        return 1
    
    # Load model
    clf, meta = load_model(model_path)
    
    # Load test data
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
    
    # Compute metrics
    metrics = compute_metrics(y, predictions, confidences, meta['threshold'])
    
    # Output
    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print("MalVec Model Evaluation")
        print("=======================")
        print(f"Model: {model_path}")
        print(f"Samples: {metrics['n_samples']}")
        print()
        print("Performance Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.1%}")
        print(f"  Precision: {metrics['precision']:.1%}")
        print(f"  Recall:    {metrics['recall']:.1%}")
        print(f"  F1 Score:  {metrics['f1']:.1%}")
        print()
        print("Confusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"                Predicted")
        print(f"                Benign  Malware")
        print(f"  Actual Benign    {cm['tn']:4d}     {cm['fp']:4d}")
        print(f"  Actual Malware   {cm['fn']:4d}     {cm['tp']:4d}")
        print()
        print("Review Status:")
        print(f"  Needs Review: {metrics['n_review']}/{metrics['n_samples']} ({metrics['review_rate']:.1%})")
        print(f"  Auto-classified: {metrics['n_high_confidence']}/{metrics['n_samples']}")
        print(f"  High-confidence Accuracy: {metrics['high_confidence_accuracy']:.1%}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
