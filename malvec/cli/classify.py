#!/usr/bin/env python
"""
MalVec Classification CLI.

Classify samples using a trained MalVec model.

Usage:
    python -m malvec.cli.classify --model ./model --sample-index 0

Output:
    Prediction, confidence, and review recommendation.
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
        description='Classify samples with MalVec',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Classify sample by index
    python -m malvec.cli.classify --model ./model --sample-index 42
    
    # Show confidence scores
    python -m malvec.cli.classify --model ./model --sample-index 42 --show-confidence
    
    # Show neighbors
    python -m malvec.cli.classify --model ./model --sample-index 42 --show-neighbors
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to model directory'
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--sample-index', '-i',
        type=int,
        help='Index of sample to classify (from EMBER dataset)'
    )
    
    group.add_argument(
        '--file', '-f',
        type=str,
        help='Path to PE file to classify'
    )
    
    parser.add_argument(
        '--show-confidence', '-c',
        action='store_true',
        help='Show confidence score'
    )
    
    parser.add_argument(
        '--show-neighbors', '-n',
        action='store_true',
        help='Show nearest neighbors'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=None,
        help='Override k for this query (default: use model k)'
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
    
    # Check files exist
    required_files = [
        f"{model_prefix}.index",
        f"{model_prefix}.config",
        f"{model_prefix}.labels.npy",
        f"{model_prefix}.meta",
    ]
    
    for f in required_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Model file not found: {f}")
    
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


def main():
    """Main classification function."""
    args = parse_args()
    
    model_path = Path(args.model)
    
    # Load model
    try:
        clf, meta = load_model(model_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    from malvec.embedder import EmbeddingGenerator, EmbeddingConfig
    
    embedding = None
    true_label = None
    sample_identifier = None
    
    # CASE 1: Real File Classification
    if args.file:
        from malvec.validator import InputValidator
        from malvec.features import FeatureExtractor
        
        sample_identifier = args.file
        try:
            # Validate
            file_path = InputValidator.validate(args.file)
            
            # Extract
            print(f"Extracting features from {file_path.name}...", file=sys.stderr)
            extractor = FeatureExtractor()
            raw_features = extractor.extract(file_path)
            
            # Embed
            config = EmbeddingConfig(embedding_dim=meta['embedding_dim'], normalize=True)
            generator = EmbeddingGenerator(config)
            embedding = generator.generate(raw_features)
            
        except Exception as e:
            print(f"Error processing file: {e}", file=sys.stderr)
            return 1

    # CASE 2: Synthetic Sample (Demo/Test)
    elif args.sample_index is not None:
        from malvec.ember_loader import load_ember_features
        
        sample_identifier = f"Index {args.sample_index}"
        
        # Validate sample index against model's training data
        if args.sample_index >= meta['n_samples']:
            print(f"Error: Sample index {args.sample_index} out of range (model has {meta['n_samples']} samples)", file=sys.stderr)
            return 1
        
        # Load the specific sample
        X, y = load_ember_features(max_samples=args.sample_index + 1)
        sample = X[args.sample_index:args.sample_index + 1]
        true_label = y[args.sample_index]
        
        # Generate embedding
        config = EmbeddingConfig(embedding_dim=meta['embedding_dim'], normalize=True)
        generator = EmbeddingGenerator(config)
        embedding = generator.generate(sample)
    
    # Override k if specified
    if args.k:
        clf.config.k = args.k
    
    # Classify
    result = clf.predict_with_review(embedding)
    
    prediction = result['predictions'][0]
    confidence = result['confidences'][0]
    needs_review = result['needs_review'][0]
    
    # Prepare output
    output = {
        'sample': sample_identifier,
        'prediction': 'malware' if prediction == 1 else 'benign',
        'prediction_code': int(prediction),
        'confidence': float(confidence),
        'needs_review': bool(needs_review),
    }
    
    if true_label is not None:
        output['true_label'] = 'malware' if true_label == 1 else 'benign'
        output['correct'] = (prediction == true_label)
    
    # Get neighbors if requested
    if args.show_neighbors:
        k = args.k or clf.config.k
        similarities, indices = clf._index.search(embedding, k=k)
        neighbor_labels = clf._labels[indices[0]]
        
        neighbors = []
        for i, (idx, sim, label) in enumerate(zip(indices[0], similarities[0], neighbor_labels)):
            neighbors.append({
                'rank': i + 1,
                'index': int(idx),
                'similarity': float(sim),
                'label': 'malware' if label == 1 else 'benign',
            })
        output['neighbors'] = neighbors
    
    # Output
    if args.json:
        print(json.dumps(output, indent=2))
    else:
        label_str = 'MALWARE' if prediction == 1 else 'BENIGN'
        review_str = '[!] NEEDS REVIEW' if needs_review else '[OK] AUTO-CLASSIFIED'
        
        print(f"Sample: {sample_identifier}")
        print(f"Prediction: {label_str}")
        
        if args.show_confidence:
            print(f"Confidence: {confidence:.0%}")
        
        print(f"Status: {review_str}")
        
        if true_label is not None:
            print(f"True Label: {'malware' if true_label == 1 else 'benign'}")
            print(f"Correct: {'YES' if output['correct'] else 'NO'}")
        
        if args.show_neighbors:
            print(f"\nNearest Neighbors (k={len(output['neighbors'])}):")
            for n in output['neighbors']:
                print(f"  #{n['rank']}: idx={n['index']}, sim={n['similarity']:.4f}, {n['label']}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
