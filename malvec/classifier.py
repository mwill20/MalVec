"""
K-NN Classifier for MalVec.

Provides malware classification using k-nearest neighbors voting
on embedding similarity.

Features:
- Majority voting (simple vote by count)
- Weighted voting (weight by similarity)
- Confidence scoring (fraction of neighbors agreeing)
- Manual review flagging (low confidence → human review)

Architecture:
    Query Embedding → VectorIndex → k Neighbors → Vote → Prediction
                                                      → Confidence
                                                      → Review Flag

Uses parallel label array (not stored in VectorIndex) for separation
of concerns: VectorIndex handles vectors, Classifier handles labels.
"""

from dataclasses import dataclass
from typing import Optional, Union, Tuple, Dict
import numpy as np

from malvec.vector_store import VectorIndex


@dataclass
class ClassifierConfig:
    """Configuration for the KNNClassifier.
    
    Attributes:
        k: Number of neighbors to use for voting
        vote_method: 'majority' or 'weighted'
        confidence_threshold: Below this, flag for manual review
        embedding_dim: Dimension of embeddings (must match index)
    """
    k: int = 5
    vote_method: str = 'majority'  # 'majority' or 'weighted'
    confidence_threshold: float = 0.6
    embedding_dim: int = 384


class KNNClassifier:
    """
    K-Nearest Neighbors classifier for malware detection.
    
    Uses VectorIndex for efficient neighbor search, maintains parallel
    label array for classification.
    
    Example:
        >>> clf = KNNClassifier()
        >>> clf.fit(train_embeddings, train_labels)
        >>> predictions = clf.predict(query_embeddings)
        >>> result = clf.predict_with_review(query_embeddings)
        >>> if result['needs_review'].any():
        ...     print("Some samples need manual review")
    """
    
    def __init__(self, config: Optional[ClassifierConfig] = None):
        """
        Initialize the KNNClassifier.
        
        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or ClassifierConfig()
        self._index: Optional[VectorIndex] = None
        self._labels: Optional[np.ndarray] = None
    
    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> 'KNNClassifier':
        """
        Fit classifier by indexing embeddings and storing labels.
        
        Args:
            embeddings: Training embeddings of shape (n_samples, embedding_dim)
            labels: Training labels of shape (n_samples,)
            
        Returns:
            self (for method chaining)
        """
        # Validate inputs
        if len(embeddings) != len(labels):
            raise ValueError(
                f"Embeddings and labels must have same length. "
                f"Got {len(embeddings)} and {len(labels)}"
            )
        
        # Create and populate index
        from malvec.vector_store import VectorIndexConfig
        
        index_config = VectorIndexConfig(
            embedding_dim=embeddings.shape[1],
            metric='cosine'
        )
        self._index = VectorIndex(index_config)
        self._index.add(embeddings)
        
        # Store labels as parallel array
        self._labels = np.array(labels)
        
        return self
    
    def predict(
        self, 
        query: np.ndarray, 
        return_confidence: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict class labels for query embeddings.
        
        Args:
            query: Query embedding(s) of shape (n, embedding_dim) or (embedding_dim,)
            return_confidence: If True, also return confidence scores
            
        Returns:
            predictions: Class labels of shape (n,)
            confidences: (optional) Confidence scores of shape (n,)
        """
        if self._index is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        
        # Handle single query
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Find k nearest neighbors
        similarities, indices = self._index.search(query, k=self.config.k)
        
        # Get neighbor labels
        neighbor_labels = self._labels[indices]  # shape: (n_queries, k)
        
        # Vote for predictions
        predictions = self._vote(neighbor_labels, similarities)
        
        if return_confidence:
            confidences = self._compute_confidence(neighbor_labels, predictions)
            return predictions, confidences
        
        return predictions
    
    def predict_with_review(self, query: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict with confidence and manual review flag.
        
        Args:
            query: Query embedding(s)
            
        Returns:
            Dict with:
                'predictions': Class labels
                'confidences': Confidence scores
                'needs_review': Boolean flags (True if below threshold)
        """
        predictions, confidences = self.predict(query, return_confidence=True)
        needs_review = confidences < self.config.confidence_threshold
        
        return {
            'predictions': predictions,
            'confidences': confidences,
            'needs_review': needs_review,
        }
    
    def _vote(
        self, 
        neighbor_labels: np.ndarray, 
        similarities: np.ndarray
    ) -> np.ndarray:
        """
        Vote to determine predictions.
        
        Args:
            neighbor_labels: Labels of neighbors, shape (n_queries, k)
            similarities: Similarity scores, shape (n_queries, k)
            
        Returns:
            predictions: Predicted class for each query
        """
        n_queries = neighbor_labels.shape[0]
        predictions = np.zeros(n_queries, dtype=int)
        
        for i in range(n_queries):
            labels_i = neighbor_labels[i]
            
            if self.config.vote_method == 'majority':
                # Simple majority vote
                predictions[i] = self._majority_vote(labels_i)
            elif self.config.vote_method == 'weighted':
                # Weight by similarity
                weights_i = similarities[i]
                predictions[i] = self._weighted_vote(labels_i, weights_i)
            else:
                raise ValueError(f"Unknown vote method: {self.config.vote_method}")
        
        return predictions
    
    def _majority_vote(self, labels: np.ndarray) -> int:
        """Simple majority vote."""
        # Count votes for each class
        counts = np.bincount(labels.astype(int))
        return counts.argmax()
    
    def _weighted_vote(self, labels: np.ndarray, weights: np.ndarray) -> int:
        """Vote weighted by similarity."""
        # Ensure weights are positive (they should be for cosine similarity)
        weights = np.maximum(weights, 0)
        
        # Sum weights for each class
        unique_labels = np.unique(labels)
        class_weights = {}
        
        for label in unique_labels:
            mask = labels == label
            class_weights[int(label)] = weights[mask].sum()
        
        # Return class with highest total weight
        return max(class_weights, key=class_weights.get)
    
    def _compute_confidence(
        self, 
        neighbor_labels: np.ndarray, 
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        Compute prediction confidence.
        
        Confidence = fraction of k neighbors agreeing with prediction.
        
        Args:
            neighbor_labels: Labels of neighbors, shape (n_queries, k)
            predictions: Predicted class for each query
            
        Returns:
            confidences: Confidence scores in [0, 1]
        """
        n_queries = neighbor_labels.shape[0]
        k = neighbor_labels.shape[1]
        confidences = np.zeros(n_queries, dtype=np.float32)
        
        for i in range(n_queries):
            # Count how many neighbors agree with prediction
            agreeing = (neighbor_labels[i] == predictions[i]).sum()
            confidences[i] = agreeing / k
        
        return confidences
    
    def get_info(self) -> dict:
        """Return classifier configuration and state."""
        return {
            'k': self.config.k,
            'vote_method': self.config.vote_method,
            'confidence_threshold': self.config.confidence_threshold,
            'is_fitted': self._index is not None,
            'n_samples': self._index.size() if self._index else 0,
        }


# Module exports
__all__ = [
    'ClassifierConfig',
    'KNNClassifier',
]
