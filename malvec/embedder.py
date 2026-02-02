"""
Embedding Generator for MalVec.

Converts EMBER feature vectors (2381 dimensions) into dense embeddings
suitable for similarity search and classification.

Approach: Johnson-Lindenstrauss Random Projection
- Projects high-dimensional features to lower dimension
- Preserves pairwise distances with high probability
- Fast (single matrix multiply)
- Uses ALL features (no information loss)

Why NOT sentence-transformers:
- Sentence transformers expect TEXT, not numbers
- Converting numbers to strings loses semantic meaning
- Transformers tokenize "0.123" as a word, not a value
- Random projection is provably correct for numeric features
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from malvec.ember_loader import EMBER_FEATURE_DIM, validate_features


@dataclass
class EmbeddingConfig:
    """Configuration for the EmbeddingGenerator.
    
    Attributes:
        embedding_dim: Output embedding dimension
        random_state: Seed for reproducible projections
        normalize: Whether to L2 normalize embeddings
    """
    embedding_dim: int = 384
    random_state: int = 42
    normalize: bool = True


class EmbeddingGenerator:
    """
    Generate embeddings from EMBER feature vectors using random projection.
    
    Uses Johnson-Lindenstrauss lemma: random projection preserves
    pairwise distances with high probability.
    
    Architecture (simple and correct):
        EMBER (2381 dims) → Random Projection (384 dims) → L2 Normalize
    
    Example:
        >>> generator = EmbeddingGenerator()
        >>> features = np.random.randn(100, 2381).astype(np.float32)
        >>> embeddings = generator.generate(features)
        >>> print(embeddings.shape)  # (100, 384)
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the EmbeddingGenerator.
        
        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or EmbeddingConfig()
        self._projector = None
        self._input_dim = None
    
    def _init_projection(self, input_dim: int):
        """
        Initialize Gaussian random projection.
        
        Uses sklearn's GaussianRandomProjection which properly implements
        the Johnson-Lindenstrauss lemma with correct scaling.
        
        Args:
            input_dim: Input feature dimension (e.g., 2381 for EMBER)
        """
        from sklearn.random_projection import GaussianRandomProjection
        
        self._projector = GaussianRandomProjection(
            n_components=self.config.embedding_dim,
            random_state=self.config.random_state
        )
        # Fit requires a sample to determine input dimension
        # We create a dummy sample just to initialize
        dummy = np.zeros((1, input_dim), dtype=np.float32)
        self._projector.fit(dummy)
        self._input_dim = input_dim
    
    def project_features(self, features: np.ndarray) -> np.ndarray:
        """
        Project features to lower dimension.
        
        Args:
            features: Feature matrix of shape (n_samples, input_dim)
            
        Returns:
            Projected features of shape (n_samples, embedding_dim)
        """
        # Handle single sample
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Initialize projector on first call
        if self._projector is None:
            self._init_projection(features.shape[1])
        
        # Transform using sklearn projector
        projected = self._projector.transform(features)
        
        return projected.astype(np.float32)
    
    def generate(self, features: np.ndarray) -> np.ndarray:
        """
        Generate embeddings from EMBER features.
        
        Pipeline:
            1. Validate input (shape, NaN, Inf)
            2. Project to embedding dimension
            3. L2 normalize (optional, enabled by default)
        
        Args:
            features: Feature matrix of shape (n_samples, 2381) or (2381,)
            
        Returns:
            Embedding matrix of shape (n_samples, embedding_dim)
            
        Raises:
            ValueError: If features have wrong dimension or contain NaN/Inf
        """
        # Handle single sample
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Validate input
        validate_features(features)
        
        # Project to lower dimension
        embeddings = self.project_features(features)
        
        # Normalize if configured
        if self.config.normalize:
            embeddings = self._normalize(embeddings)
        
        return embeddings.astype(np.float32)
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        L2 normalize embeddings for cosine similarity.
        
        After normalization, cosine similarity = dot product,
        which is much faster to compute.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-8)
        return embeddings / norms
    
    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.config.embedding_dim
    
    def get_model_info(self) -> dict:
        """Get information about the embedding model."""
        return {
            'model_name': 'random_projection',
            'model_type': 'Johnson-Lindenstrauss',
            'embedding_dim': self.config.embedding_dim,
            'input_dim': self._input_dim or EMBER_FEATURE_DIM,
            'normalize': self.config.normalize,
            'random_state': self.config.random_state,
        }


# Module exports
__all__ = [
    'EmbeddingConfig',
    'EmbeddingGenerator',
]
