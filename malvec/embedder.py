"""
Embedding Generator for MalVec.

Converts EMBER feature vectors (2381 dimensions) into dense semantic embeddings
suitable for similarity search and classification.

Approach:
1. Project high-dimensional EMBER features to lower dimension
2. Generate embeddings using a transformer-based model
3. Normalize for cosine similarity

Why this approach:
- EMBER features are sparse and high-dimensional (2381)
- Direct embedding of raw features loses structure
- Projection + transformer captures semantic relationships
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from malvec.ember_loader import EMBER_FEATURE_DIM, validate_features


@dataclass
class EmbeddingConfig:
    """Configuration for the EmbeddingGenerator.
    
    Attributes:
        model_name: Name of the embedding model to use
        embedding_dim: Output embedding dimension (determined by model)
        batch_size: Batch size for processing
        normalize: Whether to L2 normalize embeddings
        projection_dim: Intermediate projection dimension
    """
    model_name: str = 'all-MiniLM-L6-v2'
    embedding_dim: Optional[int] = None  # Determined by model
    batch_size: int = 32
    normalize: bool = True
    projection_dim: int = 384  # Project to transformer's expected input


class EmbeddingGenerator:
    """
    Generate embeddings from EMBER feature vectors.
    
    This class converts 2381-dimensional EMBER feature vectors into
    dense embeddings suitable for similarity search.
    
    Architecture:
        EMBER (2381 dims) → Projection (384 dims) → Embedding (384 dims)
    
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
        
        # Initialize projection matrix (learned or random for now)
        self._init_projection()
        
        # Lazy load the embedding model
        self._model = None
        self._embedding_dim = None
    
    def _init_projection(self):
        """Initialize the projection layer.
        
        Projects EMBER's 2381 dimensions to a lower dimension
        that can be processed by transformer models.
        """
        # Random orthogonal projection for dimension reduction
        # This preserves relative distances (Johnson-Lindenstrauss lemma)
        rng = np.random.default_rng(42)  # Reproducible
        
        # Create random matrix and orthogonalize
        random_matrix = rng.standard_normal(
            (EMBER_FEATURE_DIM, self.config.projection_dim)
        ).astype(np.float32)
        
        # QR decomposition for orthogonal projection
        q, _ = np.linalg.qr(random_matrix)
        self._projection_matrix = q.astype(np.float32)
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.config.model_name)
                self._embedding_dim = self._model.get_sentence_embedding_dimension()
            except ImportError:
                # Fallback: use projection as final embedding
                self._model = 'projection_only'
                self._embedding_dim = self.config.projection_dim
    
    def project_features(self, features: np.ndarray) -> np.ndarray:
        """
        Project EMBER features to lower dimension.
        
        Args:
            features: Feature matrix of shape (n_samples, 2381)
            
        Returns:
            Projected features of shape (n_samples, projection_dim)
        """
        # Handle single sample
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Project to lower dimension
        projected = features @ self._projection_matrix
        
        return projected.astype(np.float32)
    
    def generate(self, features: np.ndarray) -> np.ndarray:
        """
        Generate embeddings from EMBER features.
        
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
        
        # Ensure model is loaded
        self._load_model()
        
        # Project features first
        projected = self.project_features(features)
        
        # Generate embeddings in batches
        embeddings = self._generate_batched(projected)
        
        # Normalize if configured
        if self.config.normalize:
            embeddings = self._normalize(embeddings)
        
        return embeddings.astype(np.float32)
    
    def _generate_batched(self, projected: np.ndarray) -> np.ndarray:
        """Generate embeddings in batches for memory efficiency."""
        n_samples = projected.shape[0]
        batch_size = self.config.batch_size
        
        all_embeddings = []
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = projected[start_idx:end_idx]
            
            if self._model == 'projection_only':
                # Fallback: use projected features as embeddings
                batch_embeddings = batch
            else:
                # Use sentence-transformers model
                # Convert to text representation for the model
                batch_embeddings = self._embed_with_model(batch)
            
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def _embed_with_model(self, batch: np.ndarray) -> np.ndarray:
        """Embed a batch using the sentence transformer model.
        
        Since sentence-transformers expects text, we encode the numeric 
        features as a structured string representation.
        """
        # Convert numeric features to text representation
        # Format: "feature_0:0.123 feature_1:-0.456 ..."
        # This allows the transformer to process numeric data
        
        # For efficiency, we use a simplified approach:
        # Convert each row to a JSON-like string
        texts = []
        for row in batch:
            # Sample key features (not all 384) for efficiency
            # Take every 8th feature to get ~48 values
            sampled = row[::8]
            text = ' '.join([f'{v:.3f}' for v in sampled])
            texts.append(text)
        
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-8)
        return embeddings / norms
    
    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension."""
        self._load_model()
        return self._embedding_dim
    
    def get_model_info(self) -> dict:
        """Get information about the embedding model."""
        self._load_model()
        return {
            'model_name': self.config.model_name,
            'embedding_dim': self._embedding_dim,
            'projection_dim': self.config.projection_dim,
            'normalize': self.config.normalize,
            'batch_size': self.config.batch_size,
        }


# Module exports
__all__ = [
    'EmbeddingConfig',
    'EmbeddingGenerator',
]
