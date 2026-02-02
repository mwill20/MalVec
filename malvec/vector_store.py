"""
Vector Store for MalVec.

Provides FAISS-based vector indexing for efficient similarity search
on malware embeddings.

Features:
- Fast approximate nearest neighbor search
- Cosine similarity (via inner product on normalized vectors)
- L2 distance support
- Index persistence (save/load)
- Batch insert and query

Why FAISS:
- Industry standard for vector search
- Highly optimized C++ with Python bindings
- Scales to billions of vectors
- GPU support available
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import faiss


@dataclass
class VectorIndexConfig:
    """Configuration for the VectorIndex.
    
    Attributes:
        embedding_dim: Dimension of embedding vectors
        metric: Distance metric - 'cosine' or 'l2'
        use_gpu: Whether to use GPU (if available)
    """
    embedding_dim: int = 384
    metric: str = 'cosine'  # 'cosine' or 'l2'
    use_gpu: bool = False


class VectorIndex:
    """
    FAISS-based vector index for similarity search.
    
    Stores embeddings and enables fast k-NN search.
    
    Example:
        >>> index = VectorIndex()
        >>> embeddings = np.random.randn(1000, 384).astype(np.float32)
        >>> embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        >>> index.add(embeddings)
        >>> distances, indices = index.search(query, k=5)
    """
    
    def __init__(self, config: Optional[VectorIndexConfig] = None):
        """
        Initialize the VectorIndex.
        
        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or VectorIndexConfig()
        self._index = self._create_index()
        self._count = 0
    
    def _create_index(self) -> faiss.Index:
        """
        Create the appropriate FAISS index based on config.
        
        For cosine similarity: Use IndexFlatIP (inner product)
            - Requires L2-normalized vectors
            - Higher score = more similar
        
        For L2 distance: Use IndexFlatL2
            - Lower score = more similar
        """
        dim = self.config.embedding_dim
        
        if self.config.metric == 'cosine':
            # Inner product on normalized vectors = cosine similarity
            index = faiss.IndexFlatIP(dim)
        else:
            # L2 (Euclidean) distance
            index = faiss.IndexFlatL2(dim)
        
        # GPU support (if requested and available)
        if self.config.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception:
                # Fall back to CPU if GPU not available
                pass
        
        return index
    
    def add(self, embeddings: np.ndarray) -> None:
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Array of shape (n, embedding_dim)
            
        Raises:
            ValueError: If embeddings have wrong dimension
        """
        # Handle single embedding
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Validate dimension
        if embeddings.shape[1] != self.config.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.config.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        
        # Convert to float32 if needed (FAISS requirement)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Ensure contiguous array (FAISS requirement)
        if not embeddings.flags['C_CONTIGUOUS']:
            embeddings = np.ascontiguousarray(embeddings)
        
        self._index.add(embeddings)
        self._count += len(embeddings)
    
    def search(
        self, 
        query: np.ndarray, 
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query vector(s) of shape (n_queries, embedding_dim) or (embedding_dim,)
            k: Number of neighbors to return
            
        Returns:
            Tuple of (distances, indices) each of shape (n_queries, k)
            - For cosine: distances are similarities (higher = more similar)
            - For L2: distances are squared L2 distances (lower = more similar)
        """
        # Handle single query
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Convert to float32 if needed
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        
        # Ensure contiguous
        if not query.flags['C_CONTIGUOUS']:
            query = np.ascontiguousarray(query)
        
        # Search
        distances, indices = self._index.search(query, k)
        
        return distances, indices
    
    def size(self) -> int:
        """Return the number of vectors in the index."""
        return self._count
    
    def get_info(self) -> dict:
        """Return information about the index."""
        return {
            'embedding_dim': self.config.embedding_dim,
            'metric': self.config.metric,
            'size': self._count,
            'use_gpu': self.config.use_gpu,
        }
    
    def save(self, path: str) -> None:
        """
        Save the index to disk.
        
        Args:
            path: Path to save the index (will create .index and .config files)
        """
        import json
        
        # Save FAISS index
        index_path = f"{path}.index"
        
        # If GPU index, convert to CPU for saving
        if self.config.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self._index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self._index, index_path)
        
        # Save config and metadata
        config_path = f"{path}.config"
        metadata = {
            'embedding_dim': self.config.embedding_dim,
            'metric': self.config.metric,
            'use_gpu': self.config.use_gpu,
            'count': self._count,
        }
        with open(config_path, 'w') as f:
            json.dump(metadata, f)
    
    @classmethod
    def load(cls, path: str) -> 'VectorIndex':
        """
        Load an index from disk.
        
        Args:
            path: Path prefix used when saving
            
        Returns:
            Loaded VectorIndex
        """
        import json
        
        # Load config
        config_path = f"{path}.config"
        with open(config_path, 'r') as f:
            metadata = json.load(f)
        
        config = VectorIndexConfig(
            embedding_dim=metadata['embedding_dim'],
            metric=metadata['metric'],
            use_gpu=metadata['use_gpu'],
        )
        
        # Create index object
        index = cls(config)
        
        # Load FAISS index
        index_path = f"{path}.index"
        index._index = faiss.read_index(index_path)
        index._count = metadata['count']
        
        # Move to GPU if configured
        if config.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index._index = faiss.index_cpu_to_gpu(res, 0, index._index)
            except Exception:
                pass
        
        return index


# Module exports
__all__ = [
    'VectorIndexConfig',
    'VectorIndex',
]
