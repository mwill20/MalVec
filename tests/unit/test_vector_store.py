"""
Unit tests for Vector Store.

Tests the vector_store module which provides utilities for:
- FAISS-based vector indexing
- Batch insert of embeddings
- K-NN similarity search
- Index persistence (save/load)

These tests validate that we can store and retrieve embeddings efficiently.
"""

import pytest
import numpy as np
import tempfile
import os


class TestVectorStoreImport:
    """Test that the vector_store module can be imported."""
    
    def test_can_import_vector_store(self):
        """The vector_store module should be importable."""
        from malvec import vector_store
        assert vector_store is not None
    
    def test_vector_store_has_required_classes(self):
        """The vector_store should expose required classes."""
        from malvec import vector_store
        
        assert hasattr(vector_store, 'VectorIndex')
        assert hasattr(vector_store, 'VectorIndexConfig')


class TestVectorIndexConfig:
    """Test vector index configuration."""
    
    def test_config_has_default_values(self):
        """VectorIndexConfig should have sensible defaults."""
        from malvec.vector_store import VectorIndexConfig
        
        config = VectorIndexConfig()
        
        assert config.embedding_dim == 384
        assert config.metric == 'cosine'  # or 'l2'
        assert config.use_gpu is False  # CPU by default
    
    def test_config_custom_values(self):
        """Config should accept custom values."""
        from malvec.vector_store import VectorIndexConfig
        
        config = VectorIndexConfig(
            embedding_dim=256,
            metric='l2',
            use_gpu=True
        )
        
        assert config.embedding_dim == 256
        assert config.metric == 'l2'
        assert config.use_gpu is True


class TestVectorIndexCreation:
    """Test vector index creation and basic operations."""
    
    def test_index_initialization(self):
        """VectorIndex should initialize without errors."""
        from malvec.vector_store import VectorIndex, VectorIndexConfig
        
        config = VectorIndexConfig()
        index = VectorIndex(config)
        
        assert index is not None
        assert index.size() == 0
    
    def test_index_default_config(self):
        """VectorIndex should work with default config."""
        from malvec.vector_store import VectorIndex
        
        index = VectorIndex()
        assert index is not None
        assert index.size() == 0
    
    def test_add_single_embedding(self):
        """Should add a single embedding to the index."""
        from malvec.vector_store import VectorIndex, VectorIndexConfig
        
        config = VectorIndexConfig(embedding_dim=384)
        index = VectorIndex(config)
        
        # Single embedding (L2 normalized)
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        index.add(embedding.reshape(1, -1))
        
        assert index.size() == 1
    
    def test_add_batch_embeddings(self):
        """Should add a batch of embeddings to the index."""
        from malvec.vector_store import VectorIndex, VectorIndexConfig
        
        config = VectorIndexConfig(embedding_dim=384)
        index = VectorIndex(config)
        
        # Batch of embeddings
        n_samples = 100
        embeddings = np.random.randn(n_samples, 384).astype(np.float32)
        # Normalize each row
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        index.add(embeddings)
        
        assert index.size() == n_samples


class TestVectorSearch:
    """Test k-NN search functionality."""
    
    @pytest.fixture
    def populated_index(self):
        """Create an index with 1000 embeddings."""
        from malvec.vector_store import VectorIndex, VectorIndexConfig
        
        config = VectorIndexConfig(embedding_dim=384, metric='cosine')
        index = VectorIndex(config)
        
        np.random.seed(42)
        n_samples = 1000
        embeddings = np.random.randn(n_samples, 384).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        index.add(embeddings)
        return index, embeddings
    
    def test_search_returns_k_results(self, populated_index):
        """Search should return exactly k results."""
        index, embeddings = populated_index
        
        # Query with first embedding
        query = embeddings[0:1]
        k = 5
        
        distances, indices = index.search(query, k=k)
        
        assert distances.shape == (1, k)
        assert indices.shape == (1, k)
    
    def test_search_finds_exact_match(self, populated_index):
        """Searching for an exact embedding should return it as top result."""
        index, embeddings = populated_index
        
        # Query with embedding 42
        query = embeddings[42:43]
        
        distances, indices = index.search(query, k=1)
        
        # Top result should be index 42
        assert indices[0, 0] == 42
        # Distance should be ~0 for cosine (or ~1 for inner product)
        assert distances[0, 0] < 0.01 or distances[0, 0] > 0.99
    
    def test_batch_search(self, populated_index):
        """Should handle batch queries."""
        index, embeddings = populated_index
        
        # Query with 10 embeddings
        queries = embeddings[0:10]
        k = 5
        
        distances, indices = index.search(queries, k=k)
        
        assert distances.shape == (10, k)
        assert indices.shape == (10, k)
    
    def test_search_results_are_sorted(self, populated_index):
        """Results should be sorted by distance (ascending or descending)."""
        index, embeddings = populated_index
        
        query = embeddings[0:1]
        k = 10
        
        distances, _ = index.search(query, k=k)
        
        # For cosine similarity (inner product), should be descending
        # For L2 distance, should be ascending
        # Either way, should be monotonic
        diffs = np.diff(distances[0])
        is_sorted = np.all(diffs >= -1e-6) or np.all(diffs <= 1e-6)
        assert is_sorted, "Results should be sorted by distance"


class TestIndexPersistence:
    """Test saving and loading indexes."""
    
    def test_save_and_load(self):
        """Should save and load index correctly."""
        from malvec.vector_store import VectorIndex, VectorIndexConfig
        
        config = VectorIndexConfig(embedding_dim=384)
        index = VectorIndex(config)
        
        # Add some embeddings
        np.random.seed(42)
        embeddings = np.random.randn(100, 384).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        index.add(embeddings)
        
        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_index')
            index.save(path)
            
            # Load into new index
            loaded_index = VectorIndex.load(path)
            
            assert loaded_index.size() == 100
    
    def test_loaded_index_produces_same_results(self):
        """Loaded index should produce identical search results."""
        from malvec.vector_store import VectorIndex, VectorIndexConfig
        
        config = VectorIndexConfig(embedding_dim=384)
        index = VectorIndex(config)
        
        np.random.seed(42)
        embeddings = np.random.randn(100, 384).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        index.add(embeddings)
        
        # Search before save
        query = embeddings[0:1]
        distances_before, indices_before = index.search(query, k=5)
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_index')
            index.save(path)
            loaded_index = VectorIndex.load(path)
            
            # Search after load
            distances_after, indices_after = loaded_index.search(query, k=5)
            
            assert np.allclose(distances_before, distances_after)
            assert np.array_equal(indices_before, indices_after)


class TestIndexInfo:
    """Test index metadata and info."""
    
    def test_get_size(self):
        """Should report correct size."""
        from malvec.vector_store import VectorIndex, VectorIndexConfig
        
        config = VectorIndexConfig(embedding_dim=384)
        index = VectorIndex(config)
        
        assert index.size() == 0
        
        embeddings = np.random.randn(50, 384).astype(np.float32)
        index.add(embeddings)
        
        assert index.size() == 50
    
    def test_get_index_info(self):
        """Should return index information."""
        from malvec.vector_store import VectorIndex, VectorIndexConfig
        
        config = VectorIndexConfig(embedding_dim=256, metric='l2')
        index = VectorIndex(config)
        
        info = index.get_info()
        
        assert isinstance(info, dict)
        assert info['embedding_dim'] == 256
        assert info['metric'] == 'l2'
        assert info['size'] == 0


class TestValidation:
    """Test input validation."""
    
    def test_reject_wrong_embedding_dim(self):
        """Should reject embeddings with wrong dimension."""
        from malvec.vector_store import VectorIndex, VectorIndexConfig
        
        config = VectorIndexConfig(embedding_dim=384)
        index = VectorIndex(config)
        
        # Wrong dimension
        wrong_embeddings = np.random.randn(10, 256).astype(np.float32)
        
        with pytest.raises(ValueError):
            index.add(wrong_embeddings)
    
    def test_reject_non_float32(self):
        """Should reject or convert non-float32 embeddings."""
        from malvec.vector_store import VectorIndex, VectorIndexConfig
        
        config = VectorIndexConfig(embedding_dim=384)
        index = VectorIndex(config)
        
        # Float64 - should either convert or error
        embeddings = np.random.randn(10, 384).astype(np.float64)
        
        # Should either work (auto-convert) or raise error
        try:
            index.add(embeddings)
            # If it worked, size should be updated
            assert index.size() == 10
        except (ValueError, TypeError):
            # Also acceptable - explicit conversion required
            pass
