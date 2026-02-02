"""
Unit tests for Embedding Generator.

Tests the embedder module which provides utilities for:
- Converting EMBER feature vectors to embeddings via random projection
- Johnson-Lindenstrauss distance preservation
- L2 normalization for cosine similarity

These tests validate that random projection produces correct embeddings.
"""

import pytest
import numpy as np
from scipy.spatial.distance import pdist


class TestEmbedderImport:
    """Test that the embedder module can be imported."""
    
    def test_can_import_embedder(self):
        """The embedder module should be importable."""
        from malvec import embedder
        assert embedder is not None
    
    def test_embedder_has_required_classes(self):
        """The embedder should expose required classes."""
        from malvec import embedder
        
        assert hasattr(embedder, 'EmbeddingGenerator')
        assert hasattr(embedder, 'EmbeddingConfig')


class TestEmbeddingConfig:
    """Test embedding configuration."""
    
    def test_config_has_default_values(self):
        """EmbeddingConfig should have sensible defaults."""
        from malvec.embedder import EmbeddingConfig
        
        config = EmbeddingConfig()
        
        assert config.embedding_dim == 384
        assert config.random_state == 42
        assert config.normalize is True
    
    def test_config_custom_values(self):
        """Config should accept custom values."""
        from malvec.embedder import EmbeddingConfig
        
        config = EmbeddingConfig(
            embedding_dim=256,
            random_state=123,
            normalize=False
        )
        
        assert config.embedding_dim == 256
        assert config.random_state == 123
        assert config.normalize is False


class TestEmbeddingGenerator:
    """Test the EmbeddingGenerator class."""
    
    def test_generator_initialization(self):
        """EmbeddingGenerator should initialize without errors."""
        from malvec.embedder import EmbeddingGenerator, EmbeddingConfig
        
        config = EmbeddingConfig()
        generator = EmbeddingGenerator(config)
        
        assert generator is not None
        assert generator.config == config
    
    def test_generator_default_config(self):
        """EmbeddingGenerator should work with default config."""
        from malvec.embedder import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        assert generator.config is not None
        assert generator.config.embedding_dim == 384
    
    def test_generate_embeddings_shape(self):
        """Generated embeddings should have correct shape."""
        from malvec.embedder import EmbeddingGenerator
        from malvec.ember_loader import EMBER_FEATURE_DIM
        
        generator = EmbeddingGenerator()
        
        n_samples = 10
        features = np.random.randn(n_samples, EMBER_FEATURE_DIM).astype(np.float32)
        
        embeddings = generator.generate(features)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (n_samples, 384)
    
    def test_generate_single_sample(self):
        """Should handle single sample input."""
        from malvec.embedder import EmbeddingGenerator
        from malvec.ember_loader import EMBER_FEATURE_DIM
        
        generator = EmbeddingGenerator()
        
        # Single sample (1D array)
        features = np.random.randn(EMBER_FEATURE_DIM).astype(np.float32)
        
        embeddings = generator.generate(features)
        
        assert embeddings.shape == (1, 384)
    
    def test_embeddings_are_normalized(self):
        """Embeddings should be L2 normalized when configured."""
        from malvec.embedder import EmbeddingGenerator, EmbeddingConfig
        from malvec.ember_loader import EMBER_FEATURE_DIM
        
        config = EmbeddingConfig(normalize=True)
        generator = EmbeddingGenerator(config)
        
        features = np.random.randn(5, EMBER_FEATURE_DIM).astype(np.float32)
        embeddings = generator.generate(features)
        
        # Check L2 norms are approximately 1
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
    
    def test_embeddings_not_normalized_when_disabled(self):
        """Embeddings should NOT be normalized when normalize=False."""
        from malvec.embedder import EmbeddingGenerator, EmbeddingConfig
        from malvec.ember_loader import EMBER_FEATURE_DIM
        
        config = EmbeddingConfig(normalize=False)
        generator = EmbeddingGenerator(config)
        
        features = np.random.randn(5, EMBER_FEATURE_DIM).astype(np.float32)
        embeddings = generator.generate(features)
        
        # Norms should NOT all be 1.0
        norms = np.linalg.norm(embeddings, axis=1)
        assert not np.allclose(norms, 1.0, atol=1e-3)
    
    def test_embeddings_dtype(self):
        """Embeddings should be float32."""
        from malvec.embedder import EmbeddingGenerator
        from malvec.ember_loader import EMBER_FEATURE_DIM
        
        generator = EmbeddingGenerator()
        features = np.random.randn(5, EMBER_FEATURE_DIM).astype(np.float32)
        
        embeddings = generator.generate(features)
        
        assert embeddings.dtype == np.float32


class TestEmbeddingValidation:
    """Test embedding validation."""
    
    def test_reject_wrong_input_dimension(self):
        """Should reject features with wrong input dimension."""
        from malvec.embedder import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        
        # Wrong dimension (not 2381)
        wrong_features = np.random.randn(10, 1000).astype(np.float32)
        
        with pytest.raises(ValueError):
            generator.generate(wrong_features)
    
    def test_reject_nan_input(self):
        """Should reject features containing NaN."""
        from malvec.embedder import EmbeddingGenerator
        from malvec.ember_loader import EMBER_FEATURE_DIM
        
        generator = EmbeddingGenerator()
        
        features = np.random.randn(5, EMBER_FEATURE_DIM).astype(np.float32)
        features[2, 100] = np.nan
        
        with pytest.raises(ValueError):
            generator.generate(features)
    
    def test_reject_inf_input(self):
        """Should reject features containing infinity."""
        from malvec.embedder import EmbeddingGenerator
        from malvec.ember_loader import EMBER_FEATURE_DIM
        
        generator = EmbeddingGenerator()
        
        features = np.random.randn(5, EMBER_FEATURE_DIM).astype(np.float32)
        features[2, 100] = np.inf
        
        with pytest.raises(ValueError):
            generator.generate(features)


class TestJohnsonLindenstrauss:
    """Test that random projection preserves distances (JL lemma)."""
    
    def test_distance_preservation(self):
        """Relative distance ordering should be approximately preserved.
        
        For k-NN search, what matters is that if A is closer to B than to C
        in the original space, this relationship is preserved in the embedded space.
        We use Spearman rank correlation to measure this.
        """
        from malvec.embedder import EmbeddingGenerator, EmbeddingConfig
        from malvec.ember_loader import EMBER_FEATURE_DIM
        from scipy.stats import spearmanr
        
        # Use unnormalized embeddings for distance preservation test
        config = EmbeddingConfig(normalize=False)
        generator = EmbeddingGenerator(config)
        
        # Generate random features
        np.random.seed(42)
        n_samples = 30  # Fewer samples for cleaner test
        features = np.random.randn(n_samples, EMBER_FEATURE_DIM).astype(np.float32)
        
        # Compute original pairwise distances
        orig_distances = pdist(features, 'euclidean')
        
        # Generate embeddings
        embeddings = generator.generate(features)
        
        # Compute embedded pairwise distances
        emb_distances = pdist(embeddings, 'euclidean')
        
        # Spearman rank correlation - measures if ordering is preserved
        # This is what matters for k-NN search
        rank_corr, _ = spearmanr(orig_distances, emb_distances)
        
        # Expect rank correlation > 0.5 for reasonable ordering preservation
        # (random would be ~0, perfect would be 1)
        assert rank_corr > 0.5, f"Rank correlation {rank_corr:.3f} too low"
    
    def test_reproducibility(self):
        """Same random_state should produce same projection."""
        from malvec.embedder import EmbeddingGenerator, EmbeddingConfig
        from malvec.ember_loader import EMBER_FEATURE_DIM
        
        config = EmbeddingConfig(random_state=42)
        
        features = np.random.randn(10, EMBER_FEATURE_DIM).astype(np.float32)
        
        gen1 = EmbeddingGenerator(config)
        emb1 = gen1.generate(features)
        
        gen2 = EmbeddingGenerator(config)
        emb2 = gen2.generate(features)
        
        # Same seed should produce identical results
        assert np.allclose(emb1, emb2)
    
    def test_different_seeds_different_results(self):
        """Different random_state should produce different projections."""
        from malvec.embedder import EmbeddingGenerator, EmbeddingConfig
        from malvec.ember_loader import EMBER_FEATURE_DIM
        
        features = np.random.randn(10, EMBER_FEATURE_DIM).astype(np.float32)
        
        gen1 = EmbeddingGenerator(EmbeddingConfig(random_state=42))
        emb1 = gen1.generate(features)
        
        gen2 = EmbeddingGenerator(EmbeddingConfig(random_state=123))
        emb2 = gen2.generate(features)
        
        # Different seeds should produce different results
        assert not np.allclose(emb1, emb2)


class TestEmbeddingInfo:
    """Test embedding metadata and info."""
    
    def test_get_embedding_dim(self):
        """Should report embedding dimension."""
        from malvec.embedder import EmbeddingGenerator, EmbeddingConfig
        
        config = EmbeddingConfig(embedding_dim=256)
        generator = EmbeddingGenerator(config)
        
        assert generator.get_embedding_dim() == 256
    
    def test_get_model_info(self):
        """Should return model information."""
        from malvec.embedder import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        
        info = generator.get_model_info()
        
        assert isinstance(info, dict)
        assert info['model_name'] == 'random_projection'
        assert info['model_type'] == 'Johnson-Lindenstrauss'
        assert info['embedding_dim'] == 384
        assert info['normalize'] is True


class TestFeatureProjection:
    """Test feature projection."""
    
    def test_projection_layer_exists(self):
        """EmbeddingGenerator should have a projection method."""
        from malvec.embedder import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        
        assert hasattr(generator, 'project_features')
        assert callable(generator.project_features)
    
    def test_projection_reduces_dimension(self):
        """Projection should reduce 2381 dims to embedding_dim."""
        from malvec.embedder import EmbeddingGenerator, EmbeddingConfig
        from malvec.ember_loader import EMBER_FEATURE_DIM
        
        config = EmbeddingConfig(embedding_dim=256)
        generator = EmbeddingGenerator(config)
        
        features = np.random.randn(10, EMBER_FEATURE_DIM).astype(np.float32)
        
        projected = generator.project_features(features)
        
        # Should reduce from 2381 to 256
        assert projected.shape == (10, 256)
    
    def test_projection_uses_all_features(self):
        """Projection should use all input features."""
        from malvec.embedder import EmbeddingGenerator
        from malvec.ember_loader import EMBER_FEATURE_DIM
        
        generator = EmbeddingGenerator()
        
        # Create features where only one column is non-zero
        features1 = np.zeros((1, EMBER_FEATURE_DIM), dtype=np.float32)
        features1[0, 0] = 1.0
        
        features2 = np.zeros((1, EMBER_FEATURE_DIM), dtype=np.float32)
        features2[0, 1000] = 1.0
        
        emb1 = generator.generate(features1)
        emb2 = generator.generate(features2)
        
        # Different input columns should produce different embeddings
        # (proving all columns are used in projection)
        assert not np.allclose(emb1, emb2)
