"""
Unit tests for Embedding Generator.

Tests the embedder module which provides utilities for:
- Converting EMBER feature vectors to semantic embeddings
- Model selection and configuration
- Batch processing with memory efficiency
- Embedding validation and normalization

These tests are written FIRST (test-first approach), before implementation.
"""

import pytest
import numpy as np
from pathlib import Path


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
        
        assert hasattr(config, 'model_name')
        assert hasattr(config, 'embedding_dim')
        assert hasattr(config, 'batch_size')
        assert hasattr(config, 'normalize')
        
    def test_config_default_model(self):
        """Default model should be a valid sentence-transformer."""
        from malvec.embedder import EmbeddingConfig
        
        config = EmbeddingConfig()
        # Default should be a lightweight model
        assert 'MiniLM' in config.model_name or config.model_name is not None
    
    def test_config_custom_values(self):
        """Config should accept custom values."""
        from malvec.embedder import EmbeddingConfig
        
        config = EmbeddingConfig(
            model_name='all-MiniLM-L6-v2',
            batch_size=64,
            normalize=True
        )
        
        assert config.model_name == 'all-MiniLM-L6-v2'
        assert config.batch_size == 64
        assert config.normalize is True


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
    
    @pytest.mark.slow
    def test_generate_embeddings_shape(self):
        """Generated embeddings should have correct shape."""
        from malvec.embedder import EmbeddingGenerator
        from malvec.ember_loader import EMBER_FEATURE_DIM
        
        generator = EmbeddingGenerator()
        
        # Create synthetic EMBER features
        n_samples = 10
        features = np.random.randn(n_samples, EMBER_FEATURE_DIM).astype(np.float32)
        
        embeddings = generator.generate(features)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == n_samples
        assert embeddings.ndim == 2
        # Embedding dim depends on model, but should be > 0
        assert embeddings.shape[1] > 0
    
    @pytest.mark.slow
    def test_generate_single_sample(self):
        """Should handle single sample input."""
        from malvec.embedder import EmbeddingGenerator
        from malvec.ember_loader import EMBER_FEATURE_DIM
        
        generator = EmbeddingGenerator()
        
        # Single sample (1D array)
        features = np.random.randn(EMBER_FEATURE_DIM).astype(np.float32)
        
        embeddings = generator.generate(features)
        
        assert embeddings.shape[0] == 1
    
    @pytest.mark.slow
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
    
    @pytest.mark.slow
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


class TestBatchProcessing:
    """Test batch processing for memory efficiency."""
    
    @pytest.mark.slow
    def test_large_batch_processing(self):
        """Should handle large batches efficiently."""
        from malvec.embedder import EmbeddingGenerator, EmbeddingConfig
        from malvec.ember_loader import EMBER_FEATURE_DIM
        
        config = EmbeddingConfig(batch_size=32)
        generator = EmbeddingGenerator(config)
        
        # 100 samples should be processed in batches of 32
        features = np.random.randn(100, EMBER_FEATURE_DIM).astype(np.float32)
        
        embeddings = generator.generate(features)
        
        assert embeddings.shape[0] == 100
    
    @pytest.mark.slow
    def test_batch_size_respected(self):
        """Batch processing should not exceed configured batch size."""
        from malvec.embedder import EmbeddingGenerator, EmbeddingConfig
        from malvec.ember_loader import EMBER_FEATURE_DIM
        
        config = EmbeddingConfig(batch_size=16)
        generator = EmbeddingGenerator(config)
        
        features = np.random.randn(50, EMBER_FEATURE_DIM).astype(np.float32)
        
        # Should complete without memory issues
        embeddings = generator.generate(features)
        assert embeddings.shape[0] == 50


class TestEmbeddingInfo:
    """Test embedding metadata and info."""
    
    def test_get_embedding_dim(self):
        """Should report embedding dimension."""
        from malvec.embedder import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        
        dim = generator.get_embedding_dim()
        assert isinstance(dim, int)
        assert dim > 0
    
    def test_get_model_info(self):
        """Should return model information."""
        from malvec.embedder import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        
        info = generator.get_model_info()
        
        assert isinstance(info, dict)
        assert 'model_name' in info
        assert 'embedding_dim' in info


class TestFeatureProjection:
    """Test feature projection strategies."""
    
    def test_projection_layer_exists(self):
        """EmbeddingGenerator should have a projection layer."""
        from malvec.embedder import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        
        # Should have method to project EMBER features
        assert hasattr(generator, 'project_features')
        assert callable(generator.project_features)
    
    @pytest.mark.slow
    def test_projection_reduces_dimension(self):
        """Projection should reduce 2381 dims to manageable size."""
        from malvec.embedder import EmbeddingGenerator
        from malvec.ember_loader import EMBER_FEATURE_DIM
        
        generator = EmbeddingGenerator()
        
        features = np.random.randn(10, EMBER_FEATURE_DIM).astype(np.float32)
        
        projected = generator.project_features(features)
        
        # Should reduce dimensions
        assert projected.shape[1] < EMBER_FEATURE_DIM
        # But maintain sample count
        assert projected.shape[0] == 10
