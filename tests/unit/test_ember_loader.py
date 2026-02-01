"""
Unit tests for EMBER dataset loader.

Tests the ember_loader module which provides utilities for:
- Loading EMBER feature vectors
- Validating feature format
- Splitting data into train/test sets
- Accessing feature metadata

These tests are written FIRST (test-first approach), before the loader implementation.
"""

import pytest
import numpy as np
from pathlib import Path


class TestEmberLoaderImport:
    """Test that the ember_loader module can be imported."""
    
    def test_can_import_ember_loader(self):
        """The ember_loader module should be importable."""
        from malvec import ember_loader
        assert ember_loader is not None
    
    def test_ember_loader_has_required_functions(self):
        """The ember_loader should expose required functions."""
        from malvec import ember_loader
        
        # Required functions
        assert hasattr(ember_loader, 'load_ember_features')
        assert hasattr(ember_loader, 'get_feature_names')
        assert hasattr(ember_loader, 'validate_features')
        assert callable(ember_loader.load_ember_features)
        assert callable(ember_loader.get_feature_names)
        assert callable(ember_loader.validate_features)


class TestEmberFeatureFormat:
    """Test EMBER feature format validation."""
    
    def test_ember_feature_vector_dimensions(self):
        """EMBER features should have 2381 dimensions."""
        from malvec.ember_loader import EMBER_FEATURE_DIM
        assert EMBER_FEATURE_DIM == 2381
    
    def test_valid_feature_vector_passes_validation(self):
        """A valid 2381-dim float array should pass validation."""
        from malvec.ember_loader import validate_features
        
        valid_features = np.random.randn(2381).astype(np.float32)
        result = validate_features(valid_features)
        assert result is True
    
    def test_wrong_dimension_fails_validation(self):
        """Wrong dimension feature vectors should fail validation."""
        from malvec.ember_loader import validate_features
        
        wrong_dim_features = np.random.randn(1000).astype(np.float32)
        
        with pytest.raises(ValueError, match="Expected 2381 features"):
            validate_features(wrong_dim_features)
    
    def test_non_numeric_fails_validation(self):
        """Non-numeric data should fail validation."""
        from malvec.ember_loader import validate_features
        
        string_data = ["not", "numeric", "data"]
        
        with pytest.raises(TypeError):
            validate_features(string_data)
    
    def test_nan_values_detected(self):
        """NaN values in features should be detected."""
        from malvec.ember_loader import validate_features
        
        features_with_nan = np.random.randn(2381).astype(np.float32)
        features_with_nan[100] = np.nan
        
        with pytest.raises(ValueError, match="NaN"):
            validate_features(features_with_nan)
    
    def test_inf_values_detected(self):
        """Infinite values in features should be detected."""
        from malvec.ember_loader import validate_features
        
        features_with_inf = np.random.randn(2381).astype(np.float32)
        features_with_inf[100] = np.inf
        
        with pytest.raises(ValueError, match="infinite"):
            validate_features(features_with_inf)


class TestEmberFeatureMetadata:
    """Test feature name and metadata access."""
    
    def test_get_feature_names_returns_list(self):
        """get_feature_names should return a list of strings."""
        from malvec.ember_loader import get_feature_names
        
        names = get_feature_names()
        assert isinstance(names, list)
        assert len(names) == 2381
        assert all(isinstance(name, str) for name in names)
    
    def test_feature_names_are_unique(self):
        """All feature names should be unique."""
        from malvec.ember_loader import get_feature_names
        
        names = get_feature_names()
        assert len(names) == len(set(names)), "Feature names must be unique"
    
    def test_feature_groups_defined(self):
        """EMBER feature groups should be defined."""
        from malvec.ember_loader import FEATURE_GROUPS
        
        expected_groups = [
            'histogram',
            'byteentropy', 
            'strings',
            'general',
            'header',
            'section',
            'imports',
            'exports',
            'datadirectories'
        ]
        
        for group in expected_groups:
            assert group in FEATURE_GROUPS, f"Missing feature group: {group}"


class TestEmberDataLoading:
    """Test loading EMBER data (requires dataset)."""
    
    @pytest.mark.slow
    def test_load_ember_features_returns_correct_shape(self):
        """load_ember_features should return correctly shaped arrays."""
        from malvec.ember_loader import load_ember_features
        
        X, y = load_ember_features(subset='train', max_samples=100)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]  # Same number of samples
        assert X.shape[1] == 2381  # Correct feature dimension
    
    @pytest.mark.slow
    def test_labels_are_valid(self):
        """Labels should be 0 (benign) or 1 (malicious)."""
        from malvec.ember_loader import load_ember_features
        
        _, y = load_ember_features(subset='train', max_samples=100)
        
        unique_labels = set(y)
        assert unique_labels.issubset({0, 1}), f"Invalid labels: {unique_labels}"
    
    @pytest.mark.slow
    def test_can_load_train_subset(self):
        """Should be able to load training subset."""
        from malvec.ember_loader import load_ember_features
        
        X, y = load_ember_features(subset='train', max_samples=10)
        assert X.shape[0] == 10
    
    @pytest.mark.slow  
    def test_can_load_test_subset(self):
        """Should be able to load test subset."""
        from malvec.ember_loader import load_ember_features
        
        X, y = load_ember_features(subset='test', max_samples=10)
        assert X.shape[0] == 10
    
    def test_invalid_subset_raises_error(self):
        """Invalid subset name should raise an error."""
        from malvec.ember_loader import load_ember_features
        
        with pytest.raises(ValueError, match="subset"):
            load_ember_features(subset='invalid')


class TestEmberDataSummary:
    """Test data summary and statistics functions."""
    
    def test_summarize_features_function_exists(self):
        """summarize_features function should exist."""
        from malvec.ember_loader import summarize_features
        assert callable(summarize_features)
    
    @pytest.mark.slow
    def test_summarize_features_returns_dict(self):
        """summarize_features should return a dictionary with stats."""
        from malvec.ember_loader import load_ember_features, summarize_features
        
        X, y = load_ember_features(subset='train', max_samples=100)
        summary = summarize_features(X, y)
        
        assert isinstance(summary, dict)
        assert 'num_samples' in summary
        assert 'num_features' in summary
        assert 'num_malicious' in summary
        assert 'num_benign' in summary
        assert summary['num_features'] == 2381
