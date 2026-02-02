"""
Unit tests for K-NN Classifier.

Tests the classifier module which provides:
- K-NN classification using vector similarity
- Majority and weighted voting
- Confidence scoring
- Manual review flagging

These tests validate the classification logic for malware detection.
"""

import pytest
import numpy as np


class TestClassifierImport:
    """Test that the classifier module can be imported."""
    
    def test_can_import_classifier(self):
        """The classifier module should be importable."""
        from malvec import classifier
        assert classifier is not None
    
    def test_classifier_has_required_classes(self):
        """The classifier should expose required classes."""
        from malvec import classifier
        
        assert hasattr(classifier, 'KNNClassifier')
        assert hasattr(classifier, 'ClassifierConfig')


class TestClassifierConfig:
    """Test classifier configuration."""
    
    def test_config_has_default_values(self):
        """ClassifierConfig should have sensible defaults."""
        from malvec.classifier import ClassifierConfig
        
        config = ClassifierConfig()
        
        assert config.k == 5
        assert config.vote_method == 'majority'
        assert 0 < config.confidence_threshold < 1
    
    def test_config_custom_values(self):
        """Config should accept custom values."""
        from malvec.classifier import ClassifierConfig
        
        config = ClassifierConfig(
            k=10,
            vote_method='weighted',
            confidence_threshold=0.7
        )
        
        assert config.k == 10
        assert config.vote_method == 'weighted'
        assert config.confidence_threshold == 0.7


class TestClassifierCreation:
    """Test classifier initialization."""
    
    def test_classifier_initialization(self):
        """KNNClassifier should initialize without errors."""
        from malvec.classifier import KNNClassifier
        
        clf = KNNClassifier()
        assert clf is not None
    
    def test_classifier_with_config(self):
        """KNNClassifier should accept config."""
        from malvec.classifier import KNNClassifier, ClassifierConfig
        
        config = ClassifierConfig(k=3)
        clf = KNNClassifier(config)
        assert clf.config.k == 3


class TestClassifierFit:
    """Test classifier training (fit)."""
    
    def test_fit_stores_labels(self):
        """Fit should store labels for prediction."""
        from malvec.classifier import KNNClassifier
        
        clf = KNNClassifier()
        
        # Create synthetic normalized embeddings
        np.random.seed(42)
        embeddings = np.random.randn(100, 384).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        labels = np.random.randint(0, 2, 100)
        
        clf.fit(embeddings, labels)
        
        # Should have stored labels
        assert len(clf._labels) == 100
    
    def test_fit_builds_index(self):
        """Fit should build vector index."""
        from malvec.classifier import KNNClassifier
        
        clf = KNNClassifier()
        
        np.random.seed(42)
        embeddings = np.random.randn(100, 384).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        labels = np.random.randint(0, 2, 100)
        
        clf.fit(embeddings, labels)
        
        # Should have index
        assert clf._index is not None
        assert clf._index.size() == 100


class TestClassifierPredict:
    """Test prediction functionality."""
    
    @pytest.fixture
    def fitted_classifier(self):
        """Create a fitted classifier with known data."""
        from malvec.classifier import KNNClassifier, ClassifierConfig
        
        config = ClassifierConfig(k=5)
        clf = KNNClassifier(config)
        
        np.random.seed(42)
        n_samples = 100
        embeddings = np.random.randn(n_samples, 384).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Labels: first 50 = 0 (benign), last 50 = 1 (malware)
        labels = np.array([0] * 50 + [1] * 50)
        
        clf.fit(embeddings, labels)
        return clf, embeddings, labels
    
    def test_predict_returns_labels(self, fitted_classifier):
        """Predict should return class labels."""
        clf, embeddings, _ = fitted_classifier
        
        query = embeddings[0:1]
        predictions = clf.predict(query)
        
        assert predictions.shape == (1,)
        assert predictions[0] in [0, 1]
    
    def test_predict_batch(self, fitted_classifier):
        """Should handle batch predictions."""
        clf, embeddings, _ = fitted_classifier
        
        queries = embeddings[0:10]
        predictions = clf.predict(queries)
        
        assert predictions.shape == (10,)
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_known_sample(self, fitted_classifier):
        """Predicting a training sample should return its label."""
        clf, embeddings, labels = fitted_classifier
        
        # Query with sample 0 (label = 0)
        query = embeddings[0:1]
        prediction = clf.predict(query)
        
        # Self is nearest neighbor, so prediction should match label
        assert prediction[0] == labels[0]
    
    def test_predict_with_confidence(self, fitted_classifier):
        """Should return confidence when requested."""
        clf, embeddings, _ = fitted_classifier
        
        query = embeddings[0:1]
        predictions, confidences = clf.predict(query, return_confidence=True)
        
        assert predictions.shape == (1,)
        assert confidences.shape == (1,)
        assert 0 <= confidences[0] <= 1


class TestVoting:
    """Test voting mechanisms."""
    
    @pytest.fixture
    def classifier_with_clear_clusters(self):
        """Create classifier with separable clusters."""
        from malvec.classifier import KNNClassifier, ClassifierConfig
        
        config = ClassifierConfig(k=5)
        clf = KNNClassifier(config)
        
        # Create two well-separated clusters
        np.random.seed(42)
        
        # Cluster 0: centered at [1, 0, 0, ...]
        cluster0 = np.zeros((50, 384), dtype=np.float32)
        cluster0[:, 0] = 1
        cluster0 += np.random.randn(50, 384).astype(np.float32) * 0.1
        
        # Cluster 1: centered at [0, 1, 0, ...]
        cluster1 = np.zeros((50, 384), dtype=np.float32)
        cluster1[:, 1] = 1
        cluster1 += np.random.randn(50, 384).astype(np.float32) * 0.1
        
        embeddings = np.vstack([cluster0, cluster1])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        labels = np.array([0] * 50 + [1] * 50)
        
        clf.fit(embeddings, labels)
        return clf, embeddings, labels
    
    def test_majority_voting(self, classifier_with_clear_clusters):
        """Majority voting should work for clear clusters."""
        clf, embeddings, labels = classifier_with_clear_clusters
        
        # Query from cluster 0
        query = embeddings[0:1]
        prediction = clf.predict(query)
        assert prediction[0] == 0
        
        # Query from cluster 1
        query = embeddings[50:51]
        prediction = clf.predict(query)
        assert prediction[0] == 1
    
    def test_weighted_voting(self, classifier_with_clear_clusters):
        """Weighted voting should weight by similarity."""
        from malvec.classifier import KNNClassifier, ClassifierConfig
        
        clf, embeddings, labels = classifier_with_clear_clusters
        
        # Create classifier with weighted voting
        config = ClassifierConfig(k=5, vote_method='weighted')
        weighted_clf = KNNClassifier(config)
        weighted_clf.fit(embeddings, labels)
        
        # Should still classify correctly
        query = embeddings[0:1]
        prediction = weighted_clf.predict(query)
        assert prediction[0] == 0


class TestConfidence:
    """Test confidence scoring."""
    
    def test_high_confidence_for_clear_case(self):
        """Should have high confidence when all neighbors agree."""
        from malvec.classifier import KNNClassifier, ClassifierConfig
        
        config = ClassifierConfig(k=5)
        clf = KNNClassifier(config)
        
        # All samples have same label
        np.random.seed(42)
        embeddings = np.random.randn(100, 384).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        labels = np.ones(100, dtype=int)  # All malware
        
        clf.fit(embeddings, labels)
        
        query = embeddings[0:1]
        _, confidences = clf.predict(query, return_confidence=True)
        
        # All neighbors are malware, so confidence should be 1.0
        assert confidences[0] == 1.0
    
    def test_low_confidence_for_mixed_case(self):
        """Should have lower confidence when neighbors disagree."""
        from malvec.classifier import KNNClassifier, ClassifierConfig
        
        config = ClassifierConfig(k=5)
        clf = KNNClassifier(config)
        
        # Mix of labels
        np.random.seed(42)
        embeddings = np.random.randn(100, 384).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        labels = np.array([0, 1] * 50)  # Alternating
        
        clf.fit(embeddings, labels)
        
        query = embeddings[0:1]
        _, confidences = clf.predict(query, return_confidence=True)
        
        # Neighbors will be mixed, so confidence should be < 1.0
        assert 0.5 <= confidences[0] <= 1.0


class TestManualReview:
    """Test manual review flagging."""
    
    def test_needs_review_below_threshold(self):
        """Should flag for review when confidence is below threshold."""
        from malvec.classifier import KNNClassifier, ClassifierConfig
        
        # Low threshold to ensure most pass
        config = ClassifierConfig(k=5, confidence_threshold=0.9)
        clf = KNNClassifier(config)
        
        # Mix of labels will cause low confidence
        np.random.seed(42)
        embeddings = np.random.randn(100, 384).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        labels = np.array([0, 1] * 50)  # Alternating
        
        clf.fit(embeddings, labels)
        
        query = embeddings[0:1]
        result = clf.predict_with_review(query)
        
        # Should have review flag
        assert 'needs_review' in result
        assert isinstance(result['needs_review'], np.ndarray)
        assert result['needs_review'].dtype == bool
        # With threshold 0.9 and mixed labels, should need review
        assert result['needs_review'][0] == True
    
    def test_predict_with_review_returns_all_info(self):
        """predict_with_review should return prediction, confidence, and review flag."""
        from malvec.classifier import KNNClassifier
        
        clf = KNNClassifier()
        
        np.random.seed(42)
        embeddings = np.random.randn(100, 384).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        labels = np.ones(100, dtype=int)
        
        clf.fit(embeddings, labels)
        
        query = embeddings[0:1]
        result = clf.predict_with_review(query)
        
        assert 'predictions' in result
        assert 'confidences' in result
        assert 'needs_review' in result


class TestClassifierInfo:
    """Test classifier metadata."""
    
    def test_get_info(self):
        """Should return classifier info."""
        from malvec.classifier import KNNClassifier, ClassifierConfig
        
        config = ClassifierConfig(k=7, vote_method='weighted')
        clf = KNNClassifier(config)
        
        info = clf.get_info()
        
        assert isinstance(info, dict)
        assert info['k'] == 7
        assert info['vote_method'] == 'weighted'
