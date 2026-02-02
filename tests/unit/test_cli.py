"""
Unit tests for CLI scripts.

Tests the command-line interface for MalVec:
- train.py: Build model from EMBER data
- classify.py: Classify samples
- info.py: Show model information

These tests validate the CLI layer works correctly.
"""

import pytest
import subprocess
import sys
import os
import tempfile
import json


class TestCLIHelp:
    """Test that CLI scripts have proper help messages."""
    
    def test_train_help(self):
        """train.py should have help documentation."""
        result = subprocess.run(
            [sys.executable, '-m', 'malvec.cli.train', '--help'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        assert result.returncode == 0
        assert 'usage' in result.stdout.lower() or 'train' in result.stdout.lower()
    
    def test_classify_help(self):
        """classify.py should have help documentation."""
        result = subprocess.run(
            [sys.executable, '-m', 'malvec.cli.classify', '--help'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        assert result.returncode == 0
        assert 'usage' in result.stdout.lower() or 'classify' in result.stdout.lower()
    
    def test_info_help(self):
        """info.py should have help documentation."""
        result = subprocess.run(
            [sys.executable, '-m', 'malvec.cli.info', '--help'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        assert result.returncode == 0
        assert 'usage' in result.stdout.lower() or 'info' in result.stdout.lower()
    
    def test_evaluate_help(self):
        """evaluate.py should have help documentation."""
        result = subprocess.run(
            [sys.executable, '-m', 'malvec.cli.evaluate', '--help'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        assert result.returncode == 0
        assert 'usage' in result.stdout.lower() or 'evaluate' in result.stdout.lower()
    
    def test_batch_help(self):
        """batch.py should have help documentation."""
        result = subprocess.run(
            [sys.executable, '-m', 'malvec.cli.batch', '--help'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        assert result.returncode == 0
        assert 'usage' in result.stdout.lower() or 'batch' in result.stdout.lower()


class TestTrainCLI:
    """Test the training CLI."""
    
    def test_train_creates_model(self):
        """Training should create model files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable, '-m', 'malvec.cli.train',
                    '--output', tmpdir,
                    '--max-samples', '100',
                ],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
            
            # Should succeed
            assert result.returncode == 0, f"STDERR: {result.stderr}"
            
            # Should create model files
            assert os.path.exists(os.path.join(tmpdir, 'model.index'))
            assert os.path.exists(os.path.join(tmpdir, 'model.config'))
            assert os.path.exists(os.path.join(tmpdir, 'model.labels.npy'))
    
    def test_train_with_k_parameter(self):
        """Training should accept k parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable, '-m', 'malvec.cli.train',
                    '--output', tmpdir,
                    '--max-samples', '50',
                    '--k', '7',
                ],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
            
            assert result.returncode == 0, f"STDERR: {result.stderr}"


class TestClassifyCLI:
    """Test the classification CLI."""
    
    @pytest.fixture
    def trained_model(self):
        """Create a trained model for testing."""
        tmpdir = tempfile.mkdtemp()
        
        # Train a small model
        result = subprocess.run(
            [
                sys.executable, '-m', 'malvec.cli.train',
                '--output', tmpdir,
                '--max-samples', '100',
            ],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        
        assert result.returncode == 0, f"Failed to create test model: {result.stderr}"
        
        yield tmpdir
        
        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    def test_classify_sample(self, trained_model):
        """Should classify a sample from the dataset."""
        result = subprocess.run(
            [
                sys.executable, '-m', 'malvec.cli.classify',
                '--model', trained_model,
                '--sample-index', '0',
            ],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        
        assert result.returncode == 0, f"STDERR: {result.stderr}"
        # Should output prediction
        output = result.stdout.lower()
        assert 'malware' in output or 'benign' in output or 'prediction' in output
    
    def test_classify_with_confidence(self, trained_model):
        """Should show confidence when requested."""
        result = subprocess.run(
            [
                sys.executable, '-m', 'malvec.cli.classify',
                '--model', trained_model,
                '--sample-index', '0',
                '--show-confidence',
            ],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        
        assert result.returncode == 0, f"STDERR: {result.stderr}"
        # Should show confidence value
        assert 'confidence' in result.stdout.lower() or '%' in result.stdout


class TestInfoCLI:
    """Test the model info CLI."""
    
    @pytest.fixture
    def trained_model(self):
        """Create a trained model for testing."""
        tmpdir = tempfile.mkdtemp()
        
        result = subprocess.run(
            [
                sys.executable, '-m', 'malvec.cli.train',
                '--output', tmpdir,
                '--max-samples', '100',
            ],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        
        assert result.returncode == 0, f"Failed to create test model: {result.stderr}"
        
        yield tmpdir
        
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    def test_info_shows_model_stats(self, trained_model):
        """Should show model statistics."""
        result = subprocess.run(
            [
                sys.executable, '-m', 'malvec.cli.info',
                '--model', trained_model,
            ],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        
        assert result.returncode == 0, f"STDERR: {result.stderr}"
        # Should show sample count
        output = result.stdout.lower()
        assert 'samples' in output or '100' in output or 'index' in output


class TestCLIErrors:
    """Test CLI error handling."""
    
    def test_classify_missing_model(self):
        """Should error gracefully with missing model."""
        result = subprocess.run(
            [
                sys.executable, '-m', 'malvec.cli.classify',
                '--model', '/nonexistent/path',
                '--sample-index', '0',
            ],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        
        # Should fail gracefully
        assert result.returncode != 0
        assert 'error' in result.stderr.lower() or 'not found' in result.stderr.lower() or 'no such' in result.stderr.lower()
    
    def test_info_missing_model(self):
        """Should error gracefully with missing model."""
        result = subprocess.run(
            [
                sys.executable, '-m', 'malvec.cli.info',
                '--model', '/nonexistent/path',
            ],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        
        assert result.returncode != 0
    
    def test_classify_sample_out_of_range(self):
        """Should error when sample index exceeds model size."""
        import shutil
        tmpdir = tempfile.mkdtemp()
        
        try:
            # Train a small model (100 samples)
            result = subprocess.run(
                [
                    sys.executable, '-m', 'malvec.cli.train',
                    '--output', tmpdir,
                    '--max-samples', '100',
                ],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
            assert result.returncode == 0
            
            # Try to classify sample 9999 (out of range)
            result = subprocess.run(
                [
                    sys.executable, '-m', 'malvec.cli.classify',
                    '--model', tmpdir,
                    '--sample-index', '9999',
                ],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
            
            # Should fail with clear error
            assert result.returncode != 0
            assert 'out of range' in result.stderr.lower()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
