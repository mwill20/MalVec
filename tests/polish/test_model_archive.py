"""
Tests for MalVecModel archive module.

Tests archive creation, extraction, inspection, and validation.
"""

import json
import tarfile
import pytest
from pathlib import Path


class TestMalVecModelSaveArchive:
    """Tests for MalVecModel.save_archive()."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        """Create a mock model directory."""
        model_path = tmp_path / "model"
        model_path.mkdir()

        # Create required model files
        (model_path / "model.index").write_bytes(b"fake faiss index data")
        (model_path / "model.config").write_text('{"k": 5, "threshold": 0.7}')
        (model_path / "model.labels.npy").write_bytes(b"fake numpy array data")
        (model_path / "model.meta").write_text(json.dumps({
            "version": "1.0",
            "n_samples": 1000,
            "n_benign": 500,
            "n_malware": 500,
            "embedding_dim": 384
        }))

        return model_path

    def test_create_archive_default_compression(self, model_dir, tmp_path):
        """Test creating archive with default gzip compression."""
        from malvec.model import MalVecModel

        archive_path = tmp_path / "test.malvec"
        MalVecModel.save_archive(model_dir, archive_path)

        assert archive_path.exists()
        assert archive_path.stat().st_size > 0

        # Verify it's a valid gzip tarball
        with tarfile.open(archive_path, "r:gz") as tar:
            names = tar.getnames()
            assert "model.index" in names
            assert "model.config" in names
            assert "model.labels.npy" in names
            assert "model.meta" in names

    def test_create_archive_no_compression(self, model_dir, tmp_path):
        """Test creating archive without compression."""
        from malvec.model import MalVecModel

        archive_path = tmp_path / "test_uncompressed.malvec"
        MalVecModel.save_archive(model_dir, archive_path, compression="")

        assert archive_path.exists()

        # Uncompressed should be larger
        with tarfile.open(archive_path, "r:") as tar:
            assert len(tar.getnames()) == 4

    def test_create_archive_bz2_compression(self, model_dir, tmp_path):
        """Test creating archive with bz2 compression."""
        from malvec.model import MalVecModel

        archive_path = tmp_path / "test.malvec"
        MalVecModel.save_archive(model_dir, archive_path, compression="bz2")

        assert archive_path.exists()

        with tarfile.open(archive_path, "r:bz2") as tar:
            assert len(tar.getnames()) == 4

    def test_create_archive_xz_compression(self, model_dir, tmp_path):
        """Test creating archive with xz/LZMA compression."""
        from malvec.model import MalVecModel

        archive_path = tmp_path / "test.malvec"
        MalVecModel.save_archive(model_dir, archive_path, compression="xz")

        assert archive_path.exists()

        with tarfile.open(archive_path, "r:xz") as tar:
            assert len(tar.getnames()) == 4

    def test_create_archive_invalid_compression(self, model_dir, tmp_path):
        """Test that invalid compression raises ValueError."""
        from malvec.model import MalVecModel

        archive_path = tmp_path / "test.malvec"

        with pytest.raises(ValueError, match="Invalid compression"):
            MalVecModel.save_archive(model_dir, archive_path, compression="invalid")

    def test_create_archive_missing_directory(self, tmp_path):
        """Test that missing directory raises FileNotFoundError."""
        from malvec.model import MalVecModel

        missing_dir = tmp_path / "nonexistent"
        archive_path = tmp_path / "test.malvec"

        with pytest.raises(FileNotFoundError, match="not found"):
            MalVecModel.save_archive(missing_dir, archive_path)

    def test_create_archive_missing_files(self, tmp_path):
        """Test that missing model files raises FileNotFoundError."""
        from malvec.model import MalVecModel

        # Create incomplete model directory
        incomplete_dir = tmp_path / "incomplete"
        incomplete_dir.mkdir()
        (incomplete_dir / "model.index").write_bytes(b"data")
        # Missing other required files

        archive_path = tmp_path / "test.malvec"

        with pytest.raises(FileNotFoundError, match="Missing required"):
            MalVecModel.save_archive(incomplete_dir, archive_path)


class TestMalVecModelLoadArchive:
    """Tests for MalVecModel.load_archive()."""

    @pytest.fixture
    def archive_path(self, tmp_path):
        """Create a test archive."""
        from malvec.model import MalVecModel

        model_path = tmp_path / "model"
        model_path.mkdir()

        (model_path / "model.index").write_bytes(b"fake index")
        (model_path / "model.config").write_text('{"k": 5}')
        (model_path / "model.labels.npy").write_bytes(b"fake labels")
        (model_path / "model.meta").write_text('{"version": "1.0"}')

        archive = tmp_path / "test.malvec"
        MalVecModel.save_archive(model_path, archive)
        return archive

    def test_load_archive_to_temp(self, archive_path):
        """Test extracting archive to temp directory."""
        from malvec.model import MalVecModel

        extract_dir = MalVecModel.load_archive(archive_path)

        assert extract_dir.exists()
        assert (extract_dir / "model.index").exists()
        assert (extract_dir / "model.config").exists()
        assert (extract_dir / "model.labels.npy").exists()
        assert (extract_dir / "model.meta").exists()

    def test_load_archive_to_specific_dir(self, archive_path, tmp_path):
        """Test extracting archive to specified directory."""
        from malvec.model import MalVecModel

        extract_dir = tmp_path / "extracted"
        result_dir = MalVecModel.load_archive(archive_path, extract_dir)

        assert result_dir == extract_dir
        assert (extract_dir / "model.index").exists()

    def test_load_archive_creates_directory(self, archive_path, tmp_path):
        """Test that extract creates directory if needed."""
        from malvec.model import MalVecModel

        extract_dir = tmp_path / "nested" / "path"
        MalVecModel.load_archive(archive_path, extract_dir)

        assert extract_dir.exists()

    def test_load_archive_missing_file(self, tmp_path):
        """Test loading non-existent archive."""
        from malvec.model import MalVecModel

        missing = tmp_path / "nonexistent.malvec"

        with pytest.raises(FileNotFoundError):
            MalVecModel.load_archive(missing)

    def test_load_archive_invalid_format(self, tmp_path):
        """Test loading invalid archive."""
        from malvec.model import MalVecModel

        invalid = tmp_path / "invalid.malvec"
        invalid.write_text("not a tarball")

        with pytest.raises(ValueError, match="Invalid"):
            MalVecModel.load_archive(invalid)

    def test_load_archive_path_traversal_protection(self, tmp_path):
        """Test that path traversal attempts are blocked."""
        from malvec.model import MalVecModel

        # Create malicious archive with path traversal
        malicious = tmp_path / "malicious.malvec"
        with tarfile.open(malicious, "w:gz") as tar:
            # Add a file with path traversal
            info = tarfile.TarInfo(name="../../../etc/passwd")
            info.size = 4
            from io import BytesIO
            tar.addfile(info, BytesIO(b"data"))

        with pytest.raises(ValueError, match="unsafe path"):
            MalVecModel.load_archive(malicious)


class TestMalVecModelInspect:
    """Tests for MalVecModel.inspect()."""

    @pytest.fixture
    def archive_with_meta(self, tmp_path):
        """Create archive with specific metadata."""
        model_path = tmp_path / "model"
        model_path.mkdir()

        metadata = {
            "version": "2.0",
            "n_samples": 50000,
            "n_benign": 25000,
            "n_malware": 25000,
            "k": 7,
            "threshold": 0.8,
            "embedding_dim": 384
        }

        (model_path / "model.index").write_bytes(b"index")
        (model_path / "model.config").write_text('{}')
        (model_path / "model.labels.npy").write_bytes(b"labels")
        (model_path / "model.meta").write_text(json.dumps(metadata))

        from malvec.model import MalVecModel
        archive = tmp_path / "test.malvec"
        MalVecModel.save_archive(model_path, archive)
        return archive, metadata

    def test_inspect_returns_metadata(self, archive_with_meta):
        """Test inspect() returns metadata dict."""
        from malvec.model import MalVecModel

        archive_path, expected_meta = archive_with_meta
        meta = MalVecModel.inspect(archive_path)

        assert meta["version"] == expected_meta["version"]
        assert meta["n_samples"] == expected_meta["n_samples"]
        assert meta["k"] == expected_meta["k"]

    def test_inspect_missing_archive(self, tmp_path):
        """Test inspect() with missing archive."""
        from malvec.model import MalVecModel

        with pytest.raises(FileNotFoundError):
            MalVecModel.inspect(tmp_path / "missing.malvec")

    def test_inspect_invalid_json(self, tmp_path):
        """Test inspect() with invalid JSON metadata."""
        from malvec.model import MalVecModel

        # Create archive with invalid JSON in model.meta
        archive = tmp_path / "bad_json.malvec"
        with tarfile.open(archive, "w:gz") as tar:
            for name in ["model.index", "model.config", "model.labels.npy"]:
                info = tarfile.TarInfo(name=name)
                info.size = 4
                from io import BytesIO
                tar.addfile(info, BytesIO(b"data"))

            # Invalid JSON
            info = tarfile.TarInfo(name="model.meta")
            content = b"not valid json {"
            info.size = len(content)
            tar.addfile(info, BytesIO(content))

        with pytest.raises(ValueError, match="Invalid metadata"):
            MalVecModel.inspect(archive)


class TestMalVecModelListContents:
    """Tests for MalVecModel.list_contents()."""

    @pytest.fixture
    def archive_path(self, tmp_path):
        """Create test archive."""
        from malvec.model import MalVecModel

        model_path = tmp_path / "model"
        model_path.mkdir()

        (model_path / "model.index").write_bytes(b"x" * 1000)
        (model_path / "model.config").write_text('{"k": 5}')
        (model_path / "model.labels.npy").write_bytes(b"y" * 500)
        (model_path / "model.meta").write_text('{"version": "1.0"}')

        archive = tmp_path / "test.malvec"
        MalVecModel.save_archive(model_path, archive)
        return archive

    def test_list_contents(self, archive_path):
        """Test listing archive contents."""
        from malvec.model import MalVecModel

        contents = MalVecModel.list_contents(archive_path)

        assert len(contents) == 4

        names = [c["name"] for c in contents]
        assert "model.index" in names
        assert "model.config" in names

        # Check size is reported
        index_entry = next(c for c in contents if c["name"] == "model.index")
        assert index_entry["size"] == 1000
        assert index_entry["is_file"] is True

    def test_list_contents_missing_archive(self, tmp_path):
        """Test list_contents() with missing archive."""
        from malvec.model import MalVecModel

        with pytest.raises(FileNotFoundError):
            MalVecModel.list_contents(tmp_path / "missing.malvec")


class TestMalVecModelValidate:
    """Tests for MalVecModel.validate()."""

    @pytest.fixture
    def valid_archive(self, tmp_path):
        """Create valid archive."""
        from malvec.model import MalVecModel

        model_path = tmp_path / "model"
        model_path.mkdir()

        (model_path / "model.index").write_bytes(b"index")
        (model_path / "model.config").write_text('{}')
        (model_path / "model.labels.npy").write_bytes(b"labels")
        (model_path / "model.meta").write_text('{"version": "1.0"}')

        archive = tmp_path / "valid.malvec"
        MalVecModel.save_archive(model_path, archive)
        return archive

    def test_validate_valid_archive(self, valid_archive):
        """Test validate() returns True for valid archive."""
        from malvec.model import MalVecModel

        assert MalVecModel.validate(valid_archive) is True

    def test_validate_missing_file(self, tmp_path):
        """Test validate() raises for missing required file."""
        from malvec.model import MalVecModel

        # Create incomplete archive
        archive = tmp_path / "incomplete.malvec"
        with tarfile.open(archive, "w:gz") as tar:
            info = tarfile.TarInfo(name="model.index")
            info.size = 4
            from io import BytesIO
            tar.addfile(info, BytesIO(b"data"))

        with pytest.raises(ValueError, match="Missing required"):
            MalVecModel.validate(archive)

    def test_validate_missing_archive(self, tmp_path):
        """Test validate() raises for non-existent file."""
        from malvec.model import MalVecModel

        with pytest.raises(ValueError, match="not found"):
            MalVecModel.validate(tmp_path / "missing.malvec")


class TestGetModelInfo:
    """Tests for get_model_info() helper function."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        """Create model directory."""
        model_path = tmp_path / "model"
        model_path.mkdir()

        meta = {"version": "1.0", "n_samples": 1000}
        (model_path / "model.meta").write_text(json.dumps(meta))

        return model_path

    @pytest.fixture
    def model_archive(self, tmp_path):
        """Create model archive."""
        from malvec.model import MalVecModel

        model_path = tmp_path / "model_src"
        model_path.mkdir()

        (model_path / "model.index").write_bytes(b"index")
        (model_path / "model.config").write_text('{}')
        (model_path / "model.labels.npy").write_bytes(b"labels")
        (model_path / "model.meta").write_text('{"version": "2.0", "n_samples": 2000}')

        archive = tmp_path / "model.malvec"
        MalVecModel.save_archive(model_path, archive)
        return archive

    def test_get_model_info_from_directory(self, model_dir):
        """Test getting info from directory."""
        from malvec.model import get_model_info

        info = get_model_info(model_dir)

        assert info["version"] == "1.0"
        assert info["n_samples"] == 1000

    def test_get_model_info_from_archive(self, model_archive):
        """Test getting info from archive."""
        from malvec.model import get_model_info

        info = get_model_info(model_archive)

        assert info["version"] == "2.0"
        assert info["n_samples"] == 2000

    def test_get_model_info_invalid_path(self, tmp_path):
        """Test invalid path raises error."""
        from malvec.model import get_model_info

        invalid = tmp_path / "invalid.txt"
        invalid.write_text("not a model")

        with pytest.raises(ValueError, match="Not a valid model"):
            get_model_info(invalid)


class TestArchiveIntegration:
    """Integration tests for complete archive workflow."""

    def test_full_archive_workflow(self, tmp_path):
        """Test complete create -> inspect -> extract -> validate workflow."""
        from malvec.model import MalVecModel

        # Setup
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        (model_dir / "model.index").write_bytes(b"index" * 100)
        (model_dir / "model.config").write_text('{"k": 5, "threshold": 0.7}')
        (model_dir / "model.labels.npy").write_bytes(b"labels" * 50)
        (model_dir / "model.meta").write_text(json.dumps({
            "version": "1.0",
            "n_samples": 10000,
            "created": "2024-01-15"
        }))

        archive_path = tmp_path / "model.malvec"

        # Create
        MalVecModel.save_archive(model_dir, archive_path, compression="gz")
        assert archive_path.exists()

        # Validate
        assert MalVecModel.validate(archive_path) is True

        # Inspect
        meta = MalVecModel.inspect(archive_path)
        assert meta["version"] == "1.0"
        assert meta["n_samples"] == 10000

        # List contents
        contents = MalVecModel.list_contents(archive_path)
        assert len(contents) == 4

        # Extract
        extract_dir = tmp_path / "extracted"
        result_dir = MalVecModel.load_archive(archive_path, extract_dir)

        assert result_dir == extract_dir
        assert (extract_dir / "model.index").exists()
        assert (extract_dir / "model.config").exists()
        assert (extract_dir / "model.labels.npy").exists()
        assert (extract_dir / "model.meta").exists()

        # Verify extracted content matches
        with open(extract_dir / "model.meta") as f:
            extracted_meta = json.load(f)
        assert extracted_meta == meta
