"""
MalVec Model Archive Module.

Provides single-file model distribution using tar archives.
Replaces multiple files with a single .malvec archive for easy
deployment and distribution.

Archive Contents:
- model.index (FAISS index)
- model.config (classifier configuration)
- model.labels.npy (training labels)
- model.meta (model metadata)

Usage:
    # Create archive from model directory
    MalVecModel.save_archive(Path("./model"), Path("model.malvec"))

    # Load archive for classification
    model_dir = MalVecModel.load_archive(Path("model.malvec"))

    # Inspect without extracting
    meta = MalVecModel.inspect(Path("model.malvec"))
"""

import os
import sys
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List


# Required files in a valid model archive
REQUIRED_MODEL_FILES = [
    "model.index",
    "model.config",
    "model.labels.npy",
    "model.meta",
]


class MalVecModel:
    """
    Single-file model archive handler.

    Provides static methods to create, load, and inspect .malvec
    model archives. Archives use tar format with optional compression.

    Supported compression:
    - 'gz': gzip (default, good balance)
    - 'bz2': bzip2 (better compression, slower)
    - 'xz': LZMA (best compression, slowest)
    - '': no compression (fastest)
    """

    @staticmethod
    def save_archive(
        model_dir: Path,
        output_path: Path,
        compression: str = "gz"
    ) -> None:
        """
        Create a single .malvec archive from model directory.

        Args:
            model_dir: Directory containing model files.
            output_path: Path for output .malvec file.
            compression: Compression type ('gz', 'bz2', 'xz', or '').

        Raises:
            FileNotFoundError: If required model files are missing.
            ValueError: If compression type is invalid.

        Example:
            MalVecModel.save_archive(
                Path("./model"),
                Path("malvec_v1.malvec"),
                compression="gz"
            )
        """
        model_dir = Path(model_dir)
        output_path = Path(output_path)

        # Validate compression
        valid_compressions = ("gz", "bz2", "xz", "")
        if compression not in valid_compressions:
            raise ValueError(
                f"Invalid compression '{compression}'. "
                f"Use one of: {valid_compressions}"
            )

        # Validate model directory exists
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Validate required files exist
        missing = []
        for fname in REQUIRED_MODEL_FILES:
            if not (model_dir / fname).exists():
                missing.append(fname)

        if missing:
            raise FileNotFoundError(
                f"Missing required model files: {', '.join(missing)}"
            )

        # Create archive
        mode = f"w:{compression}" if compression else "w"
        with tarfile.open(output_path, mode) as tar:
            for fname in REQUIRED_MODEL_FILES:
                file_path = model_dir / fname
                tar.add(file_path, arcname=fname)

        # Report success
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Model archived to {output_path}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Compression: {compression or 'none'}")

    @staticmethod
    def load_archive(
        archive_path: Path,
        extract_dir: Optional[Path] = None,
        cleanup: bool = False
    ) -> Path:
        """
        Extract .malvec archive to a directory.

        Args:
            archive_path: Path to .malvec archive.
            extract_dir: Where to extract (default: temp directory).
            cleanup: If True, register directory for cleanup on exit.

        Returns:
            Path to extracted model directory.

        Raises:
            FileNotFoundError: If archive doesn't exist.
            ValueError: If archive is invalid or incomplete.

        Example:
            model_dir = MalVecModel.load_archive(Path("model.malvec"))
            clf = KNNClassifier.load(str(model_dir / "model"))
        """
        archive_path = Path(archive_path)

        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        # Create extract directory
        if extract_dir is None:
            extract_dir = Path(tempfile.mkdtemp(prefix="malvec_model_"))
        else:
            extract_dir = Path(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)

        # Extract archive (auto-detects compression)
        try:
            with tarfile.open(archive_path, "r:*") as tar:
                # Security: Check for path traversal
                for member in tar.getmembers():
                    if member.name.startswith('/') or '..' in member.name:
                        raise ValueError(
                            f"Archive contains unsafe path: {member.name}"
                        )

                tar.extractall(extract_dir)
        except tarfile.TarError as e:
            raise ValueError(f"Invalid archive: {e}")

        # Validate extraction
        missing = []
        for fname in REQUIRED_MODEL_FILES:
            if not (extract_dir / fname).exists():
                missing.append(fname)

        if missing:
            raise ValueError(
                f"Archive missing required files: {', '.join(missing)}"
            )

        return extract_dir

    @staticmethod
    def inspect(archive_path: Path) -> Dict[str, Any]:
        """
        Show model metadata without fully extracting.

        Args:
            archive_path: Path to .malvec archive.

        Returns:
            Model metadata dictionary.

        Raises:
            FileNotFoundError: If archive doesn't exist.
            ValueError: If archive is invalid.

        Example:
            meta = MalVecModel.inspect(Path("model.malvec"))
            print(f"Model trained on {meta['n_samples']} samples")
        """
        archive_path = Path(archive_path)

        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        try:
            with tarfile.open(archive_path, "r:*") as tar:
                # Extract just the metadata file
                meta_file = tar.extractfile("model.meta")
                if meta_file is None:
                    raise ValueError("Archive missing model.meta")

                metadata = json.load(meta_file)

        except tarfile.TarError as e:
            raise ValueError(f"Invalid archive: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid metadata JSON: {e}")

        return metadata

    @staticmethod
    def list_contents(archive_path: Path) -> List[Dict[str, Any]]:
        """
        List contents of archive without extracting.

        Args:
            archive_path: Path to .malvec archive.

        Returns:
            List of file info dicts with name, size, mtime.

        Example:
            contents = MalVecModel.list_contents(Path("model.malvec"))
            for item in contents:
                print(f"{item['name']}: {item['size']} bytes")
        """
        archive_path = Path(archive_path)

        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        contents = []
        with tarfile.open(archive_path, "r:*") as tar:
            for member in tar.getmembers():
                contents.append({
                    "name": member.name,
                    "size": member.size,
                    "mtime": member.mtime,
                    "is_file": member.isfile(),
                })

        return contents

    @staticmethod
    def validate(archive_path: Path) -> bool:
        """
        Validate archive without extracting.

        Args:
            archive_path: Path to .malvec archive.

        Returns:
            True if archive is valid.

        Raises:
            ValueError: If archive is invalid (with reason).
        """
        archive_path = Path(archive_path)

        if not archive_path.exists():
            raise ValueError(f"Archive not found: {archive_path}")

        try:
            with tarfile.open(archive_path, "r:*") as tar:
                names = tar.getnames()

                # Check all required files present
                for required in REQUIRED_MODEL_FILES:
                    if required not in names:
                        raise ValueError(f"Missing required file: {required}")

                # Check no path traversal
                for name in names:
                    if name.startswith('/') or '..' in name:
                        raise ValueError(f"Unsafe path in archive: {name}")

                # Try to read metadata
                meta_file = tar.extractfile("model.meta")
                if meta_file:
                    json.load(meta_file)

        except tarfile.TarError as e:
            raise ValueError(f"Invalid tar archive: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid metadata JSON: {e}")

        return True


def get_model_info(model_path: Path) -> Dict[str, Any]:
    """
    Get model information from directory or archive.

    Automatically handles both directory and .malvec archive formats.

    Args:
        model_path: Path to model directory or .malvec file.

    Returns:
        Model metadata dictionary.

    Example:
        info = get_model_info(Path("./model"))
        # or
        info = get_model_info(Path("model.malvec"))
    """
    model_path = Path(model_path)

    # Check directory first (tarfile.is_tarfile fails on directories on Windows)
    if model_path.is_dir():
        # It's a directory
        meta_path = model_path / "model.meta"
        if not meta_path.exists():
            raise FileNotFoundError(f"No model.meta in {model_path}")

        with open(meta_path) as f:
            return json.load(f)
    elif model_path.suffix == '.malvec' or (model_path.is_file() and tarfile.is_tarfile(str(model_path))):
        # It's an archive
        return MalVecModel.inspect(model_path)
    else:
        raise ValueError(f"Not a valid model path: {model_path}")
