"""
MalVec Configuration Module.

Provides YAML-based configuration for MalVec settings.
Supports loading from file, environment variables, and programmatic
configuration.

Example config.yaml:
    # Model settings
    model_path: ./production_model
    embedding_dim: 384
    k: 5
    confidence_threshold: 0.7

    # Security settings
    sandbox_enabled: true
    timeout: 30
    max_memory: 536870912  # 512MB

    # Logging
    audit_log: /var/log/malvec/audit.log
    verbose: false

Usage:
    # Load from file
    config = MalVecConfig.from_file(Path("config.yaml"))

    # Use defaults
    config = MalVecConfig()

    # Save current config
    config.save(Path("config.yaml"))
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict


def _check_yaml_available() -> bool:
    """Check if PyYAML is available."""
    try:
        import yaml
        return True
    except ImportError:
        return False


HAS_YAML = _check_yaml_available()


def _default_audit_log() -> Path:
    """Get default audit log path based on platform."""
    if sys.platform == 'win32':
        base = Path(os.environ.get('LOCALAPPDATA', '.'))
        return base / 'malvec' / 'logs' / 'audit.log'
    else:
        return Path('/var/log/malvec/audit.log')


@dataclass
class MalVecConfig:
    """
    Global MalVec configuration.

    Can be loaded from YAML file, environment variables, or
    created programmatically with defaults.

    Attributes:
        model_path: Path to model directory or archive.
        embedding_dim: Embedding dimension (must match model).
        k: Number of neighbors for K-NN.
        confidence_threshold: Threshold for auto-classification.
        sandbox_enabled: Whether to sandbox feature extraction.
        timeout: Extraction timeout in seconds.
        max_memory: Maximum memory for extraction in bytes.
        max_filesize: Maximum input file size in bytes.
        audit_log: Path to audit log file.
        verbose: Enable verbose output.
    """

    # Model settings
    model_path: Path = field(default_factory=lambda: Path("./model"))
    embedding_dim: int = 384
    k: int = 5
    confidence_threshold: float = 0.7

    # Security settings
    sandbox_enabled: bool = True
    timeout: int = 30
    max_memory: int = 512 * 1024 * 1024  # 512MB
    max_filesize: int = 50 * 1024 * 1024  # 50MB

    # Logging settings
    audit_log: Path = field(default_factory=_default_audit_log)
    verbose: bool = False

    def __post_init__(self):
        """Validate and convert configuration values."""
        # Convert string paths to Path objects
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
        if isinstance(self.audit_log, str):
            self.audit_log = Path(self.audit_log)

        # Validate ranges
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(
                f"confidence_threshold must be 0-1, got {self.confidence_threshold}"
            )
        if self.timeout < 1:
            raise ValueError(f"timeout must be >= 1, got {self.timeout}")
        if self.max_memory < 1024 * 1024:  # At least 1MB
            raise ValueError(f"max_memory too small: {self.max_memory}")
        if self.max_filesize < 1024:  # At least 1KB
            raise ValueError(f"max_filesize too small: {self.max_filesize}")

    @classmethod
    def from_file(cls, path: Path) -> 'MalVecConfig':
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            MalVecConfig instance.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ImportError: If PyYAML is not installed.
            ValueError: If configuration is invalid.
        """
        if not HAS_YAML:
            raise ImportError(
                "PyYAML required for config files. Install with: pip install pyyaml"
            )

        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        # Convert paths from strings
        if 'model_path' in data:
            data['model_path'] = Path(data['model_path'])
        if 'audit_log' in data:
            data['audit_log'] = Path(data['audit_log'])

        return cls(**data)

    @classmethod
    def from_env(cls) -> 'MalVecConfig':
        """
        Load configuration from environment variables.

        Environment variables:
        - MALVEC_MODEL_PATH
        - MALVEC_EMBEDDING_DIM
        - MALVEC_K
        - MALVEC_CONFIDENCE_THRESHOLD
        - MALVEC_SANDBOX_ENABLED
        - MALVEC_TIMEOUT
        - MALVEC_MAX_MEMORY
        - MALVEC_MAX_FILESIZE
        - MALVEC_AUDIT_LOG
        - MALVEC_VERBOSE

        Returns:
            MalVecConfig with values from environment.
        """
        data = {}

        env_mapping = {
            'MALVEC_MODEL_PATH': ('model_path', Path),
            'MALVEC_EMBEDDING_DIM': ('embedding_dim', int),
            'MALVEC_K': ('k', int),
            'MALVEC_CONFIDENCE_THRESHOLD': ('confidence_threshold', float),
            'MALVEC_SANDBOX_ENABLED': ('sandbox_enabled', lambda x: x.lower() == 'true'),
            'MALVEC_TIMEOUT': ('timeout', int),
            'MALVEC_MAX_MEMORY': ('max_memory', int),
            'MALVEC_MAX_FILESIZE': ('max_filesize', int),
            'MALVEC_AUDIT_LOG': ('audit_log', Path),
            'MALVEC_VERBOSE': ('verbose', lambda x: x.lower() == 'true'),
        }

        for env_var, (attr, converter) in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    data[attr] = converter(value)
                except (ValueError, TypeError):
                    pass  # Skip invalid values, use defaults

        return cls(**data)

    def save(self, path: Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Output path for YAML file.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        if not HAS_YAML:
            raise ImportError(
                "PyYAML required for config files. Install with: pip install pyyaml"
            )

        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict with string paths
        data = self.to_dict()

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"Configuration saved to {path}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary with all configuration values.
        """
        data = asdict(self)
        # Convert Path objects to strings for serialization
        data['model_path'] = str(data['model_path'])
        data['audit_log'] = str(data['audit_log'])
        return data

    def merge(self, other: 'MalVecConfig') -> 'MalVecConfig':
        """
        Merge another config into this one.

        Values from other config override this config.

        Args:
            other: Config to merge from.

        Returns:
            New MalVecConfig with merged values.
        """
        merged_data = self.to_dict()
        other_data = other.to_dict()

        # Only override non-default values
        for key, value in other_data.items():
            if value is not None:
                merged_data[key] = value

        return MalVecConfig(**merged_data)


def get_default_config() -> MalVecConfig:
    """
    Get default configuration.

    Checks for config file in standard locations:
    1. ./malvec.yaml (current directory)
    2. ~/.config/malvec/config.yaml (user config)
    3. /etc/malvec/config.yaml (system config, Linux only)

    Returns:
        MalVecConfig with defaults or from config file.
    """
    # Check standard locations
    config_paths = [
        Path("malvec.yaml"),
        Path("malvec.yml"),
        Path.home() / ".config" / "malvec" / "config.yaml",
    ]

    if sys.platform != 'win32':
        config_paths.append(Path("/etc/malvec/config.yaml"))

    for path in config_paths:
        if path.exists():
            try:
                return MalVecConfig.from_file(path)
            except Exception:
                pass  # Fall through to defaults

    # No config file found, use defaults
    return MalVecConfig()


def generate_example_config(path: Path = None) -> str:
    """
    Generate example configuration file content.

    Args:
        path: Optional path to write file.

    Returns:
        YAML configuration string.
    """
    example = """# MalVec Configuration
# Copy this file to malvec.yaml and customize as needed

# Model settings
model_path: ./model
embedding_dim: 384
k: 5
confidence_threshold: 0.7

# Security settings
sandbox_enabled: true
timeout: 30
max_memory: 536870912  # 512MB
max_filesize: 52428800  # 50MB

# Logging
audit_log: /var/log/malvec/audit.log
verbose: false
"""

    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(example)
        print(f"Example configuration written to {path}")

    return example
