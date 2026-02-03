# MalVec API Reference

Complete Python API documentation for MalVec malware classification system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Modules](#core-modules)
   - [Classifier](#classifier)
   - [Extractor](#extractor)
   - [Embedding](#embedding)
3. [Utility Modules](#utility-modules)
   - [Progress](#progress)
   - [Config](#config)
   - [Exceptions](#exceptions)
   - [Model Archive](#model-archive)
4. [Security Modules](#security-modules)
   - [Sandbox](#sandbox)
   - [Audit](#audit)
   - [Isolation](#isolation)
5. [CLI Modules](#cli-modules)
6. [Examples](#examples)

---

## Quick Start

```python
from pathlib import Path
from malvec.classifier import KNNClassifier
from malvec.extractor import FeatureExtractor

# Load model
clf = KNNClassifier.load("./model/model")

# Extract features from PE file
extractor = FeatureExtractor()
features = extractor.extract(Path("sample.exe"))

# Classify
prediction, confidence, neighbors = clf.predict(features)
print(f"Prediction: {prediction}, Confidence: {confidence:.2%}")
```

---

## Core Modules

### Classifier

**Module:** `malvec.classifier`

K-Nearest Neighbors classifier using FAISS for efficient similarity search.

#### KNNClassifier

```python
class KNNClassifier:
    """
    K-NN classifier for malware detection.

    Uses FAISS for efficient similarity search over embeddings.
    """

    def __init__(
        self,
        k: int = 5,
        confidence_threshold: float = 0.7,
        embedding_dim: int = 384
    ):
        """
        Initialize classifier.

        Args:
            k: Number of neighbors for voting.
            confidence_threshold: Minimum confidence for auto-classification.
            embedding_dim: Dimension of embeddings.
        """
```

##### Methods

**fit(embeddings, labels)**

```python
def fit(
    self,
    embeddings: np.ndarray,
    labels: np.ndarray
) -> 'KNNClassifier':
    """
    Fit classifier on training data.

    Args:
        embeddings: Shape (n_samples, embedding_dim).
        labels: Shape (n_samples,), values 0 (benign) or 1 (malware).

    Returns:
        Self for chaining.

    Example:
        clf = KNNClassifier(k=5)
        clf.fit(embeddings, labels)
    """
```

**predict(embedding)**

```python
def predict(
    self,
    embedding: np.ndarray
) -> tuple[str, float, list[dict]]:
    """
    Predict class for a single embedding.

    Args:
        embedding: Shape (embedding_dim,) or (1, embedding_dim).

    Returns:
        Tuple of:
        - prediction: "MALWARE" or "BENIGN"
        - confidence: 0.0 to 1.0
        - neighbors: List of {index, distance, label} dicts

    Example:
        prediction, confidence, neighbors = clf.predict(embedding)
        if confidence >= 0.7:
            print(f"Auto-classified as {prediction}")
        else:
            print("Needs manual review")
    """
```

**save(path)**

```python
def save(self, path: str) -> None:
    """
    Save classifier to disk.

    Creates four files:
    - {path}.index: FAISS index
    - {path}.config: Configuration JSON
    - {path}.labels.npy: Labels array
    - {path}.meta: Metadata JSON

    Args:
        path: Base path (without extension).

    Example:
        clf.save("./model/model")
    """
```

**load(path)** (classmethod)

```python
@classmethod
def load(cls, path: str) -> 'KNNClassifier':
    """
    Load classifier from disk.

    Args:
        path: Base path (without extension).

    Returns:
        Loaded KNNClassifier instance.

    Example:
        clf = KNNClassifier.load("./model/model")
    """
```

##### Properties

- `n_samples: int` - Number of training samples
- `k: int` - Number of neighbors
- `confidence_threshold: float` - Classification threshold
- `embedding_dim: int` - Embedding dimension

---

### Extractor

**Module:** `malvec.extractor`

Feature extraction from PE binaries.

#### FeatureExtractor

```python
class FeatureExtractor:
    """
    Extract features from PE binaries.

    Uses LIEF for parsing and extracts EMBER-compatible features.
    Runs in sandbox for security.
    """

    def __init__(
        self,
        sandbox: bool = True,
        timeout: int = 30,
        max_memory: int = 512 * 1024 * 1024
    ):
        """
        Initialize extractor.

        Args:
            sandbox: Enable sandboxed extraction.
            timeout: Max extraction time (seconds).
            max_memory: Max memory usage (bytes).
        """
```

##### Methods

**extract(file_path)**

```python
def extract(self, file_path: Path) -> np.ndarray:
    """
    Extract features from PE file.

    Args:
        file_path: Path to PE binary.

    Returns:
        Feature vector (numpy array).

    Raises:
        InvalidFileError: If file is not a valid PE.
        ExtractionTimeoutError: If extraction times out.
        ExtractionError: If extraction fails.

    Example:
        extractor = FeatureExtractor()
        features = extractor.extract(Path("sample.exe"))
    """
```

**validate(file_path)**

```python
def validate(self, file_path: Path) -> bool:
    """
    Validate file before extraction.

    Checks:
    - File exists and is readable
    - File size within limits
    - Magic bytes indicate PE format

    Args:
        file_path: Path to check.

    Returns:
        True if valid.

    Raises:
        InvalidFileError: If validation fails.
    """
```

---

### Embedding

**Module:** `malvec.embedding`

Generate embeddings from feature vectors.

#### EmbeddingGenerator

```python
class EmbeddingGenerator:
    """
    Generate embeddings from feature vectors.

    Uses random projection for dimensionality reduction.
    """

    def __init__(
        self,
        input_dim: int = 2381,
        output_dim: int = 384,
        seed: int = 42
    ):
        """
        Initialize generator.

        Args:
            input_dim: Input feature dimension.
            output_dim: Output embedding dimension.
            seed: Random seed for reproducibility.
        """
```

##### Methods

**generate(features)**

```python
def generate(self, features: np.ndarray) -> np.ndarray:
    """
    Generate embeddings from features.

    Args:
        features: Shape (n_samples, input_dim) or (input_dim,).

    Returns:
        Embeddings with shape (n_samples, output_dim) or (output_dim,).

    Example:
        embedder = EmbeddingGenerator()
        embedding = embedder.generate(features)
    """
```

---

## Utility Modules

### Progress

**Module:** `malvec.progress`

Progress reporting with rich terminal output.

#### ProgressReporter

```python
class ProgressReporter:
    """
    User-friendly progress reporting.

    Uses 'rich' for beautiful output, falls back to print.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize reporter.

        Args:
            verbose: Show output (False suppresses all).
        """
```

##### Methods

**task(description, total)**

```python
@contextmanager
def task(
    self,
    description: str,
    total: Optional[int] = None
) -> Iterator[Callable[[int], None]]:
    """
    Context manager for progress task.

    Args:
        description: Task description.
        total: Total steps (None for spinner).

    Yields:
        Update function to call with current progress.

    Example:
        with progress.task("Processing", total=100) as update:
            for i in range(100):
                process_item(i)
                update(i + 1)
    """
```

**status(message)**

```python
def status(self, message: str) -> None:
    """Print success status message."""
```

**error(message)**

```python
def error(self, message: str) -> None:
    """Print error message (always shown)."""
```

**warning(message)**

```python
def warning(self, message: str) -> None:
    """Print warning message."""
```

**info(message)**

```python
def info(self, message: str) -> None:
    """Print info message."""
```

**print_header()**

```python
def print_header(self, title: str = "MalVec") -> None:
    """Print ASCII art header."""
```

**print_table(title, data)**

```python
def print_table(self, title: str, data: dict) -> None:
    """
    Print formatted table.

    Args:
        title: Table title.
        data: Key-value pairs.
    """
```

---

### Config

**Module:** `malvec.config`

YAML-based configuration.

#### MalVecConfig

```python
@dataclass
class MalVecConfig:
    """
    Global configuration.

    Attributes:
        model_path: Path to model.
        embedding_dim: Embedding dimension.
        k: K-NN neighbors.
        confidence_threshold: Classification threshold.
        sandbox_enabled: Enable sandboxing.
        timeout: Extraction timeout (seconds).
        max_memory: Max memory (bytes).
        max_filesize: Max file size (bytes).
        audit_log: Path to audit log.
        verbose: Enable verbose output.
    """

    model_path: Path = Path("./model")
    embedding_dim: int = 384
    k: int = 5
    confidence_threshold: float = 0.7
    sandbox_enabled: bool = True
    timeout: int = 30
    max_memory: int = 512 * 1024 * 1024
    max_filesize: int = 50 * 1024 * 1024
    audit_log: Path = Path("/var/log/malvec/audit.log")
    verbose: bool = False
```

##### Methods

**from_file(path)** (classmethod)

```python
@classmethod
def from_file(cls, path: Path) -> 'MalVecConfig':
    """
    Load config from YAML file.

    Args:
        path: Path to YAML file.

    Returns:
        Loaded configuration.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ImportError: If pyyaml not installed.

    Example:
        config = MalVecConfig.from_file(Path("config.yaml"))
    """
```

**from_env()** (classmethod)

```python
@classmethod
def from_env(cls) -> 'MalVecConfig':
    """
    Load config from environment variables.

    Reads MALVEC_* environment variables.

    Returns:
        Configuration with env overrides.

    Example:
        # export MALVEC_K=7
        config = MalVecConfig.from_env()
        assert config.k == 7
    """
```

**save(path)**

```python
def save(self, path: Path) -> None:
    """Save configuration to YAML file."""
```

**to_dict()**

```python
def to_dict(self) -> dict:
    """Convert to dictionary."""
```

#### Helper Functions

**get_default_config()**

```python
def get_default_config() -> MalVecConfig:
    """
    Get default config, checking standard locations.

    Checks:
    1. ./malvec.yaml
    2. ~/.config/malvec/config.yaml
    3. /etc/malvec/config.yaml (Linux)

    Returns:
        Config from file or defaults.
    """
```

**generate_example_config(path)**

```python
def generate_example_config(path: Path = None) -> str:
    """
    Generate example YAML config.

    Args:
        path: Optional path to write file.

    Returns:
        YAML string.
    """
```

---

### Exceptions

**Module:** `malvec.exceptions`

User-friendly exception classes.

#### Exception Hierarchy

```
MalVecError (base)
├── ModelNotFoundError
├── InvalidModelError
├── InvalidFileError
├── FileNotFoundError
├── ExtractionTimeoutError
├── ExtractionError
├── ValidationError
├── SandboxViolationError
├── ConfigurationError
├── DatasetError
└── ClassificationError
```

#### MalVecError

```python
class MalVecError(Exception):
    """
    Base exception with helpful hints.

    Attributes:
        message: Error description.
        hint: Suggestion for resolution.
    """

    def __init__(self, message: str, hint: str = None):
        """
        Initialize error.

        Args:
            message: What went wrong.
            hint: How to fix it.
        """
```

#### Common Exceptions

```python
# Model not found
raise ModelNotFoundError(Path("./model"))
# Error: Model not found at ./model
# Hint: Train a model first: python -m malvec.cli.train --output ./model

# Invalid file
raise InvalidFileError(Path("test.txt"), "Not a PE file")
# Error: Invalid file test.txt: Not a PE file
# Hint: Ensure file is a valid PE binary and < 50MB

# Extraction timeout
raise ExtractionTimeoutError(timeout=30)
# Error: Feature extraction exceeded 30s timeout
# Hint: File may be malformed. Try increasing timeout with --timeout

# Sandbox violation
raise SandboxViolationError(
    "Memory limit exceeded",
    violation_type="memory"
)
# Error: Memory limit exceeded
# Hint: Increase memory limit or process smaller files.
```

#### Helper Function

```python
def format_exception_for_cli(exc: Exception) -> str:
    """
    Format exception for CLI display.

    Args:
        exc: Exception to format.

    Returns:
        User-friendly message without traceback.
    """
```

---

### Model Archive

**Module:** `malvec.model`

Single-file model distribution.

#### MalVecModel

```python
class MalVecModel:
    """
    Single-file model archive handler.

    Creates .malvec archives containing:
    - model.index (FAISS)
    - model.config (JSON)
    - model.labels.npy (NumPy)
    - model.meta (JSON)
    """
```

##### Static Methods

**save_archive(model_dir, output_path, compression)**

```python
@staticmethod
def save_archive(
    model_dir: Path,
    output_path: Path,
    compression: str = "gz"
) -> None:
    """
    Create archive from model directory.

    Args:
        model_dir: Directory with model files.
        output_path: Output .malvec path.
        compression: 'gz', 'bz2', 'xz', or '' (none).

    Example:
        MalVecModel.save_archive(
            Path("./model"),
            Path("model_v1.malvec")
        )
    """
```

**load_archive(archive_path, extract_dir)**

```python
@staticmethod
def load_archive(
    archive_path: Path,
    extract_dir: Path = None
) -> Path:
    """
    Extract archive to directory.

    Args:
        archive_path: .malvec file.
        extract_dir: Where to extract (default: temp).

    Returns:
        Path to extracted directory.

    Example:
        model_dir = MalVecModel.load_archive(Path("model.malvec"))
        clf = KNNClassifier.load(str(model_dir / "model"))
    """
```

**inspect(archive_path)**

```python
@staticmethod
def inspect(archive_path: Path) -> dict:
    """
    Get metadata without full extraction.

    Args:
        archive_path: .malvec file.

    Returns:
        Metadata dictionary.

    Example:
        meta = MalVecModel.inspect(Path("model.malvec"))
        print(f"Samples: {meta['n_samples']}")
    """
```

**list_contents(archive_path)**

```python
@staticmethod
def list_contents(archive_path: Path) -> list[dict]:
    """
    List archive contents.

    Returns:
        List of {name, size, mtime, is_file}.
    """
```

**validate(archive_path)**

```python
@staticmethod
def validate(archive_path: Path) -> bool:
    """
    Validate archive integrity.

    Raises:
        ValueError: If invalid.

    Returns:
        True if valid.
    """
```

---

## Security Modules

### Sandbox

**Module:** `malvec.sandbox`

Sandboxed execution with resource limits.

#### SandboxConfig

```python
@dataclass
class SandboxConfig:
    """
    Sandbox configuration.

    Attributes:
        timeout: Max execution time (seconds).
        max_memory: Max memory (bytes).
        max_filesize: Max input file size (bytes).
        allow_network: Allow network access (advisory).
    """

    timeout: int = 30
    max_memory: int = 512 * 1024 * 1024
    max_filesize: int = 50 * 1024 * 1024
    allow_network: bool = False
```

#### SandboxContext

```python
class SandboxContext:
    """
    Context manager for sandboxed execution.

    Example:
        config = SandboxConfig(timeout=30)
        with SandboxContext(config) as sandbox:
            result = sandbox.run(dangerous_function, arg1, arg2)
    """

    def __init__(self, config: SandboxConfig):
        """Initialize with configuration."""

    def run(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run function in sandbox.

        Args:
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Function result.

        Raises:
            SandboxViolationError: If limits exceeded.
        """
```

---

### Audit

**Module:** `malvec.audit`

Structured audit logging.

#### AuditLogger

```python
class AuditLogger:
    """
    Structured JSON audit logging.

    Logs security-relevant events for SIEM integration.
    """

    def __init__(
        self,
        log_dir: Path = None,
        include_file_names: bool = False
    ):
        """
        Initialize logger.

        Args:
            log_dir: Directory for log files.
            include_file_names: Include names (False = hash only).
        """
```

##### Methods

**log_classification(...)**

```python
def log_classification(
    self,
    file_path: Path,
    prediction: str,
    confidence: float,
    processing_time_ms: float,
    needs_review: bool = False
) -> None:
    """
    Log classification result.

    Args:
        file_path: Classified file (hashed).
        prediction: MALWARE or BENIGN.
        confidence: 0.0 to 1.0.
        processing_time_ms: Processing duration.
        needs_review: Flagged for manual review.
    """
```

**log_validation_failure(...)**

```python
def log_validation_failure(
    self,
    file_path: Path,
    reason: str,
    error_type: str
) -> None:
    """Log validation failure."""
```

**log_sandbox_violation(...)**

```python
def log_sandbox_violation(
    self,
    file_path: Path,
    violation: str,
    violation_type: str
) -> None:
    """Log sandbox security violation."""
```

---

### Isolation

**Module:** `malvec.isolation`

Process isolation for crash protection.

#### run_isolated

```python
def run_isolated(
    func: Callable,
    *args,
    timeout: int = 30,
    **kwargs
) -> Any:
    """
    Run function in isolated subprocess.

    Provides crash isolation - if LIEF crashes, main process survives.

    Args:
        func: Function to run.
        *args: Positional arguments.
        timeout: Max execution time.
        **kwargs: Keyword arguments.

    Returns:
        Function result.

    Raises:
        ExtractionTimeoutError: If timeout exceeded.
        ExtractionError: If subprocess crashes.

    Example:
        features = run_isolated(extract_features, file_path, timeout=30)
    """
```

---

## CLI Modules

### Train CLI

**Module:** `malvec.cli.train`

```bash
python -m malvec.cli.train [OPTIONS]

Options:
  --output, -o PATH      Output model directory (required)
  --max-samples, -n INT  Max training samples [default: 10000]
  --k INT                K-NN neighbors [default: 5]
  --threshold FLOAT      Confidence threshold [default: 0.6]
  --embedding-dim INT    Embedding dimension [default: 384]
  --verbose, -v          Show progress
  --config, -c PATH      YAML config file
```

### Classify CLI

**Module:** `malvec.cli.classify`

```bash
python -m malvec.cli.classify [OPTIONS]

Options:
  --model, -m PATH       Model directory or .malvec file (required)
  --file, -f PATH        PE file to classify (required)
  --show-confidence, -c  Show confidence percentage
  --show-neighbors, -n   Show K nearest neighbors
  --k INT                Override K value
  --json                 Output as JSON
  --timeout INT          Extraction timeout [default: 30]
```

### Batch CLI

**Module:** `malvec.cli.batch`

```bash
python -m malvec.cli.batch [OPTIONS]

Options:
  --model, -m PATH       Model directory (required)
  --input-dir PATH       Directory with samples (required)
  --output PATH          Output CSV file (required)
  --recursive            Search subdirectories
  --parallel INT         Worker count [default: 4]
  --timeout INT          Per-file timeout [default: 30]
```

### Archive CLI

**Module:** `malvec.cli.archive`

```bash
# Create archive
python -m malvec.cli.archive create --model DIR --output FILE [--compression gz|bz2|xz]

# Inspect archive
python -m malvec.cli.archive inspect FILE

# Extract archive
python -m malvec.cli.archive extract FILE --output DIR

# List contents
python -m malvec.cli.archive list FILE
```

---

## Examples

### Complete Classification Pipeline

```python
from pathlib import Path
from malvec.classifier import KNNClassifier
from malvec.extractor import FeatureExtractor
from malvec.embedding import EmbeddingGenerator
from malvec.progress import ProgressReporter
from malvec.audit import AuditLogger
from malvec.exceptions import MalVecError

# Initialize components
clf = KNNClassifier.load("./model/model")
extractor = FeatureExtractor(sandbox=True, timeout=30)
embedder = EmbeddingGenerator()
progress = ProgressReporter(verbose=True)
audit = AuditLogger()

def classify_file(file_path: Path) -> dict:
    """Classify a single PE file."""
    try:
        # Extract features
        progress.status(f"Processing {file_path.name}")
        features = extractor.extract(file_path)

        # Generate embedding
        embedding = embedder.generate(features)

        # Classify
        prediction, confidence, neighbors = clf.predict(embedding)
        needs_review = confidence < clf.confidence_threshold

        # Log result
        audit.log_classification(
            file_path, prediction, confidence,
            processing_time_ms=123,
            needs_review=needs_review
        )

        return {
            "file": str(file_path),
            "prediction": prediction,
            "confidence": confidence,
            "needs_review": needs_review
        }

    except MalVecError as e:
        progress.error(str(e))
        return {"file": str(file_path), "error": str(e)}

# Run classification
result = classify_file(Path("sample.exe"))
print(result)
```

### Batch Processing with Progress

```python
from pathlib import Path
from malvec.progress import ProgressReporter

progress = ProgressReporter(verbose=True)
samples = list(Path("./samples").glob("*.exe"))

results = []
with progress.task("Classifying samples", total=len(samples)) as update:
    for i, sample in enumerate(samples):
        result = classify_file(sample)
        results.append(result)
        update(i + 1)

# Summary
malware_count = sum(1 for r in results if r.get("prediction") == "MALWARE")
progress.status(f"Found {malware_count}/{len(results)} malware samples")
```

### Using Configuration

```python
from malvec.config import MalVecConfig, get_default_config

# Load from file
config = MalVecConfig.from_file(Path("config.yaml"))

# Or get defaults (checks standard locations)
config = get_default_config()

# Or from environment
config = MalVecConfig.from_env()

# Use config
clf = KNNClassifier(
    k=config.k,
    confidence_threshold=config.confidence_threshold,
    embedding_dim=config.embedding_dim
)

extractor = FeatureExtractor(
    sandbox=config.sandbox_enabled,
    timeout=config.timeout,
    max_memory=config.max_memory
)
```

### Working with Archives

```python
from pathlib import Path
from malvec.model import MalVecModel, get_model_info

# Create archive
MalVecModel.save_archive(
    Path("./model"),
    Path("production_model.malvec"),
    compression="gz"
)

# Inspect without extracting
meta = MalVecModel.inspect(Path("production_model.malvec"))
print(f"Model has {meta['n_samples']} samples")

# Load for classification
model_dir = MalVecModel.load_archive(Path("production_model.malvec"))
clf = KNNClassifier.load(str(model_dir / "model"))

# Works with both directories and archives
info = get_model_info(Path("production_model.malvec"))
```

---

## Version History

| Version | Changes |
|---------|---------|
| 1.0 | Initial API documentation |

---

## See Also

- [User Guide](USER_GUIDE.md)
- [Security Architecture](SECURITY.md)
- [Deployment Guide](DEPLOYMENT.md)
