# MalVec Build Instructions

## 🎯 Build Sequence

This document guides the implementation in the correct order to ensure security boundaries are established before functionality.

---

## 📋 Pre-Build Checklist

- [ ] Read PROJECT_BRIEF.md (context)
- [ ] Read NORTHSTAR.md (architecture)
- [ ] Read LESSON_PLAN.md (what we'll teach)
- [ ] Understand Defense-in-Depth layers

---

## 🏗️ Phase 1: Project Foundation

### 1.1 Repository Structure

```bash
mkdir -p malvec/{__init__.py,validator.py,extractor.py,embedder.py,store.py,classifier.py,utils.py}
mkdir -p tests/{unit,integration,e2e}
mkdir -p scripts
mkdir -p research
mkdir -p docs
mkdir -p data/{samples,embeddings,models}
```

### 1.2 Dependencies

Create `requirements.txt`:
```
# Static analysis
pefile>=2023.2.7
lief>=0.13.0

# Embeddings
sentence-transformers>=2.2.0
torch>=2.0.0

# Vector DB
faiss-cpu>=1.7.4  # or faiss-gpu for CUDA

# Data & utils
numpy>=1.24.0
scikit-learn>=1.3.0
pandas>=2.0.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Visualization
matplotlib>=3.7.0
plotly>=5.14.0
```

### 1.3 Setup Script

Create `setup.py` for package installation.

---

## 🛡️ Phase 2: Security Layer (Priority 1)

**Philosophy:** Establish security boundaries BEFORE adding functionality.

### 2.1 Sample Validator (`malvec/validator.py`)

**Purpose:** Ensure inputs are safe to process (Layer 1 defense)

**Test-First Approach:**
```python
# tests/unit/test_validator.py
def test_reject_oversized_file():
    """Files >100MB should be rejected"""
    pass

def test_reject_wrong_file_type():
    """Only PE/ELF files allowed"""
    pass

def test_verify_magic_bytes():
    """Magic bytes must match declared type"""
    pass
```

**Implementation Requirements:**
- Max file size: 100MB
- Allowed types: `.exe`, `.dll`, `.elf`, `.so`
- Magic byte verification
- Return sanitized metadata (NEVER raw paths)
- Hash computation (SHA256)

**Security Invariants:**
- ✅ File existence checked before reading
- ✅ Size checked before loading into memory
- ✅ Magic bytes verified
- ✅ Output contains NO raw file paths

---

### 2.2 Sandbox Configuration (`malvec/utils.py`)

**Purpose:** Isolate dangerous operations (Layer 2 defense)

**Components:**
```python
class Sandbox:
    """Context manager for isolated execution"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    def __enter__(self):
        # Set resource limits
        # Disable network access (if possible)
        pass
    
    def __exit__(self, *args):
        # Cleanup
        pass
```

**Test Cases:**
```python
def test_sandbox_timeout():
    """Operations exceeding timeout are killed"""
    pass

def test_sandbox_resource_limits():
    """CPU/memory limits are enforced"""
    pass
```

---

## 🔧 Phase 3: Feature Extraction

### 3.1 Feature Extractor (`malvec/extractor.py`)

**Purpose:** Extract static features WITHOUT execution (Layer 2 defense)

**Test-First Approach:**
```python
# tests/unit/test_extractor.py
def test_extract_pe_imports():
    """Extract import table from PE file"""
    pass

def test_extract_elf_sections():
    """Extract section headers from ELF"""
    pass

def test_timeout_on_large_file():
    """Extraction times out on pathological inputs"""
    pass

def test_graceful_failure_on_corrupt_pe():
    """Malformed PEs don't crash extractor"""
    pass
```

**Implementation:**
```python
class FeatureExtractor:
    TIMEOUT_SECONDS = 30
    
    def extract(self, sample: ValidatedSample) -> FeatureVector:
        """
        Extract features in sandboxed subprocess
        
        Returns:
            FeatureVector with:
            - imports: List[str]
            - sections: List[str]
            - entropy: float
            - strings: List[str]
        """
        with Sandbox(timeout=self.TIMEOUT_SECONDS):
            # Use pefile for PE files
            # Use LIEF for ELF files
            # Catch ALL parser exceptions
            pass
```

**Security Invariants:**
- ✅ Runs in subprocess with timeout
- ✅ All parser exceptions caught
- ✅ No network access during extraction
- ✅ Returns structured data (no raw binary)

---

## 🧠 Phase 4: Embedding Generation

### 4.1 Embedding Generator (`malvec/embedder.py`)

**Purpose:** Convert features to vectors (Layer 3 defense: model isolation)

**Test-First Approach:**
```python
def test_embedding_generation():
    """Features → 768-dim normalized vector"""
    pass

def test_model_checksum_verification():
    """Reject tampered models"""
    pass

def test_embedding_validation():
    """Reject NaN/zero vectors"""
    pass
```

**Implementation:**
```python
class EmbeddingGenerator:
    def __init__(self, model_path: Path, expected_hash: str):
        # Verify model checksum
        actual_hash = self._compute_hash(model_path)
        if actual_hash != expected_hash:
            raise ModelTamperError("Checksum mismatch")
        
        self.model = self._load_model(model_path)
    
    def generate(self, features: FeatureVector) -> Embedding:
        """
        Generate embedding with validation
        
        Returns:
            768-dim numpy array (normalized to unit vector)
        """
        # Convert features to text
        # Generate embedding
        # VALIDATE output (no NaN, correct shape)
        # Normalize to unit vector
        pass
```

**Security Invariants:**
- ✅ Model integrity verified (checksum)
- ✅ Embeddings validated (NaN check, shape check)
- ✅ Runs with timeout
- ✅ Output normalized (prevents injection)

---

## 🗄️ Phase 5: Vector Storage

### 5.1 Vector Store (`malvec/store.py`)

**Purpose:** Persist embeddings for similarity search

**Test-First Approach:**
```python
def test_store_embedding():
    """Store embedding + metadata"""
    pass

def test_find_similar():
    """K-NN search returns top-k neighbors"""
    pass

def test_parameterized_queries():
    """SQL injection prevention"""
    pass
```

**Implementation:**
```python
class VectorStore:
    def __init__(self, db_path: Path):
        self._ensure_permissions()  # 600 on DB file
        self.index = self._load_or_create_index()
        self.metadata_db = self._init_metadata_db()
    
    def store(self, sample_hash: str, embedding: Embedding, label: str):
        """Store with ACID guarantees"""
        # Add to FAISS index
        # Store metadata in SQLite (parameterized query)
        pass
    
    def find_similar(self, query: Embedding, k: int = 10) -> List[SimilarSample]:
        """Cosine similarity search"""
        # FAISS search
        # Retrieve metadata
        pass
```

**Security Invariants:**
- ✅ Database file has 600 permissions
- ✅ All queries parameterized (SQL injection prevention)
- ✅ Transactions used (ACID guarantees)
- ✅ Index corruption detected on startup

---

## 🎯 Phase 6: Classification

### 6.1 K-NN Classifier (`malvec/classifier.py`)

**Purpose:** Make malicious/benign decision

**Test-First Approach:**
```python
def test_knn_voting():
    """Majority vote from k neighbors"""
    pass

def test_confidence_threshold():
    """Low confidence → flag for review"""
    pass

def test_no_single_neighbor_dictates():
    """One outlier can't override majority"""
    pass
```

**Implementation:**
```python
class KNNClassifier:
    def __init__(self, k: int = 10, confidence_threshold: float = 0.7):
        self.k = k
        self.confidence_threshold = confidence_threshold
    
    def classify(self, neighbors: List[SimilarSample]) -> ClassificationResult:
        """
        K-NN voting with confidence scoring
        
        Returns:
            ClassificationResult with:
            - label: str
            - confidence: float
            - flagged_for_review: bool
            - reasoning: str
        """
        # Count votes
        # Calculate confidence
        # Flag if below threshold
        pass
```

**Security Invariants:**
- ✅ Requires K neighbors (no single-vote decisions)
- ✅ Confidence threshold enforced
- ✅ Low-confidence samples flagged
- ✅ Reasoning provided (explainability)

---

## 🖥️ Phase 7: CLI Interface

### 7.1 Training Script (`scripts/train.py`)

**Purpose:** Build vector DB from labeled samples

```python
# Usage: python scripts/train.py --samples /path/to/labeled/samples
```

**Features:**
- Load labeled samples
- Extract features
- Generate embeddings
- Build FAISS index
- Store metadata

### 7.2 Classification Script (`scripts/classify.py`)

**Purpose:** Classify unknown sample

```python
# Usage: python scripts/classify.py /path/to/unknown.exe
```

**Output:**
```
Classification: MALICIOUS
Confidence: 0.87 (87%)
Reasoning: 9/10 neighbors voted 'malicious'
Top Similar Samples:
  1. wannacry_variant_2.exe (similarity: 0.95)
  2. wannacry_variant_1.exe (similarity: 0.93)
  ...
```

### 7.3 Visualization Script (`scripts/visualize.py`)

**Purpose:** Generate cluster plots

```python
# Usage: python scripts/visualize.py --output clusters.html
```

**Creates:**
- t-SNE dimensionality reduction
- Interactive Plotly scatter plot
- Color-coded by label

---

## 🧪 Phase 8: Testing

### 8.1 Unit Tests

**Coverage target: >80%**

Test each component in isolation:
- Validator
- Extractor
- Embedder
- Store
- Classifier

### 8.2 Integration Tests

Test component interactions:
- Validator → Extractor
- Extractor → Embedder
- Embedder → Store
- Store → Classifier

### 8.3 End-to-End Tests

Full pipeline tests:
```python
def test_classify_known_malware():
    """Known malware → 'malicious' classification"""
    pass

def test_classify_benign_software():
    """Benign software → 'benign' classification"""
    pass

def test_polymorphic_variant_detection():
    """Variants of same family cluster together"""
    pass
```

### 8.4 Adversarial Tests

**Critical:**
```python
def test_malware_never_executes():
    """Process monitor confirms no code execution"""
    pass

def test_timeout_on_obfuscated_pe():
    """Heavily obfuscated files timeout gracefully"""
    pass

def test_corrupt_pe_doesnt_crash():
    """Malformed PEs are rejected cleanly"""
    pass
```

---

## 📚 Phase 9: Documentation

### 9.1 README.md

**Contents:**
- What is MalVec?
- Quick start (3 commands)
- Installation
- Basic usage
- Links to deeper docs

### 9.2 ARCHITECTURE.md

**Copy from NORTHSTAR.md, then update with:**
- Actual implementation details
- Performance benchmarks
- Lessons learned

### 9.3 API.md

**Document all public classes:**
- Type signatures
- Example usage
- Error codes

### 9.4 lessons_learned.md

**Capture as you build:**
- What worked
- What didn't
- Decisions and trade-offs
- Common pitfalls

**Example entry:**
```markdown
## Lesson: Embedding Validation is Non-Negotiable

**Date:** 2025-02-01
**Context:** Initial implementation didn't validate embeddings
**Problem:** Corrupted PE → NaN embedding → FAISS crash
**Solution:** Added validation in EmbeddingGenerator.generate()
**Impact:** Zero crashes after 10,000+ samples
**Takeaway:** Always validate ML outputs as untrusted input
```

---

## 🎓 Phase 10: Lessons (Post-Build)

**Only AFTER tool is working:**

1. Use pedagogy templates in `docs/pedagogy_templates/`
2. Generate novice track (16 lessons)
3. Generate professional track (22 lessons)
4. Use actual code for walkthroughs
5. Create hands-on labs with real tool

**Why wait:**
- Lessons need working code examples
- Labs need executable commands
- Metrics need real benchmarks
- Can't teach what doesn't exist yet

---

## 🔑 Key Principles Throughout

**Security First:**
- Validate at boundaries
- Sandbox dangerous operations
- Fail safely (manual review > auto-classify)

**Test First:**
- Write failing test
- Implement until passing
- Refactor with confidence

**Document Decisions:**
- Why this approach?
- What alternatives considered?
- What trade-offs accepted?

**Enable Teaching:**
- Clear component boundaries
- Self-documenting structure
- Demonstrable concepts

---

## ✅ Completion Checklist

**Phase 1: Foundation**
- [ ] Repository structure created
- [ ] Dependencies installed
- [ ] Setup script works

**Phase 2: Security Layer**
- [ ] Validator rejects bad inputs
- [ ] Sandbox isolates operations
- [ ] All security tests pass

**Phase 3-6: Core Components**
- [ ] Feature extraction works
- [ ] Embeddings generate correctly
- [ ] Vector store persists data
- [ ] Classifier makes decisions

**Phase 7: CLI**
- [ ] Training script builds DB
- [ ] Classification script works
- [ ] Visualization generates plots

**Phase 8: Testing**
- [ ] >80% code coverage
- [ ] All integration tests pass
- [ ] Adversarial tests pass
- [ ] No malware executions

**Phase 9: Documentation**
- [ ] README complete
- [ ] ARCHITECTURE updated
- [ ] API documented
- [ ] Lessons learned captured

**Phase 10: Lessons**
- [ ] Templates ready
- [ ] Novice track generated
- [ ] Professional track generated
- [ ] Labs validated

---

**You're ready to build. Remember: Security over speed, tests before code, document as you go.**
