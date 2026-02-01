# Lesson 01: Production-Ready Project Architecture

> **Track:** Professional  
> **Phase:** 1 - Project Foundation  
> **Duration:** 60-90 minutes  
> **Prerequisites:** Python package development, Git workflows, CI/CD concepts

---

## 🎯 Learning Objectives

By the end of this lesson, you will:

1. Design a Python package structure for production deployment
2. Configure security boundaries through infrastructure (not just code)
3. Implement decision documentation as a first-class practice
4. Understand the EMBER-first strategy for ML security tools
5. Set up a reproducible development environment

---

## 📚 Advanced Concepts

### 1. Two-Mode Architecture Design

MalVec operates in two modes with fundamentally different security profiles:

```python
class MalVecPipeline:
    def __init__(self, mode: str = 'ember'):
        """
        Args:
            mode: 'ember' for pre-extracted features (safe)
                  'binary' for raw file analysis (requires sandbox)
        """
        self.mode = mode
        
        # Core components (always loaded)
        self.embedder = EmbeddingGenerator(...)
        self.store = VectorStore(...)
        self.classifier = KNNClassifier(...)
        
        # Security components (binary mode only)
        if mode == 'binary':
            self.validator = SampleValidator()
            self.extractor = FeatureExtractor()
```

**Design Pattern:** Mode selection happens once at initialization. Components are conditionally loaded, not conditionally called.

**Why this matters:**

- Clear audit trail: "What mode was this running in?"
- Resource efficiency: Don't load validator/extractor if not needed
- Security boundaries: Binary mode can require elevated permissions

### 2. Defense-in-Depth Through Structure

Security isn't just code - it's infrastructure:

| Layer | Implementation | Failure Mode |
|-------|----------------|--------------|
| .gitignore | Prevent sample commits | Git history contamination |
| data/ directory | Isolate samples from code | Accidental execution |
| Mode separation | Different initialization paths | Feature confusion |
| Type hints | Catch API misuse | Runtime errors |

```python
# malvec/__init__.py - Security as documentation
"""
Security Invariants (NON-NEGOTIABLE):
1. Malware NEVER executes - static analysis only
2. File paths NEVER in output - use hashes
3. All inputs validated at boundaries
4. Sandboxing enforced - timeouts, no network
5. Fail safely - manual review on errors
"""
```

### 3. Dependency Management as Documentation

Organize `requirements.txt` by build phase, not alphabetically:

```
# Phase 2: EMBER Integration
# NOTE: Install from git
# pip install git+https://github.com/elastic/ember.git

# Phase 3: Embeddings
sentence-transformers>=2.2.0
torch>=2.0.0

# Phase 4: Vector Database
faiss-cpu>=1.7.4

# Phase 7: Binary Pipeline (deferred)
pefile>=2023.2.7
lief>=0.13.0
```

**Benefits:**

- New developers understand the build sequence
- Clear what's needed at each stage
- Enables partial installation for CI

### 4. The Lessons Learned Pattern

Real-time decision capture, not post-hoc documentation:

```markdown
## Lesson: EMBER-First Development Strategy

**Date:** 2026-02-01  
**Context:** Choosing between full pipeline vs EMBER-first  
**Problem:** Malware samples pose security risk during development  
**Solution:** Use pre-extracted EMBER features for Phases 2-6  
**Impact:** Build order changed, validator/extractor deferred to Phase 7  
**Takeaway:** Separate novel approach (embeddings) from infrastructure (binary handling)
```

**Anti-pattern:** "We'll document at the end" → You won't, and context is lost.

---

## 🔧 Hands-On Exercises

### Exercise 1.1: Audit the Security Boundaries

Review the MalVec `.gitignore` and identify:

1. All executable file extensions blocked
2. All secret/credential patterns blocked
3. Any missing patterns you'd add

**Deliverable:** A PR-ready `.gitignore` addition with comments explaining each line.

### Exercise 1.2: Design a Mode Configuration

Create a configuration class that enforces mode constraints:

```python
# exercises/mode_config.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional

class PipelineMode(Enum):
    EMBER = "ember"
    BINARY = "binary"

@dataclass
class MalVecConfig:
    mode: PipelineMode
    
    # Binary mode requirements
    sandbox_enabled: bool = False
    max_sample_size_mb: float = 10.0
    
    def __post_init__(self):
        # TODO: Implement validation
        # Binary mode MUST have sandbox_enabled = True
        pass
    
    def validate(self) -> list[str]:
        """Return list of configuration errors."""
        errors = []
        # TODO: Add validation rules
        return errors
```

**Requirements:**

1. Binary mode requires `sandbox_enabled = True`
2. EMBER mode cannot have `max_sample_size_mb` < 1.0
3. Return descriptive error messages

### Exercise 1.3: Create a Decision Record

Document a hypothetical decision using the lessons_learned template:

**Scenario:** You're choosing between FAISS and ChromaDB for vector storage.

Write a complete decision record including:

- Context (what you were trying to do)
- Options considered (with pros/cons)
- Decision made
- Rationale
- Impact on other components

---

## ✅ Checkpoint: Design Review

### Review Question 1: Why defer binary handling to Phase 7?

**Expected answer points:**

- Zero malware risk during core development
- Faster iteration on the novel part (embeddings)
- Proves approach works before adding infrastructure
- Enables safe teaching for novice track
- Validator/extractor are infrastructure, not innovation

### Review Question 2: How does mode separation improve security?

**Expected answer points:**

- Clear audit trail for incidents
- Conditional loading prevents feature confusion
- Binary mode can require elevated permissions/sandbox
- EMBER mode is safe for CI/development
- No scattered conditionals throughout code

### Review Question 3: What makes a good `.gitignore` for security tools?

**Expected answer points:**

- All executable formats (PE, ELF, Mach-O extensions)
- Dataset files (often large, sometimes contain samples)
- Secrets and credentials (.env, *.pem,*.key)
- Generated files (embeddings, models)
- IDE and OS artifacts
- Comments explaining WHY each category is excluded

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    MalVec Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐     Mode Selection (init-time)             │
│  │   Config    │─────────────────────────────┐              │
│  └─────────────┘                             │              │
│                                              ▼              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   EMBER Mode                         │    │
│  │  ┌──────────┐   ┌──────────┐   ┌────────────┐       │    │
│  │  │ Embedder │ → │  Store   │ → │ Classifier │       │    │
│  │  └──────────┘   └──────────┘   └────────────┘       │    │
│  │       ↑                                              │    │
│  │       │ Pre-extracted features                       │    │
│  │       │ (EMBER JSON)                                 │    │
│  └───────┼─────────────────────────────────────────────┘    │
│          │                                                   │
│  ┌───────┼─────────────────────────────────────────────┐    │
│  │       │              BINARY Mode                     │    │
│  │  ┌────┴─────┐   ┌───────────┐                       │    │
│  │  │Validator │ → │ Extractor │                       │    │
│  │  └──────────┘   └───────────┘                       │    │
│  │       ↑                                              │    │
│  │       │ Raw binary                                   │    │
│  │       │ (PE/ELF file) ← SANDBOXED                   │    │
│  └───────┼─────────────────────────────────────────────┘    │
│          │                                                   │
│  ┌───────┴───────────────────────────────────────────────┐  │
│  │                   Security Layer                       │  │
│  │  • Malware NEVER executes                             │  │
│  │  • File paths NEVER in output                         │  │
│  │  • All inputs validated                               │  │
│  │  • Sandboxing enforced                                │  │
│  │  • Fail safely                                        │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎓 Key Takeaways

1. **Mode as configuration, not code:** Initialize differently, not "if mode everywhere"

2. **Security through structure:** .gitignore, directory isolation, and mode separation are infrastructure, not code

3. **Dependencies tell a story:** Phase-organized requirements.txt is documentation

4. **Decisions decay without documentation:** Capture in real-time, not retrospectively

5. **Defer infrastructure, prove innovation:** EMBER-first validates the embedding approach before building binary handling

---

## 📖 Further Reading

- [Twelve-Factor App - Config](https://12factor.net/config)
- [Architecture Decision Records (ADR)](https://adr.github.io/)
- [Python Packaging User Guide](https://packaging.python.org/)
- [EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models](https://arxiv.org/abs/1804.04637)

---

## ➡️ Next Lesson

**Lesson 02: EMBER Integration & Feature Engineering** - Load the EMBER dataset, explore feature vectors, and understand the 2,381-dimensional feature space.

---

*Lesson created during Phase 1 build. Last updated: 2026-02-01*
