"""
MalVec - Malware Detection via Embedding-Space Analysis

A secure, production-ready tool for identifying polymorphic malware variants
using embedding geometry and K-NN classification.

Security Invariants (NON-NEGOTIABLE):
1. Malware NEVER executes - static analysis only
2. File paths NEVER in output - use hashes
3. All inputs validated at boundaries
4. Sandboxing enforced - timeouts, no network
5. Fail safely - manual review on errors

Two Modes:
- 'ember': Pre-extracted EMBER features (development/research)
- 'binary': Raw binary analysis (production)
"""

__version__ = "0.1.0"
__author__ = "MalVec Team"

# Lazy imports to avoid loading everything at once
# Components will be imported as needed

# Phase 2: EMBER loader
from malvec import ember_loader

# Phase 3: Embedding generator
from malvec import embedder

__all__ = [
    "__version__",
    "__author__",
    "ember_loader",
    "embedder",
]

