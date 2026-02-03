# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-02

### Added
- **Core pipeline**: PE binary feature extraction, embedding generation, FAISS K-NN classification
- **EMBER integration**: Load and process EMBER 2018 dataset with 2381-dimensional feature vectors
- **Embedding engine**: Johnson-Lindenstrauss random projection (384-dim) via sklearn GaussianRandomProjection
- **Vector storage**: FAISS index with add/search/save/load operations
- **K-NN classifier**: k=5 majority voting with configurable confidence threshold
- **CLI tools**: train, classify, batch, evaluate, info, archive commands
- **Native binary analysis**: PE feature extraction using LIEF and pefile (production mode)
- **Security hardening**: Process isolation, sandboxing (30s timeout, 512MB cap), audit logging
- **Input validation**: File size, magic bytes, and format validation at all boundaries
- **Model archives**: `.malvec` tarball format with create/inspect/extract support
- **Configuration**: YAML files and environment variable support via `malvec.config`
- **Custom exceptions**: User-friendly error hierarchy with actionable messages
- **Progress indicators**: Rich terminal output with automatic fallback to plain text
- **Documentation**: User guide, API reference, security docs, deployment guide
- **Educational content**: Dual-track curriculum (novice and professional) with 16 lessons
- **Test suite**: Unit, integration, security, and polish tests

### Security
- Static analysis only — malware never executes
- File paths excluded from all output (SHA256 hashes used instead)
- Defense-in-depth: validation, isolation, sandboxing, audit logging, fail-safe defaults
