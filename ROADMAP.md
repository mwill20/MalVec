# MalVec Roadmap

## Current Version: 1.0

### Completed ✅

- Core K-NN classification pipeline
- EMBER-compatible feature extraction (2381 dims)
- Johnson-Lindenstrauss embedding (384 dims)
- FAISS vector store with cosine similarity
- Sandboxed execution (subprocess isolation, timeout, memory limits)
- Structured audit logging (SHA256, JSON)
- CLI interface (classify, batch, train, evaluate, archive)
- `.malvec` model archive format
- Docker support
- GitHub Actions CI/CD
- Comprehensive documentation (User Guide, Security, Deployment, API)
- 260+ tests (unit, security, integration, polish)

---

## Upcoming Releases

### v1.1 — Platform Expansion (Q2 2026)

- [ ] ELF binary support (Linux malware)
- [ ] BODMAS dataset integration
- [ ] Performance optimizations (GPU-accelerated FAISS)
- [ ] REST API server (FastAPI)
- [ ] Web UI for interactive classification

### v1.2 — Scale & Intelligence (Q3 2026)

- [ ] Approximate K-NN (FAISS IVF index)
- [ ] Active learning pipeline
- [ ] Model versioning system
- [ ] Batch retraining workflow
- [ ] Explainability dashboard (neighbor visualization)

### v2.0 — Production Platform (Q4 2026)

- [ ] Deep learning embeddings (optional transformer encoder)
- [ ] Multi-model ensemble
- [ ] Real-time detection service (streaming ingestion)
- [ ] SIEM integration (Splunk, Elastic SIEM)
- [ ] Commercial support options

---

## Research Directions

- **Adversarial robustness** — resistance to evasion attacks
- **Zero-day detection** — anomaly detection for unknown families
- **Behavioral analysis** — combining static + dynamic features
- **Cross-platform** — macOS Mach-O, Android APK/DEX support

---

## Community Requests

See [GitHub Issues](https://github.com/mwill20/MalVec/issues) for feature requests and bug reports.

## Contributing

Want to help build the future of MalVec? See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get started.
