# MalVec Project Brief

## 🎯 One-Sentence Summary

MalVec is a malware detection system that uses **embedding-space analysis** to identify malicious binaries by their semantic similarity, enabling detection of polymorphic variants that evade traditional signature-based approaches.

---

## 🔥 The Problem

**Traditional signature-based antivirus fails against polymorphic malware:**
- Attackers mutate malware to evade exact-match signatures
- Each variant requires manual reverse engineering
- Detection lags behind variant creation (cat-and-mouse game)
- 40% of malware samples are polymorphic variants

**The Gap:**
Existing ML approaches use hand-crafted features (imports, entropy, strings). These miss **semantic similarity** - malware families that behave similarly but look different structurally.

---

## 💡 The Solution

**Embedding-space detection:**
1. Convert malware binaries to high-dimensional vectors (embeddings)
2. Similar malware clusters in embedding space
3. Classify unknowns by their nearest neighbors
4. Detect polymorphic variants without retraining

**Key Insight:**
Malware families live in "neighborhoods" - distance in embedding space reveals relationships that feature engineering misses.

---

## 🏗️ Core Architecture (High-Level)

```
Binary File → Feature Extraction → Embedding Generation → Vector DB → Similarity Search → Classification
     ↓              ↓                     ↓                  ↓              ↓               ↓
[Untrusted]    [Sandboxed]         [Isolated Proc]      [FAISS]      [K-NN]         [Malicious/Benign]
```

**Security Boundaries:**
- Malware NEVER executes (static analysis only)
- Feature extraction runs in sandbox (timeout protection)
- All inputs validated at system boundaries
- Defense-in-Depth across 6 layers

---

## 📊 Success Metrics

**Research Success:**
- [ ] >90% accuracy on test dataset
- [ ] Beats baseline ML (random forest on features) by >5%
- [ ] Detects polymorphic variants in same cluster

**Engineering Success:**
- [ ] Zero malware executions during processing
- [ ] <10% flagged for manual review
- [ ] Processes 1000 samples in <10 minutes

**Operational Success:**
- [ ] Integrates into daily SOC workflow
- [ ] Reduces analyst time by >50%
- [ ] <1% false positive rate on benign software

---

## 🎓 Educational Mission

**Dual-Track Learning System:**

This project teaches BOTH the tool AND the domain through progressive lessons:

**Novice Track (~16 lessons):**
- Malware detection fundamentals
- ML/embedding basics
- Tool usage and metrics
- Career preparation

**Professional Track (~22 lessons):**
- Production architecture decisions
- Performance optimization
- Adversarial robustness
- Operational ML patterns

**Outcome:** Users walk away with working tool + domain expertise + portfolio piece

---

## 🛡️ Security-First Constraints

**CRITICAL INVARIANTS:**
1. Malware samples NEVER execute
2. File paths NEVER appear in user-facing output (prevent info disclosure)
3. Embedding model runs in separate process with timeout
4. All database queries use parameterized statements
5. On error: flag for review (don't auto-classify)

**Defense-in-Depth Layers:**
1. Input validation (file type, size, magic bytes)
2. Sandboxing (subprocess, timeout, no network)
3. Model isolation (separate process, checksum verification)
4. Output filtering (sanitize paths, error messages)
5. Monitoring (log all processing with hashes)
6. Fail-safe defaults (manual review on failures)

---

## 🚀 Build Philosophy

**BMAD Framework:**
- **Breaks:** What failure modes must be prevented? (malware execution, data leakage)
- **Modeling:** Core abstraction? (malware families as clusters in vector space)
- **Automation:** What gets eliminated? (manual variant analysis)
- **Defense:** Security layers? (6-layer defense-in-depth)

**AgentOS Patterns:**
- Single-agent workflow (no multi-agent needed)
- State persistence via vector DB + metadata SQLite
- Tools: pefile, LIEF, sentence-transformers, FAISS

**Implementation Priorities:**
1. **Security first** - All boundaries validated before functionality
2. **Testability** - Write tests before implementation
3. **Documentation** - Architecture decisions recorded in real-time
4. **Lessons** - Code structure enables teaching (clear components)

---

## 📁 Repository Structure (Target)

```
malvec/
├── README.md                 # Quick start
├── ARCHITECTURE.md           # NorthStar blueprint (detailed)
├── CHANGELOG.md             # Version history
├── requirements.txt         # Pinned dependencies
├── setup.py
│
├── docs/
│   ├── api.md
│   ├── training.md
│   ├── deployment.md
│   └── lessons_learned.md   # Institutional knowledge
│
├── malvec/                  # Main package
│   ├── __init__.py
│   ├── validator.py         # Sample validation
│   ├── extractor.py         # Feature extraction
│   ├── embedder.py          # Embedding generation
│   ├── store.py             # Vector storage
│   ├── classifier.py        # K-NN classification
│   └── utils.py
│
├── research/                # Notebooks (not production)
│   ├── 01_baseline_comparison.ipynb
│   ├── 02_embedding_quality.ipynb
│   └── 03_cluster_analysis.ipynb
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── scripts/
│   ├── train.py             # Build vector DB
│   ├── classify.py          # Classify sample (CLI)
│   └── visualize.py         # Generate cluster plots
│
├── lessons/                 # Created AFTER build
│   ├── novice/
│   └── professional/
│
└── data/                    # Gitignored
    ├── samples/
    ├── embeddings/
    └── models/
```

---

## 🎯 Next Steps for Anti Gravity Claude

**Phase 1: Foundation (You Are Here)**
- Read NorthStar blueprint
- Understand security boundaries
- Review build instructions

**Phase 2: Core Implementation**
- Set up project structure
- Implement validators (security layer 1)
- Build feature extractor (sandboxed)
- Create embedder (isolated)

**Phase 3: Detection Engine**
- Vector store (FAISS + SQLite)
- K-NN classifier
- CLI interface

**Phase 4: Testing & Hardening**
- Unit tests for each component
- Integration tests for pipeline
- Adversarial tests (polymorphic samples)
- Security validation

**Phase 5: Documentation**
- Complete API docs
- Deployment guide
- Lessons learned capture

**Phase 6: Lessons (Post-Build)**
- Generate dual-track curriculum
- Use actual code for walkthroughs
- Create hands-on labs with real tool

---

## 🔑 Key Reminders for Builder

**Security Over Speed:**
- Every "quick hack" becomes a CVE
- Validate at boundaries, not deep in logic
- Fail safely (flag for review vs auto-classify)

**Test Before Code:**
- Write failing test first
- Implement until test passes
- Refactor with confidence

**Document Decisions:**
- Why this approach vs alternatives?
- What trade-offs did we accept?
- What would we do differently?

**Enable Teaching:**
- Clear component boundaries
- Self-documenting code structure
- Examples that demonstrate concepts

---

**Builder:** You're not just writing code - you're creating a learning platform that happens to detect malware. Every design decision should enable both functionality AND pedagogy.

**Remember:** A bug that lets malware execute is catastrophic. When in doubt, be conservative.
