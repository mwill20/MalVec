# MalVec: Product Vision

> **Purpose:** This document defines the end-state goals for MalVec. Reference this when making architectural decisions to ensure alignment with user experience goals.

---

## 1. User Interaction Model

### Primary Users

| User Type | Interaction | Goal |
|-----------|-------------|------|
| **SOC Analyst** | CLI commands | Quickly classify suspicious files |
| **Security Engineer** | Python API | Integrate into existing pipelines |
| **Learner** | Lessons + Labs | Build malware analysis skills |
| **Researcher** | Notebooks + Visualizations | Analyze malware families |

---

## 2. CLI Experience (Day-to-Day Usage)

### Scenario 1: Analyst Receives Suspicious File

```powershell
PS> python scripts/classify.py C:\Quarantine\suspicious.exe

╔══════════════════════════════════════════════════════════════╗
║  MalVec Classification Report                                 ║
╠══════════════════════════════════════════════════════════════╣
║  Sample Hash:    a3f2b8c1...e9d4                              ║
║  Classification: MALICIOUS                                    ║
║  Confidence:     94.2%                                        ║
║  Family Match:   Emotet (variant cluster)                     ║
╠══════════════════════════════════════════════════════════════╣
║  REASONING                                                    ║
║  ─────────────────────────────────────────────────────────── ║
║  • 9/10 neighbors voted 'malicious'                          ║
║  • Closest match: emotet_v3_unpacked.dll (0.96 similarity)   ║
║  • Import pattern matches banking trojan family               ║
╠══════════════════════════════════════════════════════════════╣
║  TOP 5 SIMILAR SAMPLES                                        ║
║  ─────────────────────────────────────────────────────────── ║
║  1. emotet_v3_unpacked.dll    │ 0.96 │ malicious             ║
║  2. emotet_v2_packed.exe      │ 0.94 │ malicious             ║
║  3. emotet_variant_7.exe      │ 0.93 │ malicious             ║
║  4. qakbot_loader.dll         │ 0.87 │ malicious             ║
║  5. legitimate_banking.dll    │ 0.42 │ benign                ║
╚══════════════════════════════════════════════════════════════╝

⚠️  RECOMMENDED ACTION: Quarantine and escalate to Tier 2
```

### Scenario 2: Low Confidence → Manual Review

```powershell
PS> python scripts/classify.py C:\Quarantine\ambiguous.exe

╔══════════════════════════════════════════════════════════════╗
║  MalVec Classification Report                                 ║
╠══════════════════════════════════════════════════════════════╣
║  Sample Hash:    7b3e9a2f...c1d8                              ║
║  Classification: UNCERTAIN                                    ║
║  Confidence:     58.3%  ⚠️ BELOW THRESHOLD (70%)              ║
║  Flagged:        MANUAL REVIEW REQUIRED                       ║
╠══════════════════════════════════════════════════════════════╣
║  REASONING                                                    ║
║  ─────────────────────────────────────────────────────────── ║
║  • 6/10 neighbors voted 'malicious', 4/10 'benign'           ║
║  • Sample sits between legitimate installer and dropper      ║
║  • High entropy sections suggest packing OR compression      ║
╚══════════════════════════════════════════════════════════════╝

⚠️  MANUAL ANALYSIS RECOMMENDED - Do not auto-quarantine
```

---

## 3. Training Workflow

```powershell
PS> python scripts/train.py --samples C:\MalwareZoo\labeled\

MalVec Training Pipeline
════════════════════════════════════════════════════════════════

[Phase 1/4] Validating Samples...
  ✓ 1,247 samples validated
  ✗ 23 rejected (corrupt/oversized)
  ⏱️ 12.3 seconds

[Phase 2/4] Extracting Features...
  ✓ Processing sample 1,247/1,247
  ⏱️ 3 minutes 42 seconds

[Phase 3/4] Generating Embeddings...
  ✓ 1,247 embeddings created (768-dim each)
  ⏱️ 8 minutes 15 seconds

[Phase 4/4] Building Vector Index...
  ✓ FAISS index created
  ✓ Metadata stored in SQLite
  ⏱️ 4.2 seconds

════════════════════════════════════════════════════════════════
TRAINING COMPLETE

Database: data/embeddings/malvec.index
Metadata: data/embeddings/metadata.db
Samples:  1,247 (892 malicious, 355 benign)
Ready for classification!
════════════════════════════════════════════════════════════════
```

---

## 4. Visualization (Research/Analysis)

```powershell
PS> python scripts/visualize.py --output clusters.html
```

Generates an interactive Plotly visualization:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MalVec Cluster Visualization                  │
│                                                                  │
│        🔴 Emotet cluster                                         │
│           🔴🔴🔴                                                  │
│              🔴🔴                    🟠 Ransomware cluster       │
│                                        🟠🟠🟠                     │
│                                          🟠🟠                     │
│                                                                  │
│     🟢🟢🟢🟢🟢                                                    │
│       🟢🟢🟢🟢       🟣 Unknown (your sample!)                   │
│         🟢🟢           🟣 ← Sits near Emotet                     │
│      Benign cluster                                              │
│                                                                  │
│  [Hover for details] [Click to filter] [Zoom enabled]           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Python API (For Integration)

```python
from malvec import MalVecPipeline

# Initialize with pre-trained database
pipeline = MalVecPipeline(
    db_path="data/embeddings/malvec.index",
    confidence_threshold=0.70
)

# Classify a sample
result = pipeline.classify("suspicious.exe")

print(result.classification)  # "MALICIOUS"
print(result.confidence)      # 0.942
print(result.family_match)    # "Emotet"
print(result.neighbors[:3])   # Top 3 similar samples

# Batch processing for large-scale analysis
results = pipeline.classify_batch(
    samples=["file1.exe", "file2.dll", "file3.exe"],
    parallel=True
)

# Extract features only (for research)
features = pipeline.extract_features("sample.exe")
print(features.imports)       # ['kernel32.CreateFileW', ...]
print(features.entropy)       # 7.82
print(features.sections)      # ['.text', '.rdata', '.rsrc']

# Generate embedding only (for custom analysis)
embedding = pipeline.generate_embedding("sample.exe")
print(embedding.shape)        # (768,)
```

---

## 6. Learning Experience

### Novice Track: The Journey

A security student starts with zero malware analysis experience and walks away job-ready.

```
┌─────────────────────────────────────────────────────────────────┐
│                    NOVICE LEARNING PATH                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Week 1-2: FOUNDATIONS                                           │
│  ├── Lesson 01: Why Antivirus Fails                             │
│  │   └── Lab: Compare signature vs embedding detection          │
│  ├── Lesson 02: ML Basics for Detection                         │
│  │   └── Lab: Train simple classifier, see it fail on variants  │
│  └── Lesson 03: Embeddings Explained                            │
│      └── Lab: Generate first embedding, visualize similarity    │
│                                                                  │
│  Week 3-4: TOOL MASTERY                                          │
│  ├── Lesson 04-10: Deep dive into each MalVec component         │
│  │   └── Labs: Run each component, observe inputs/outputs       │
│  └── Capstone: End-to-end malware detection lab                 │
│                                                                  │
│  Week 5: DOMAIN KNOWLEDGE                                        │
│  ├── Lesson 11-13: Malware families, evasion, trade-offs        │
│  └── Labs: Identify family from cluster, tune thresholds        │
│                                                                  │
│  Week 6: CAREER PREP                                             │
│  ├── Lesson 14-16: Portfolio, interviews, next steps            │
│  └── Outcome: GitHub showcase + interview readiness             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

GRADUATION OUTCOME:
✓ Can explain embedding-based detection to colleagues
✓ Can run MalVec end-to-end
✓ Can answer entry-level interview questions
✓ Has portfolio project ready for job applications
```

### Professional Track: Production Expertise

```
┌─────────────────────────────────────────────────────────────────┐
│                  PROFESSIONAL LEARNING PATH                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Module 1: ARCHITECTURE DECISIONS                                │
│  ├── Lesson 01-05: Model selection, vector DB trade-offs,       │
│  │                 index optimization, sharding, versioning     │
│  └── Labs: Benchmark alternatives, justify decisions            │
│                                                                  │
│  Module 2: ADVERSARIAL ROBUSTNESS                                │
│  ├── Lesson 06-09: Polymorphic handling, poisoning defense,     │
│  │                 evasion detection, ensemble methods          │
│  └── Labs: Attack your own system, build defenses               │
│                                                                  │
│  Module 3: PRODUCTION ENGINEERING                                │
│  ├── Lesson 10-15: Batch processing, real-time detection,       │
│  │                 GPU acceleration, cost, monitoring, drift    │
│  └── Labs: Optimize to production SLOs                          │
│                                                                  │
│  Module 4: INTEGRATION                                           │
│  ├── Lesson 16-19: SIEM, EDR, threat intel, API design          │
│  └── Labs: Integrate with Splunk, build detection API           │
│                                                                  │
│  Module 5: RESEARCH & INNOVATION                                 │
│  ├── Lesson 20-22: State of art, future directions, contrib     │
│  └── Labs: Replicate paper, submit first PR                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

GRADUATION OUTCOME:
✓ Can design production detection systems
✓ Can optimize for specific constraints (latency, cost, accuracy)
✓ Can answer staff+ interview questions
✓ Can contribute to and extend MalVec
```

---

## 7. Demo Scenarios

### Demo 1: "The Polymorphic Threat" (5 minutes)

*Perfect for security meetups, conference talks, or portfolio showcase.*

```
SETUP: You have 5 variants of the same ransomware family.
       Traditional AV detects only the original.

DEMO FLOW:
1. Show signature-based detection → catches 1/5 variants
2. Run MalVec → catches 5/5 variants
3. Visualize cluster → all 5 cluster together
4. Explain: "Embedding space captures WHAT malware does, 
            not WHAT it looks like"

WOW MOMENT: The visualization shows all variants in same 
            neighborhood despite ~40% code difference
```

### Demo 2: "Zero-Day Detection" (3 minutes)

*Shows the predictive power of embedding similarity.*

```
SETUP: New malware sample, never seen before.

DEMO FLOW:
1. Run classification → "MALICIOUS, 91% confidence"
2. Show reasoning → "9/10 neighbors are Emotet variants"
3. Reveal: This sample was uploaded to VirusTotal today,
          but it clusters with samples from 2 years ago

WOW MOMENT: "We detected this BEFORE signatures existed 
            because it BEHAVES like its ancestors"
```

### Demo 3: "The Security Boundary" (2 minutes)

*For security-conscious audiences.*

```
DEMO FLOW:
1. Show malicious binary in input folder
2. Run classification with process monitor visible
3. Point out: "See? The binary was NEVER executed"
4. Show logs: hash computation, feature extraction, all static

WOW MOMENT: "We analyzed live malware without risk.
            Defense-in-depth means nothing runs."
```

---

## 8. Repository Reader Experience

When someone discovers MalVec on GitHub:

```
┌─────────────────────────────────────────────────────────────────┐
│  📁 MalVec                                                       │
│  ├── 📄 README.md          ← "Get started in 3 commands"        │
│  ├── 📄 ARCHITECTURE.md    ← Deep dive for engineers            │
│  ├── 📁 malvec/            ← Clean, teachable source code       │
│  ├── 📁 tests/             ← "Ah, this is well-tested"          │
│  ├── 📁 lessons/           ← Self-paced learning curriculum     │
│  │   ├── novice/           │
│  │   └── professional/     │
│  ├── 📁 research/          ← Jupyter notebooks for exploration  │
│  └── 📁 docs/              │
│       └── lessons_learned.md ← "What would they do differently?" │
└─────────────────────────────────────────────────────────────────┘

FIRST IMPRESSIONS:
• "This is production-quality, not a toy"
• "I can learn malware analysis from this"
• "The architecture is clear and extensible"
• "Tests prove it works and is secure"
```

---

## 9. Success Metrics

| Metric | Target | Why It Matters |
|--------|--------|----------------|
| **Accuracy** | >90% on test set | Detection must be reliable |
| **Polymorphic Detection** | Variants cluster together | The whole point |
| **Zero Executions** | 0 malware processes | Security is non-negotiable |
| **Processing Speed** | 1000 samples <10 min | Practical for daily use |
| **Low False Positives** | <1% on benign | Analysts trust the tool |
| **Flagged for Review** | <10% uncertain | Humans focus on edge cases |
| **Lesson Completion** | Learners can demo | Educational mission success |

---

## 10. The MalVec Value Proposition

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   FOR SOC ANALYSTS:    Fast, explainable malware classification │
│                                                                  │
│   FOR ENGINEERS:       Production-ready API, integrable         │
│                                                                  │
│   FOR LEARNERS:        Dual-track curriculum from zero to hero  │
│                                                                  │
│   FOR RESEARCHERS:     Visualizable embedding space             │
│                                                                  │
│   FOR PORTFOLIO:       Impressive, demonstrable security tool   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

*Document created: 2026-02-01*
*Last updated: 2026-02-01*
