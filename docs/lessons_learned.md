# MalVec Lessons Learned

> **Purpose:** Capture decisions, problems, solutions, and insights as we build.  
> This document becomes source material for Phase 10 (Lessons).

---

## Lesson: EMBER-First Development Strategy

**Date:** 2026-02-01  
**Context:** Deciding how to approach dataset selection for development  
**Problem:** Building a malware detection system requires malware samples, which pose security risks during development  
**Solution:** Use EMBER dataset (pre-extracted features) for Phases 2-6, add binary handling in Phase 7  
**Impact:**

- Zero malware risk during core development
- Faster iteration (skip validator/extractor initially)
- Two-mode architecture enables safe teaching

**Takeaway:** When security and speed conflict, find an approach that delivers both. Pre-extracted features let us prove the embedding approach works without touching actual malware.

---

## Lesson: Two-Mode Architecture

**Date:** 2026-02-01  
**Context:** Designing the pipeline to support both EMBER features and raw binaries  
**Problem:** Different inputs (JSON features vs PE files) need different processing paths  
**Solution:** Single pipeline class with `mode` parameter that conditionally loads validator/extractor  
**Impact:**

- Clean separation of concerns
- Easy testing (EMBER mode needs no security infrastructure)
- Production mode adds security layers transparently

**Takeaway:** Design for modes, not conditionals scattered throughout code. The mode decision happens once at initialization.

---

## Lesson: EMBER Package Not on PyPI

**Date:** 2026-02-01  
**Context:** Installing dependencies for Phase 2  
**Problem:** `pip install ember-ml>=0.1.0` failed - package doesn't exist on PyPI  
**Solution:** EMBER is installed from GitHub: `pip install git+https://github.com/elastic/ember.git`  
**Impact:** Updated requirements.txt with installation instructions as comments  
**Takeaway:** Always verify package names before assuming PyPI availability. Security/research packages often live only on GitHub.

---

## Lesson: Create Lessons Alongside Code

**Date:** 2026-02-01  
**Context:** Planning when to create teaching materials  
**Problem:** Deferring lessons to "Phase 10" means context is lost and lessons become disconnected from code  
**Solution:** Create both novice and professional lessons immediately after completing each phase  
**Impact:** Lessons directory created with Phase 1 lessons written same day as code  
**Takeaway:** Documentation (including lessons) is not an afterthought - it's a deliverable of every phase.

---

## Lesson: Test Structure Before Tests

**Date:** 2026-02-01  
**Context:** Setting up test infrastructure in Phase 1  
**Decision:** Create test directory structure but no test files yet  
**Rationale:** Test-first means tests wait until Phase 2 when we have something to test  
**Impact:** Clean separation between structure and implementation  
**Takeaway:** Infrastructure before code, structure before tests, always.

---

## Lesson: Synthetic Data Fallback Pattern

**Date:** 2026-02-01  
**Context:** Testing EMBER loader without 7GB dataset download  
**Problem:** Real EMBER data is 7GB - blocks development and CI  
**Solution:** Implement layered fallback: Real data → EMBER library → Synthetic data  
**Impact:** Tests pass immediately, new developers can start without dataset  
**Trade-off:** Synthetic data won't catch data-specific bugs (need integration tests with real data)  
**Takeaway:** Never block development on optional large assets. Provide fallbacks, but mark real-data tests clearly.

---

## Lesson: Test Markers Distinguish Data Requirements

**Date:** 2026-02-01  
**Context:** Phase 2 review revealed synthetic vs real data distinction needed  
**Problem:** Tests pass with synthetic but haven't proven real EMBER works  
**Solution:** Add `@pytest.mark.real_data` marker for tests requiring actual dataset  
**Impact:** CI can run fast (synthetic), pre-release runs full suite (real)  
**Takeaway:** Make implicit data dependencies explicit through test markers.

---

## Lesson: Validation Checkpoints Catch Honest Gaps

**Date:** 2026-02-01  
**Context:** Phase 2→3 transition review  
**Problem:** Easy to say "done" without proving all paths work  
**Solution:** Provide brutally honest assessment when asked for validation  
**Impact:** Conditions set (download EMBER before Phase 5) before they become blockers  
**Takeaway:** Honesty about limitations is more valuable than false confidence.

---

## Lesson: Johnson-Lindenstrauss Saves the Day

**Date:** 2026-02-01  
**Context:** Need to reduce 2381 EMBER dimensions for embedding  
**Problem:** 2381 dims is too large for efficient transformer processing  
**Solution:** Random orthogonal projection (JL lemma) preserves distances  
**Impact:** Simple matrix multiply reduces dims while preserving structure  
**Takeaway:** Sometimes random is good enough - JL guarantees distance preservation with random projection.

---

## Lesson: Text Models Don't Understand Numbers (RETRACTED)

**Date:** 2026-02-01 (Original), 2026-02-01 (Corrected)  
**Context:** Wanted to use sentence-transformers for embedding EMBER features  
**Original Approach:** Convert numbers to text strings, feed to transformer  
**Why It Was Wrong:**

- Transformers tokenize "0.123" as a word, not a numeric value
- Only sampled 48 of 384 features (88% information loss)
- Spearman correlation with original distances: ~0.30 (garbage)

**Correct Approach:** Use sklearn.GaussianRandomProjection

- Proper JL lemma implementation with correct scaling
- Uses ALL features (no sampling)
- Spearman correlation: >0.98 (excellent)

**Lesson:** Never use text models for numeric data. If it's numbers, use numeric methods.

---

## Lesson: QR Orthogonalization Hurts JL Projection

**Date:** 2026-02-01  
**Context:** Thought orthogonal projection would be better than random  
**Problem:** QR-orthogonalized matrix had worse distance preservation  
**Root Cause:** JL lemma requires Gaussian distribution; QR changes this  
**Evidence:** QR correlation 0.30 vs sklearn 0.98  
**Solution:** Use sklearn.GaussianRandomProjection (handles scaling correctly)  
**Takeaway:** Don't "improve" proven algorithms without understanding the math.

---

## Meta-Lesson: Fast Iteration Beats Perfect Planning

**Date:** 2026-02-01  
**Context:** Phase 3 text bridge was fundamentally broken  
**What Happened:**

1. Phase 3 v1 shipped with text bridge (Spearman ~0.30)
2. Review caught it immediately
3. Phase 3 v2 shipped with sklearn (Spearman >0.98)

**Key Insight:** The revision (100 lines, battle-tested sklearn) is **superior**
to what we would have built if we'd overthought it initially.

**Why This Works:**

- Fast iteration with honest feedback beats trying to get it perfect first time
- Empirical validation (distance correlation) caught what code review missed
- Using standard libraries (sklearn) beats custom implementations
- Simpler is almost always better

**The Numbers:**

| Metric | v1 (broken) | v2 (fixed) |
|--------|-------------|------------|
| Features used | 48/384 (12%) | 2381/2381 (100%) |
| Spearman | ~0.30 | >0.98 |
| Lines | ~200 | ~100 |

**Takeaway:** Ship, measure, learn, improve. The loop is more valuable than the plan.

---

## Lesson: Separation of Concerns in ML Pipelines

**Date:** 2026-02-01  
**Context:** Deciding where to store classification labels (VectorIndex vs KNNClassifier)  
**Options Considered:**

1. Add metadata to VectorIndex
2. Parallel label array in KNNClassifier (chosen)
3. DataFrame sidecar

**Decision:** Option 2 - Parallel label array in classifier

**Rationale:**

- VectorIndex = vector operations only (Single Responsibility)
- KNNClassifier = classification logic only
- Parallel array enables composability (use index for clustering, deduplication)
- Simpler testing (test search and classification independently)

**Evidence:** E2E test proved the design works:

```
Query sample 42 (malware):
  Rank 1: idx=42, label=malware, similarity=1.0000
  Rank 2: idx=16, label=malware, similarity=0.1446
  Rank 3: idx=71, label=malware, similarity=0.1398
  Label agreement: 3/5 = majority vote works
```

**Takeaway:** Resist adding "just one more feature." The simplest correct abstraction is usually right.

---

## Template for Future Lessons

```markdown
## Lesson: [Title]

**Date:** YYYY-MM-DD  
**Context:** [What you were doing]  
**Problem:** [What went wrong or what decision was needed]  
**Solution:** [How you fixed it or what you chose]  
**Impact:** [What changed as a result]  
**Takeaway:** [One-sentence principle for future reference]
```

---

*Update this document as you build. These entries become teaching material.*
