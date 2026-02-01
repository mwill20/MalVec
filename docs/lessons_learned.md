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
