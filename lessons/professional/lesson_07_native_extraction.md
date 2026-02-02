# Lesson 7: Deployment Engineering - Native Feature Extraction

**Phase:** 7
**Topic:** Removal of Helper Libraries & Robustness

## 🎯 fast_track

Review the transition from `ember` library wrapper to native implementation:

```bash
# Compare architectures:
# Old: malvec -> ember -> lightgbm -> lief
# New: malvec -> lief
```

## 📚 Concepts

1. **Dependency Minimization**: Reducing attack surface and build complexity by removing `lightgbm` and `ember`.
2. **Robustness**: Handling malformed PE files (LIEF errors) gracefully without crashing the service.
3. **Feature Parity**: Re-implementing complex feature logic (Entropy, Byte Histograms) to match legacy systems exactly.

## 🏗️ Design Decisions

**Why Native?**
The `ember` python package creates a hard dependency on `lightgbm` (Gradient Boosting framework). For *inference only*, we don't need the training framework. Implementing the feature extractor natively using `lief` allows for a lighter, faster container.

**Error Handling Strategy:**

- WEAK: Crash on bad PE.
- STRONG: Catch `lief.bad_format`, return "empty" features (zeros), flag for review.
- IMPLEMENTED: Robust try/except blocks in `extractor.py` ensuring pipeline continuity.

## 🎓 Practice

Review `tests/unit/test_features.py`. How do we test `sys.executable`? Why is this a better integration test than mocking?

## 🔮 Next Steps

Performance tuning and Security isolation (Phase 8).
