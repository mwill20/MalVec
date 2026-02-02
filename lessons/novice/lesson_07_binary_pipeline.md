# Lesson 7: Handling Real Malware - The Binary Pipeline

**Phase:** 7
**Topic:** Input Validation & Feature Extraction

## 🎯 fast_track

Use the new native extractor to process real PE binaries:

```bash
# 1. Inspect the validator
cat malvec/validator.py

# 2. Inspect the native extractor
cat malvec/extractor.py

# 3. Classify a real binary (e.g. python.exe)
python -m malvec.cli.classify --model ./model --file $(which python)
```

## 📚 Concepts

1. **Input Validation**: Protecting the pipeline from DoS (50MB limit) and invalid formats.
2. **Static Analysis**: Extracting features without executing the code.
3. **Feature Vectors**: Converting `bytez` -> `float32[2381]`.
4. **Determinism**: Ensuring the same file always yields the same vector.

## 🏗️ Architecture

The pipeline has bridged the gap between synthetic data and real files:

```
Real File -> [InputValidator] -> [FeatureExtractor] -> [Embedder] -> ...
```

- **InputValidator**: Checks `MZ` magic bytes and size < 50MB.
- **FeatureExtractor**: Uses LIEF to parse PE headers, sections, imports, and strings.

## 🛠️ Implementation

**Key Checkpoint:** The `FeatureExtractor` in `malvec/extractor.py` is now a NATIVE implementation, replacing the heavy `ember` library dependency.

## 🎓 Practice

Try classifying different system binaries. Note that predictions will be random if using a synthetic model, but the *process* is valid.

## 🔮 Next Steps

Now that we can process files, we must secure the processing logic (Phase 8).
