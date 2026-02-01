# MalVec - Anti Gravity Claude Knowledge Base

## 🎯 Project Context

You're building **MalVec**: a malware detection system using embedding-space analysis to identify polymorphic variants that evade signature-based detection.

**Your Mission:**
- Build a secure, production-ready detection tool
- Create clean code that enables teaching (dual-track lessons later)
- Document decisions as you make them
- Test before implementing

---

## 🔑 Critical Constraints (Always Enforce)

### Security Invariants (NON-NEGOTIABLE)

1. **Malware NEVER executes** - Static analysis only
2. **File paths NEVER in output** - Prevent info disclosure
3. **All inputs validated** - At system boundaries
4. **Sandboxing enforced** - Timeouts, no network
5. **Fail safely** - Manual review > auto-classify on errors

### Build Philosophy

- **Security first** - Every boundary validated before functionality
- **Test first** - Failing test → implementation → passing test
- **Document first** - Architecture decisions recorded in real-time
- **Teach later** - Code structure enables lessons (Phase 10)

---

## 📚 Key Documents (Read in Order)

1. **PROJECT_BRIEF.md** - Start here (context, problem, solution)
2. **NORTHSTAR.md** - Full architectural blueprint
3. **BUILD_INSTRUCTIONS.md** - Step-by-step implementation guide
4. **LESSON_PLAN.md** - What we'll teach (shapes implementation)

---

## 🏗️ Architecture Summary

### Data Flow
```
Binary → Validate → Extract Features → Generate Embedding → Vector DB → K-NN → Classification
   ↓         ↓              ↓                  ↓              ↓        ↓          ↓
[Untrust] [Layer1]     [Sandboxed]        [Isolated]     [FAISS]  [Voting]  [Result]
```

### Components (Build in This Order)

**Phase 2: Security Layer (Priority 1)**
- `validator.py` - Input validation (Layer 1 defense)
- `utils.py` - Sandbox context manager (Layer 2 defense)

**Phase 3-6: Core Components**
- `extractor.py` - Feature extraction (pefile, LIEF)
- `embedder.py` - Embedding generation (sentence-transformers)
- `store.py` - Vector storage (FAISS + SQLite)
- `classifier.py` - K-NN classification

**Phase 7: CLI**
- `scripts/train.py` - Build vector DB
- `scripts/classify.py` - Classify sample
- `scripts/visualize.py` - Cluster plots

---

## 🛡️ Defense-in-Depth Layers

Every component implements multiple security layers:

**Layer 1: Input Validation**
- File type whitelist (PE, ELF only)
- Size limits (100MB max)
- Magic byte verification

**Layer 2: Sandboxing**
- Subprocess execution
- 30-second timeout
- No network access
- Resource limits

**Layer 3: Model Isolation**
- Separate process for embedding model
- Checksum verification
- Output validation (NaN check)

**Layer 4: Output Filtering**
- Hash file paths (never raw paths)
- Sanitize error messages
- Structured data only

**Layer 5: Monitoring**
- Log all processing (hash, timestamp, result)
- Anomaly detection on times
- Daily accuracy reports

**Layer 6: Fail-Safe**
- On error → flag for manual review
- On timeout → mark analysis_failed
- On low confidence → require human approval

---

## 🧪 Testing Requirements

### Test-First Approach (ALWAYS)

```python
# Step 1: Write failing test
def test_reject_oversized_file():
    """Files >100MB should be rejected"""
    large_file = create_file(size_mb=150)
    with pytest.raises(ValidationError):
        validator.validate(large_file)

# Step 2: Implement until test passes
class SampleValidator:
    MAX_SIZE = 100 * 1024 * 1024
    
    def validate(self, file_path):
        if file_path.stat().st_size > self.MAX_SIZE:
            raise ValidationError("File exceeds size limit")

# Step 3: Verify test passes
# Step 4: Refactor if needed
```

### Coverage Targets

- **Unit tests:** >80% coverage
- **Integration tests:** All component interactions
- **E2E tests:** Full pipeline validation
- **Adversarial tests:** Security boundary validation

### Critical Test Cases

**Must Have:**
- Malware never executes (process monitor verification)
- Timeout on obfuscated files
- Corrupt PE doesn't crash
- NaN embeddings rejected
- SQL injection prevented
- Low confidence flagged

---

## 📖 Documentation Requirements

### As You Build

**ARCHITECTURE.md Updates:**
- Design decisions (why this approach?)
- Trade-offs accepted
- Performance benchmarks (actual numbers)

**lessons_learned.md Entries:**
```markdown
## Lesson: [Title]
**Date:** YYYY-MM-DD
**Context:** [What you were doing]
**Problem:** [What went wrong]
**Solution:** [How you fixed it]
**Impact:** [What changed]
**Takeaway:** [One-sentence principle]
```

**API.md:**
- All public classes
- Type signatures
- Example usage
- Error codes

---

## 🎓 Teaching Enablement

### Code Should Be:

**Demonstrable:**
- Clear component boundaries
- Single responsibility per class
- Observable metrics

**Explainable:**
- Self-documenting structure
- Meaningful variable names
- Inline comments for "why" not "what"

**Modifiable:**
- Easy to extend
- Safe to refactor (tests!)
- Patterns over clever tricks

### Example: Good vs Bad

**❌ Bad (clever but opaque):**
```python
def p(f):
    return [x for x in m(f) if v(x)]
```

**✅ Good (clear and teachable):**
```python
def extract_pe_imports(pe_file: Path) -> List[str]:
    """Extract import table from PE file.
    
    Returns only valid import names (non-empty, alphanumeric).
    Used for embedding generation in Lesson 06.
    """
    imports = parse_import_table(pe_file)
    valid_imports = [imp for imp in imports if is_valid_import_name(imp)]
    return valid_imports
```

---

## 🚀 Build Sequence Quick Reference

1. **Setup** - Repository structure, dependencies
2. **Security** - Validator, sandbox
3. **Extraction** - Features from binaries
4. **Embedding** - Vector generation
5. **Storage** - FAISS + SQLite
6. **Classification** - K-NN voting
7. **CLI** - Scripts for train/classify/visualize
8. **Testing** - Unit, integration, E2E, adversarial
9. **Documentation** - README, ARCHITECTURE, API, lessons_learned
10. **Lessons** - Generate dual-track curriculum (uses templates)

---

## 💡 Key Reminders

**When Adding Features:**
1. Read relevant section of NORTHSTAR.md
2. Write test first
3. Implement with security checks
4. Document decision in lessons_learned.md
5. Update ARCHITECTURE.md if design changes

**When Stuck:**
1. Check BUILD_INSTRUCTIONS.md for guidance
2. Review NORTHSTAR.md for architectural context
3. Look at LESSON_PLAN.md for what needs to be demonstrable
4. Ask clarifying questions (don't guess on security)

**When Tempted to Take Shortcuts:**
- Remember: A bug that lets malware execute is catastrophic
- Speed doesn't matter if it's insecure
- Tests save time in the long run
- Documentation is for future-you (and learners)

---

## 🎯 Success Criteria

**You're done when:**
- [ ] All tests pass (>80% coverage)
- [ ] No malware executions (verified)
- [ ] Documentation complete
- [ ] Benchmarks meet targets (1000 samples <10 min)
- [ ] Security review passes (all 6 layers validated)
- [ ] Code enables teaching (clear boundaries, good names)

**Then and only then:** Generate lessons using pedagogy templates.

---

## 🔗 Quick Links

- **Architecture Details:** See NORTHSTAR.md
- **Build Steps:** See BUILD_INSTRUCTIONS.md  
- **Learning Goals:** See LESSON_PLAN.md
- **Context:** See PROJECT_BRIEF.md

---

**Remember:** You're not just writing code - you're creating a learning platform that happens to detect malware. Every design decision should enable both functionality AND pedagogy.

**When in doubt, be conservative. Security first, always.**
