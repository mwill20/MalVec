# Lesson 01: Project Foundation

> **Track:** Novice  
> **Phase:** 1 - Project Foundation  
> **Duration:** 45-60 minutes  
> **Prerequisites:** Basic Python knowledge, familiarity with command line

---

## 🎯 Learning Objectives

By the end of this lesson, you will:

1. Understand the MalVec architecture at a high level
2. Set up a professional Python project structure
3. Configure Git for security-sensitive projects
4. Create a virtual environment and manage dependencies
5. Document design decisions from day one

---

## 📚 Key Concepts

### 1. Why Project Structure Matters

**Bad structure:**

```
malware_detector.py  # Everything in one file
samples/             # Malware samples next to code
```

**Good structure:**

```
malvec/
├── malvec/          # Package code
├── tests/           # Separate test structure
├── data/            # Gitignored data directory
└── docs/            # Documentation
```

**Principle:** Structure reflects intent. Separate code, tests, data, and docs from the start.

### 2. Security-First .gitignore

When building security tools, your `.gitignore` is **critical infrastructure**:

```gitignore
# CRITICAL: Never commit samples!
data/samples/
*.exe
*.dll
*.elf

# Never commit secrets
.env
*.pem
credentials/
```

**Why this matters:** One accidental commit of malware to GitHub could:

- Get your repository flagged/banned
- Spread malware to anyone who clones
- Create legal liability

### 3. Virtual Environments

Virtual environments isolate project dependencies:

```bash
# Create isolated environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Now pip install only affects this project
pip install torch
```

**Why not install globally?**

- Different projects need different versions
- Reproducibility requires explicit deps
- Security: limit blast radius

### 4. Documenting Decisions (Lessons Learned)

Create `docs/lessons_learned.md` immediately:

```markdown
## Lesson: [Decision Title]

**Date:** YYYY-MM-DD  
**Context:** What were you trying to do?  
**Problem:** What went wrong or needed deciding?  
**Solution:** What did you choose?  
**Takeaway:** One-sentence principle
```

**Why document decisions?**

- Future you will forget why
- New team members need context
- Decisions become teaching material

---

## 🔧 Hands-On Exercise

### Exercise 1.1: Analyze the MalVec Structure

Open a terminal in the MalVec project and run:

```bash
tree /F malvec
# Or on Windows without tree:
dir /s /b malvec
```

**Questions to answer:**

1. What file initializes the malvec package?
2. What security invariants are documented in `__init__.py`?
3. Why is `data/samples/` in `.gitignore`?

### Exercise 1.2: Create Your Own Security .gitignore

Create a new file called `test_gitignore.txt` and list 10 file types/paths that should NEVER be committed to a malware analysis project.

**Hint categories:**

- Executable formats
- Secrets and credentials
- IDE files
- Large datasets
- Log files

### Exercise 1.3: Read the Decision Log

Open `docs/lessons_learned.md` and answer:

1. What was the first major decision documented?
2. What is the "two-mode architecture"?
3. Why is EMBER-first the chosen approach?

---

## ✅ Checkpoint Questions

Test your understanding:

### Q1: Why separate `tests/unit/`, `tests/integration/`, and `tests/e2e/`?

<details>
<summary>Click for answer</summary>

Different test types have different purposes:

- **Unit tests:** Test individual functions in isolation (fast, many)
- **Integration tests:** Test components working together (medium)
- **E2E tests:** Test full user workflows (slow, few)

Separating them lets you run `pytest tests/unit/` for quick feedback.
</details>

### Q2: What happens if you commit a malware sample to GitHub?

<details>
<summary>Click for answer</summary>

Multiple bad outcomes:

1. GitHub may flag/suspend your repository
2. Anyone who clones gets the malware
3. Automated security scanners alert
4. Legal issues in some jurisdictions
5. Even if you delete, it's in git history forever

**Prevention:** Comprehensive `.gitignore` + never store samples in repo
</details>

### Q3: Why document the EMBER-first decision in `lessons_learned.md`?

<details>
<summary>Click for answer</summary>

1. **Future reference:** Months later, you'll forget why
2. **Onboarding:** New team members need context
3. **Teaching:** These decisions become lesson material
4. **Accountability:** Makes trade-offs explicit

The EMBER-first decision affects the entire build order - it's not obvious without documentation.
</details>

---

## 🎓 Key Takeaways

1. **Structure before code:** Set up proper directories, git, and venv before writing a single line of application code

2. **Security is infrastructure:** `.gitignore` isn't an afterthought - it's a security control

3. **Document decisions immediately:** Don't trust future-you to remember why

4. **Dependencies by phase:** Organize `requirements.txt` by when things are needed, not alphabetically

5. **Test structure ready:** Create test directories even before you have tests

---

## 📖 Further Reading

- [The Hitchhiker's Guide to Python - Structuring Your Project](https://docs.python-guide.org/writing/structure/)
- [GitHub .gitignore templates](https://github.com/github/gitignore)
- [EMBER Dataset Paper](https://arxiv.org/abs/1804.04637)

---

## ➡️ Next Lesson

**Lesson 02: EMBER Dataset Integration** - Load the EMBER dataset, understand feature structure, and prepare for embedding generation.

---

*Lesson created during Phase 1 build. Last updated: 2026-02-01*
