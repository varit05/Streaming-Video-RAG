# Project Code Review, Linting & Type Checking Standards

## ✅ Required Tools & Configuration

This project uses modern Python tooling for code quality. All rules are enforced and must pass before code can be merged.

---

## 🔧 Linting Tools

### 1. **Ruff** (Primary Linter)
- **Purpose**: Fast Python linter and formatter replacement
- **Configuration Location**: `pyproject.toml` [tool.ruff]
- **Enabled Rule Sets**:
  - `E` - pycodestyle errors
  - `F` - pyflakes logical errors
  - `I` - import ordering
  - `B` - bugbear best practices
  - `C4` - code complexity
  - `SIM` - simplifications

### 2. **Black** (Code Formatter)
- **Purpose**: Uncompromising code formatting
- **Line Length**: 120 characters
- **Target Python Versions**: 3.10 - 3.14
- **Rule**: No manual formatting - Black decides code style

### 3. **Mypy** (Static Type Checker)
- **Purpose**: Type safety enforcement
- **Mode**: Strict mode enabled
- **Configuration**:
  - ✅ `strict = true`
  - ✅ `python_version = 3.10`
  - ⚠️ `ignore_missing_imports = true` (temporary for external libraries)
  - 🚧 `disallow_untyped_defs = false` (WILL BE ENABLED SOON)

---

## 📋 Mandatory Code Review Checklist

All PRs must satisfy these requirements:

### ✅ Linting Requirements
- [ ] `ruff check .` passes with 0 errors
- [ ] `black --check .` confirms code is formatted
- [ ] No unused imports, variables or dead code
- [ ] Import statements are correctly ordered
- [ ] No commented out code remains

### ✅ Type Checking Requirements
- [ ] `mypy .` passes with 0 errors
- [ ] All function signatures have type annotations
- [ ] Return types are explicitly declared
- [ ] Variables with ambiguous types are annotated
- [ ] No `Any` type usage unless absolutely necessary

### ✅ General Code Standards
- [ ] Functions are < 50 lines
- [ ] No nested logic deeper than 3 levels
- [ ] All public APIs have docstrings
- [ ] Error handling is explicit
- [ ] Logging is used instead of print statements
- [ ] All configuration comes from environment variables

---

## 🚀 CI Pipeline Checks

These commands run automatically on every commit:
```bash
# Format check
black --check .

# Linting
ruff check .

# Type checking
mypy .

# Auto-fix (run locally before commit)
black .
ruff check . --fix
```

---

## 📌 Future Roadmap (Coming Soon)
1. Enable `disallow_untyped_defs = true` for full type coverage
2. Add pre-commit hooks
3. Add test coverage requirements
4. Add security linting with bandit
5. Add complexity checking with radon

---

## 💡 Skill Memorization
These standards are now part of the project's permanent knowledge. All code contributions will be evaluated against these rules.

Last updated: 2026-04-21