# Testing and Linting Spec

Unified testing and linting infrastructure using `prek` (a faster, Rust-based drop-in replacement for pre-commit).

## Current State

| Category | Status |
|----------|--------|
| Python tests | 350 tests, ~18s parallel / ~30s sequential |
| Swift tests | None (empty `Tests/` directory) |
| Other tests | None (no JS/TS) |
| Python linting | `ruff` available (v0.14.14) |
| Swift linting | Not installed |
| Pre-commit hooks | Not configured |

## Goals

1. Pre-commit hook that blocks commits on failures
2. Fast feedback loop (lint before tests)
3. Easy setup for new developers
4. Runs only on changed files when possible

## Implementation

### 1. Install prek

prek is installed via standalone installer (single binary, no Python dependency):
```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/j178/prek/releases/latest/download/prek-installer.sh | sh
```

pytest-xdist is still needed for parallel tests (in `python/requirements.txt`).

### 2. Pre-commit Configuration

Create `.pre-commit-config.yaml` in repo root:

```yaml
repos:
  # Ruff - Python linting and formatting
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
        files: ^python/
      - id: ruff-format
        files: ^python/

  # Local hooks for Swift and tests
  - repo: local
    hooks:
      # Swift build (catches compile errors)
      - id: swift-build
        name: Swift Build
        entry: swift build
        language: system
        pass_filenames: false
        files: \.(swift)$

      # Python tests (parallel)
      - id: pytest
        name: Python Tests
        entry: bash -c 'cd python && source venv/bin/activate 2>/dev/null || source venv-arm64/bin/activate && pytest tests/ -n auto -q --tb=short'
        language: system
        pass_filenames: false
        always_run: true
        stages: [pre-commit]
```

### 3. Ruff Configuration

Add to `python/pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # Pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
]
ignore = [
    "E501",   # line too long (handled by formatter)
    "B008",   # function call in default argument
]

[tool.ruff.lint.isort]
known-first-party = ["hackmatrix"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

### 4. Swift Linting (Future)

Swift linting tools (`swiftlint`, `swiftformat`) are not available in the Linux dev container. Options:

1. **Skip Swift linting in container** - The `swift build` step catches compile errors
2. **Add swiftlint to macOS workflow** - For local dev on Mac
3. **Use swift-format** - Part of Swift toolchain (may be available)

For now, `swift build` serves as the Swift validation step.

## Setup

**One-time setup after clone:**
```bash
# Install prek (single binary, no Python needed)
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/j178/prek/releases/latest/download/prek-installer.sh | sh

# Install hooks
prek install
```

This creates `.git/hooks/pre-commit` automatically.

**For dev container**, prek is automatically installed in `postCreateCommand`.

## Usage

```bash
# Run all hooks manually (same as pre-commit runs)
prek run --all-files

# Run specific hook
prek run ruff --all-files
prek run pytest --all-files

# Fix Python formatting
ruff format python/

# Fix Python lint issues (auto-fixable)
ruff check --fix python/

# Skip hooks in emergency (not recommended)
git commit --no-verify
```

## Timing Breakdown

| Check | Time |
|-------|------|
| Python linting (ruff) | ~1s |
| Python format check | ~1s |
| Swift build | ~2s (cached), ~30s (clean) |
| Python tests (parallel, `-n auto`) | ~18s |
| **Total** | **~22s** (with cached Swift build) |

### Parallel Test Notes

Tests run in parallel using `pytest-xdist` with `-n auto` (uses all CPU cores). This is safe because:
- Each `SwiftEnvWrapper` spawns its own subprocess (no shared state)
- JAX environment uses pure functions (no mutable state)
- Test fixtures create fresh env instances per test

## File Structure

```
hack-matrix/
├── .pre-commit-config.yaml     # Hook configuration (version-controlled, compatible with prek)
└── python/
    ├── pyproject.toml          # Ruff config
    └── requirements.txt        # pytest-xdist for parallel tests
```

## Notes

- `prek install` must be run once after clone (creates `.git/hooks/pre-commit`)
- Hooks run automatically on `git commit`
- Use `git commit --no-verify` to bypass in emergencies (not recommended)
- Ruff hooks auto-fix issues; if they modify files, the commit is blocked so you can review
- Swift tests don't exist yet - when added, include them as a local hook
- `pytest-xdist` enables parallel test execution (`-n auto`)
- prek is a faster, Rust-based drop-in replacement for pre-commit (https://github.com/j178/prek)
