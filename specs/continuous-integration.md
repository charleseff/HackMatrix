# Continuous Integration (CI) Spec

GitHub Actions CI configuration that mirrors the pre-commit hooks for automated testing on push and pull requests.

## Status

**Complete** - CI workflow configured and ready to use.

## Overview

The CI pipeline runs the same checks as the pre-commit hooks to ensure code quality and test coverage before merging changes to the main branch.

## CI Workflow

Located at `.github/workflows/ci.yml`

### Triggers

- **Push to main branch**: Runs on all commits to main
- **Pull requests to main**: Runs on all PRs targeting main

### Jobs

Single job `lint-and-test` that runs on Ubuntu Linux (matches dev container environment):

1. **Setup Python 3.10**: Install Python with pip caching
2. **Setup Swift 5.9**: Install Swift toolchain
3. **Install Python dependencies**: Install from `python/requirements.txt`
4. **Run Ruff linter**: Check Python code style and potential bugs
5. **Run Ruff formatter check**: Ensure Python code is properly formatted
6. **Build Swift project**: Validate Swift code compiles
7. **Run Python tests**: Execute 350 tests in parallel using pytest-xdist
8. **Upload test results**: Archive pytest cache as artifact (7-day retention)

### Execution Time

Expected total runtime: **~2-3 minutes** (on GitHub runners)

| Step | Time |
|------|------|
| Setup (Python + Swift) | ~30-45s |
| Install dependencies | ~30s (with cache) |
| Ruff linting | ~2s |
| Ruff format check | ~2s |
| Swift build | ~10-15s |
| Python tests (parallel) | ~20-30s |

## Comparison with Pre-commit Hooks

The CI workflow mirrors the pre-commit configuration in `.pre-commit-config.yaml`:

| Check | Pre-commit Hook | CI Workflow | Notes |
|-------|----------------|-------------|-------|
| Ruff linter | `ruff --fix` | `ruff check .` | CI uses check-only (no auto-fix) |
| Ruff format | `ruff-format` | `ruff format --check .` | CI uses check-only |
| Swift build | `swift build` | `swift build` | Same command |
| Python tests | `pytest -n auto -q --tb=short` | `pytest -n auto -v --tb=short` | CI uses verbose mode |

### Why CI doesn't auto-fix

Pre-commit hooks can auto-fix issues (e.g., `ruff --fix`) because they run locally and can modify files before commit. CI runs in read-only mode and should only report issues, not modify code.

## Usage

### For Contributors

CI runs automatically on:
1. Every push to main
2. Every pull request

**No manual action required** - GitHub will show check status on PRs.

### Viewing Results

- **PR page**: Check status appears next to each commit
- **Actions tab**: Full logs and test artifacts available at `https://github.com/charleseff/hack-matrix/actions`
- **Artifacts**: Test results are uploaded and available for 7 days

### Handling Failures

If CI fails on your PR:

1. **Check the logs** in the Actions tab to see which step failed
2. **Run pre-commit hooks locally** to catch issues before pushing:
   ```bash
   pre-commit run --all-files
   ```
3. **Fix the issues** and push again
4. **CI will re-run automatically** on the new push

### Common Failure Scenarios

| Failure | Cause | Fix |
|---------|-------|-----|
| Ruff linter | Code style issues | Run `ruff check --fix python/` locally |
| Ruff format | Formatting issues | Run `ruff format python/` locally |
| Swift build | Compilation errors | Run `swift build` locally to see errors |
| Python tests | Test failures | Run `pytest tests/ -n auto -v` locally to debug |

## Local Development Workflow

**Recommended workflow** to avoid CI failures:

1. **Install pre-commit hooks** (one-time setup):
   ```bash
   cd python && source venv/bin/activate
   pip install pre-commit
   pre-commit install
   ```

2. **Develop normally** - hooks run automatically on `git commit`

3. **If hooks fail**, fix issues and commit again

4. **Push to GitHub** - CI will pass if pre-commit passed

### Manual Pre-commit Checks

Run all hooks without committing:
```bash
pre-commit run --all-files
```

Run specific hook:
```bash
pre-commit run ruff --all-files
pre-commit run pytest --all-files
```

## CI Configuration Details

### Python Version

**3.10** - Matches the `requires-python = ">=3.10"` in `python/pyproject.toml`

### Swift Version

**5.9** - Standard version for Linux compatibility and dev container

### Caching

- **Pip cache**: Speeds up dependency installation (~30s vs ~2-3min)
- **Cache key**: Based on `python/requirements.txt` content
- **Auto-invalidation**: Cache refreshes when requirements.txt changes

### Artifacts

Test results (`.pytest_cache/`) are uploaded as artifacts:
- **Retention**: 7 days
- **Size**: Minimal (~KB)
- **Purpose**: Debugging test failures in CI

## Future Enhancements

Potential additions (not currently implemented):

1. **Coverage reporting**: Add pytest-cov and upload coverage to Codecov
2. **Swift tests**: Add when Swift tests are implemented
3. **Matrix testing**: Test multiple Python versions (3.10, 3.11, 3.12)
4. **macOS runner**: Test GUI builds on macOS (currently Linux-only)
5. **Release automation**: Auto-publish on version tags
6. **Dependabot**: Auto-update dependencies

## Files

```
hack-matrix/
├── .github/
│   └── workflows/
│       └── ci.yml                  # CI workflow configuration
├── .pre-commit-config.yaml         # Pre-commit hooks (mirrors CI)
└── specs/
    ├── testing-and-linting.md      # Testing framework spec
    └── continuous-integration.md   # This file
```

## Notes

- CI runs on **Ubuntu Linux**, matching the dev container environment
- **No macOS runner** currently configured (Swift builds on Linux only)
- **GUI builds** are not tested in CI (requires macOS with Xcode)
- **Headless Swift builds** are validated (used for ML training)
- CI uses the same tools and versions as the pre-commit hooks for consistency
