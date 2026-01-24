# CI Setup Spec

**Status: Complete**

## Goal

Set up GitHub Actions CI to run all tests (Swift and Python) on every push and pull request.

## Background

The project has two test suites:
- **Swift tests**: Run via `swift test`, test game logic directly
- **Python tests**: Run via `pytest tests/`, test the Python-Swift bridge and environment behavior

Python tests depend on the Swift binary being built first (`swift build` produces the headless CLI).

## Current Test Commands

```bash
# Swift
swift test

# Python (requires Swift binary)
cd python && source venv/bin/activate && pytest tests/ -v
```

## Implementation

### Decisions Made

| Question | Decision | Rationale |
|----------|----------|-----------|
| Swift Version | 5.9 via `swift-actions/setup-swift` | Standard Linux-compatible version |
| Python Version | 3.10 (single version) | Matches `requires-python = ">=3.10"` in pyproject.toml |
| Job Structure | Single job (Option A) | Simpler, sequential execution ~2-3 min total |
| Caching | Pip cache via setup-python | Fast enough for CI needs |
| Dev Container Reuse | No (Option B) | Faster to install dependencies directly |

### Workflow File

Located at `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python 3.10
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
        cache: 'pip'
        cache-dependency-path: python/requirements.txt
    - name: Set up Swift
      uses: swift-actions/setup-swift@v2
      with:
        swift-version: '5.9'
    - name: Install Python dependencies
      run: |
        cd python
        pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run Ruff linter
      run: |
        cd python
        ruff check .
    - name: Run Ruff formatter check
      run: |
        cd python
        ruff format --check .
    - name: Build Swift project
      run: swift build
    - name: Run Python tests
      run: |
        cd python
        pytest tests/ -n auto -v --tb=short
    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v4
      with:
        name: test-results
        path: python/.pytest_cache/
        retention-days: 7
```

## Success Criteria

1. [x] GitHub Actions workflow file created (`.github/workflows/ci.yml`)
2. [x] Swift build runs in CI
3. [x] Python tests run and pass in CI (350 tests)
4. [x] CI runs on push to main and on PRs
5. [x] Caching configured (pip cache)
6. [x] Open questions resolved (see Decisions Made table)
7. [x] Ruff linting integrated into CI

## Related Documentation

- [specs/continuous-integration.md](./continuous-integration.md) - Full CI usage guide
- [specs/testing-and-linting.md](./testing-and-linting.md) - Pre-commit hooks that CI mirrors

## References

- [GitHub Actions Swift setup](https://github.com/swift-actions/setup-swift)
- [Dev Container CI action](https://github.com/devcontainers/ci)
- Current dev container: `.devcontainer/Dockerfile`
