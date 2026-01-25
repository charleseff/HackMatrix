# GitHub Actions Workflows

This directory contains GitHub Actions CI/CD workflows for the HackMatrix project.

## Active Workflows

### CI Workflow (`ci.yml`)

**Purpose**: Automated testing and linting on every push and pull request

**Triggers**:
- Push to `main` branch
- Pull requests targeting `main` branch

**Steps**:
1. Setup Python 3.10 and Swift 5.9
2. Install dependencies from `python/requirements.txt`
3. Run Ruff linter (Python code style)
4. Run Ruff formatter check (Python formatting)
5. Build Swift project (headless binary)
6. Run 350+ Python tests in parallel
7. Upload test results as artifacts

**Expected runtime**: ~2-3 minutes

**Mirrors**: Prek hooks in `.pre-commit-config.yaml`

## Development Workflow

**To avoid CI failures**, install prek hooks locally:

```bash
# Install prek (single binary, no Python needed)
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/j178/prek/releases/latest/download/prek-installer.sh | sh
prek install
```

This runs the same checks locally before you commit.

## Debugging Failed CI

1. Check the Actions tab for detailed logs
2. Run `prek run --all-files` locally to reproduce
3. Fix issues and push again

## Documentation

See `/workspaces/hack-matrix/specs/continuous-integration.md` for complete details.
