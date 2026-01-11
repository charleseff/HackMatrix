# Building HackMatrix

HackMatrix uses Swift Package Manager for all builds:
- **macOS**: Full GUI with SwiftUI/SpriteKit
- **Linux**: Headless CLI only (for ML training)

## Build

```bash
swift build
```

## Run

```bash
# GUI mode (macOS only)
.build/debug/HackMatrix

# Headless mode (for Python ML training)
.build/debug/HackMatrix --headless-cli
```

## Docker (Linux)

```bash
# Build
docker run --rm -v "$(pwd)":/workspace -w /workspace swift:6.0.3-jammy swift build

# Test
docker run --rm -v "$(pwd)":/workspace -w /workspace swift:6.0.3-jammy \
  bash -c 'echo "{\"action\": \"reset\"}" | .build/debug/HackMatrix --headless-cli'
```

## Architecture

The codebase uses conditional compilation (`#if canImport(SpriteKit)`) to include GUI code only on macOS. On Linux, only the headless game logic is compiled.

| Component | macOS | Linux |
|-----------|-------|-------|
| Game logic | ✓ | ✓ |
| Headless CLI | ✓ | ✓ |
| SwiftUI GUI | ✓ | - |
| SpriteKit rendering | ✓ | - |

## Python Integration

Update the executable path in `python/hackmatrix/gym_env.py`:
```python
# macOS (current)
executable = "DerivedData/HackMatrix/Build/Products/Debug/HackMatrix.app/Contents/MacOS/HackMatrix"

# SPM build (new)
executable = ".build/debug/HackMatrix"
```
