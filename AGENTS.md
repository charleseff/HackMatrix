## Build & Run

HackMatrix uses a hybrid build approach:
- **SPM** (`swift build`): Headless CLI for ML training (macOS + Linux)
- **Xcode** (`xcodebuild`): Full GUI app (macOS only)

```bash
# Training (headless CLI)
swift build
.build/debug/HackMatrix --headless-cli

# GUI app (macOS)
xcodebuild -scheme HackMatrix -configuration Debug build
```

## Validation

Run these after implementing to get immediate feedback:

- Tests: `swift test`
- Build (headless): `swift build`
- Build (GUI): `xcodebuild -scheme HackMatrix -configuration Debug build`

## Operational Notes

- SPM excludes `App.swift` and GUI code via conditional compilation
- Xcode includes everything and produces a proper .app bundle for macOS GUI
- Python integration at `python/` uses venv

### Codebase Patterns

- Game logic shared between SPM and Xcode builds
- Source code in `HackMatrix/` and `Sources/`
- Tests in `Tests/HackMatrixTests/`
