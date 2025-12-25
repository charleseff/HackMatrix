# HackMatrix Game Architecture

HackMatrix is a turn-based tactical roguelike game built with Swift/SpriteKit, featuring reinforcement learning (RL) training capabilities via a Python-Swift bridge.

## Project Structure

```
HackMatrix/
├── HackMatrix/                  # Swift source code
│   ├── App.swift                # Entry point, handles CLI flags
│   ├── GameState.swift          # Core game logic (~2000 lines, single source of truth)
│   ├── GameScene.swift          # SpriteKit rendering, animation, user input
│   ├── Player.swift             # Player entity (position, health, resources)
│   ├── Enemy.swift              # Enemy types (Virus, Daemon, Glitch, Cryptog)
│   ├── Program.swift            # 22 program types with costs and effects
│   ├── Grid.swift               # 6x6 grid, cells, blocks, resources
│   ├── Constants.swift          # Grid size, stage configs, dev mode
│   ├── Pathfinding.swift        # A* pathfinding for enemy AI
│   ├── HeadlessGameCLI.swift    # JSON stdin/stdout protocol handler
│   ├── HeadlessGame.swift       # Game wrapper for headless mode
│   ├── VisualGameController.swift # GUI mode with stdin/stdout control
│   ├── ObservationBuilder.swift # Converts GameState → ML observation
│   ├── Observation.swift        # Observation data structures
│   ├── GameCommandProtocol.swift # Shared command parsing
│   ├── MenuScene.swift          # Main menu UI
│   ├── HighScoreManager.swift   # Score persistence
│   └── Assets.xcassets/         # Sprites and images
├── HackMatrix.xcodeproj/        # Xcode project
├── python/                      # Python ML training
│   ├── hackmatrix/              # Python package
│   │   ├── __init__.py
│   │   └── gym_env.py           # Gymnasium environment wrapper
│   ├── scripts/
│   │   ├── train.py             # MaskablePPO training script
│   │   ├── watch_trained_agent.py
│   │   ├── manual_play.py
│   │   ├── random_test.py
│   │   └── show_observation_space.py
│   ├── requirements.txt
│   ├── setup.py
│   └── README.md
├── plans/                       # Implementation plan documents
├── .vscode/                     # VSCode debug configurations
├── CLAUDE.md                    # This file
├── TRAINING_GUIDE.md            # ML training documentation
└── Package.swift                # Swift package manifest (Xcode preferred)
```

---

## Project Conventions

### Building

- **Always use Xcode build**, NOT `swift build`
- Build command: `xcodebuild -scheme HackMatrix -configuration Debug`
- Output location: `DerivedData/HackMatrix/Build/Products/Debug/HackMatrix.app/Contents/MacOS/HackMatrix`
- Python expects the executable at this location (matches VSCode launch.json)

### Git Workflow

- **Always create a new branch** for any non-trivial work
- Branch naming: descriptive kebab-case (e.g., `reward-system-refactor`, `fix-movement-bug`)
- Workflow:
  1. Create branch: `git checkout -b feature-name`
  2. Make changes and test thoroughly
  3. Review code before committing
  4. Commit with descriptive messages
  5. Merge to main when complete and tested
  6. Delete feature branch after merge

### File Organization

- **Plan files** go in `plans/` directory
- **Swift source** goes in `HackMatrix/` directory
- **Python code** follows package structure in `python/hackmatrix/`
- **Training scripts** go in `python/scripts/`

---

## Game Mechanics

### Core Gameplay

- **Grid**: 6x6 tactical board
- **Stages**: 8 total, progressive difficulty
- **Objective**: Complete all stages by reaching exits, maximize score

### Player Resources

| Resource | Description |
|----------|-------------|
| HP | 3 max, game over at 0 |
| Credits | Currency for program costs |
| Energy | Second currency for program costs |
| Data Siphons | Collectible resource (up to 10) |
| Score | Points from siphoning blocks |

### Turn Structure

**Player's Turn** - one of:
- **Move** → Turn ends, enemy turn begins
- **Attack** → Turn ends, enemy turn begins
- **Siphon** → Turn ends, enemy turn begins
- **Execute Program** → Turn does NOT end (can chain programs)
  - **Exception: Wait program** → Turn ends

**Enemy's Turn**:
1. Transmissions spawn (convert to enemies based on timer)
2. Enemies move/attack
3. Scheduled tasks execute
4. Enemy status resets

### Enemy Types

| Type | HP | Speed | Special |
|------|-----|-------|---------|
| Virus 🦠 | 2 | 2 cells/turn | Fast movement |
| Daemon 👹 | 3 | 1 cell/turn | High HP |
| Glitch ⚡️ | 2 | 1 cell/turn | Can move on blocks |
| Cryptog 👻 | 2 | 1 cell/turn | Invisible unless in same row/col |

### Programs (22 types)

Programs cost credits and/or energy. Examples:
- `PUSH/PULL`: Push/pull enemies 1 cell
- `CRASH`: Clear 8 surrounding cells
- `WARP`: Teleport to random enemy
- `ROW/COL`: Attack all in row/column
- `WAIT`: Skip turn (ends turn)
- `SHOW`: Reveal Cryptogs
- `RESET`: Restore to 3 HP
- See `Program.swift` for full list with costs

---

## Architecture Overview

### Entry Points

| Flag | Purpose | GUI | Execution |
|------|---------|-----|-----------|
| (none) | Human plays game | Yes | Interactive |
| `--headless-cli` | ML training | No | Instant |
| `--visual-cli` | Watch ML play | Yes | Animated |
| `--debug-scenario` | Test specific scenario | Yes | Interactive |
| `--dev-mode` | Developer mode | Yes | Interactive |

### Call Hierarchies

**GUI Mode (Human Player):**
```
App → ContentView → GameScene
  User Input → GameScene.keyDown/mouseDown
    → tryExecuteActionAndAnimate()
      → GameState.tryExecuteAction() [game logic]
      → animateActionResult() [visuals]
```

**Headless CLI Mode (ML Training):**
```
App → HeadlessGameCLI → StdinCommandReader
  Python stdin → executeStep()
    → HeadlessGame.step()
      → GameState.tryExecuteAction() [same logic]
      → ObservationBuilder.build() [state → observation]
```

**Visual CLI Mode (Watch ML):**
```
App → ContentView → GameScene + VisualGameController
  Python stdin → executeStep()
    → GameScene.tryExecuteActionAndAnimate()
      → GameState.tryExecuteAction() [same logic]
      → animateActionResult() [visuals]
      → Wait for animation → return observation
```

### Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| GameState | `GameState.swift` | All game logic (movement, combat, programs, stage gen) - single source of truth |
| GameScene | `GameScene.swift` | Visual rendering, animation, user input handling |
| ObservationBuilder | `ObservationBuilder.swift` | Convert GameState → GameObservation for ML |
| HeadlessGameCLI | `HeadlessGameCLI.swift` | stdin/stdout protocol for headless mode |
| VisualGameController | `VisualGameController.swift` | stdin/stdout protocol for visual mode (syncs with animations) |
| HackEnv | `gym_env.py` | Gymnasium environment wrapper for Python |

---

## Python-Swift Bridge (ML Training)

### Architecture

```
Python (train.py / watch_trained_agent.py)
    └── hackmatrix.HackEnv (Gymnasium environment)
            └── subprocess: HackMatrix --headless-cli
                    └── HeadlessGameCLI.swift (JSON stdin/stdout)
                            └── HeadlessGame.swift (game wrapper)
                                    └── GameState.tryExecuteAction()
```

Both GUI and headless modes use the same `GameState.tryExecuteAction()` core logic.

### JSON Protocol

Python sends one JSON command per line to stdin:
```json
{"action": "reset"}
{"action": "step", "actionIndex": 0}
{"action": "getValidActions"}
```

Swift responds with JSON on stdout:
```json
{"observation": {...}, "reward": 0.0, "done": false, "info": {}}
{"validActions": [0, 2, 4]}
```

### Action Space (28 actions)

| Index | Action |
|-------|--------|
| 0-3 | Move (up, down, left, right) |
| 4 | Siphon |
| 5-27 | Programs (22 total, in ProgramType.allCases order) |

### Observation Space

**Player state** (10 values, normalized 0-1):
```
[row, col, hp, credits, energy, stage, siphons, attack, showActivated, scheduledTasksDisabled]
```

**Program inventory** (26 values): Binary vector of owned programs

**Grid** (6×6×43): Each cell has 43 features:
- Enemy (6): one-hot type + hp + stunned
- Block (5): one-hot type + points + siphoned
- Program (26): one-hot program type
- Transmission (2): spawn count + turns remaining
- Resources (2): credits + energy
- Special (2): is_data_siphon + is_exit

---

## Development Workflow

### Setup

1. **Build Swift app**:
   ```bash
   xcodebuild -scheme HackMatrix -configuration Debug
   ```

2. **Setup Python environment**:
   ```bash
   cd python
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e .  # Install hackmatrix package in editable mode
   ```

### Running

**Play the game (GUI)**:
```bash
./DerivedData/HackMatrix/Build/Products/Debug/HackMatrix.app/Contents/MacOS/HackMatrix
```

**Test Python environment**:
```bash
cd python
python scripts/random_test.py
```

**Train RL agent**:
```bash
cd python
python scripts/train.py --timesteps 1000000
```

**Watch trained agent**:
```bash
cd python
python scripts/watch_trained_agent.py --model ./models/best_model.zip
```

### Testing

- Always rebuild Swift app after changes: `xcodebuild -scheme HackMatrix -configuration Debug`
- Test environment connectivity: `python scripts/random_test.py`
- Debug Swift stderr: check `/tmp/swift_headless.log`

---

## ML Training

### Training Configuration

```python
# Default MaskablePPO settings in train.py:
learning_rate=3e-4
n_steps=2048
batch_size=64
n_epochs=10
gamma=0.99
gae_lambda=0.95
clip_range=0.2
ent_coef=0.1  # High exploration to prevent entropy collapse
```

### Reward Shaping

- Siphoning blocks: +0.005 per block
- Acquiring NEW programs: +0.02 each
- Score collection: +0.1 per point
- Stage completion: +1.0
- Game won (all 8 stages): score × 10.0 + 10
- Death: 0.0

### Monitoring

```bash
tensorboard --logdir python/logs/
```

Key metrics:
- `train/entropy_loss`: Should stay around -1.0 to -1.5 (collapse if → 0)
- `rollout/ep_rew_mean`: Should climb steadily
- `rollout/ep_len_mean`: Longer = surviving more turns

### Commands

```bash
# Fresh training
python scripts/train.py --timesteps 100000000

# Resume from checkpoint
python scripts/train.py --resume ./models/maskable_ppo_TIMESTAMP/best_model.zip

# Parallel environments (faster on multi-core)
python scripts/train.py --num-envs 4 --timesteps 100000000
```

---

## Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `GameState.swift` | ~2000 | Core game logic, movement, combat, programs |
| `GameScene.swift` | ~1000 | SpriteKit rendering, animations, input |
| `Program.swift` | ~140 | 22 program types, costs, effects |
| `Enemy.swift` | ~150 | Enemy types, behavior |
| `gym_env.py` | ~340 | Python Gymnasium wrapper |
| `train.py` | ~230 | MaskablePPO training script |

---

## Debugging

### Debug Scenario Mode

Load a fixed game state for testing:
```bash
./HackMatrix.app/.../HackMatrix --debug-scenario
```

Configure the scenario in `GameState.createDebugScenario()`.

### Developer Mode

Filter available programs:
```bash
./HackMatrix.app/.../HackMatrix --dev-mode --programs=warp,crash,row
```

### Swift Logging

Swift stderr output goes to:
- Headless mode: `/tmp/swift_headless.log`
- Visual mode: `/tmp/swift_visual.log`

Enable verbose logging:
```bash
python scripts/train.py --debug  # Full verbose
python scripts/train.py --info   # Important events only
```
