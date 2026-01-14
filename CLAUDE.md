# HackMatrix

A roguelike strategy game with reinforcement learning training support. Players navigate an 8-stage dungeon, collecting programs, defeating enemies, and optimizing score.

## Quick Reference

```bash
# Build for training (headless)
swift build

# Build GUI app (macOS only)
xcodebuild -scheme HackMatrix -configuration Debug build

# Run Python training
cd python && source venv/bin/activate && python scripts/train.py

# Test environment
cd python && source venv/bin/activate && python scripts/random_test.py
```

---

## Project Structure

```
HackMatrix/
├── HackMatrix/              # Swift source files
│   ├── App.swift            # GUI entry point (@main, SwiftUI)
│   ├── GameState.swift      # Core game logic (single source of truth)
│   ├── GameScene.swift      # SpriteKit rendering and animations
│   ├── HeadlessGameCLI.swift   # JSON protocol handler for ML training
│   ├── HeadlessGame.swift   # Game wrapper for headless mode
│   ├── ObservationBuilder.swift # GameState → Observation for ML
│   ├── RewardCalculator.swift   # RL reward calculation
│   ├── Program.swift        # 23 program types and costs
│   ├── Enemy.swift          # 4 enemy types (Virus, Daemon, Glitch, Cryptog)
│   ├── Grid.swift           # 6x6 game grid
│   ├── Player.swift         # Player state
│   └── Constants.swift      # Game constants
├── Sources/SPMMain/         # SPM entry point (headless only)
├── python/
│   ├── hackmatrix/          # Python package
│   │   ├── gym_env.py       # Gymnasium environment wrapper
│   │   ├── training_config.py  # MaskablePPO hyperparameters
│   │   └── training_db.py   # SQLite training history
│   └── scripts/
│       ├── train.py         # Main training script (W&B integration)
│       ├── random_test.py   # Random action test
│       ├── manual_play.py   # Visual mode for watching agent
│       └── watch_trained_agent.py  # Load and watch trained model
├── plans/                   # Implementation plans
├── docs/                    # Documentation and investigations
├── Package.swift            # SPM configuration
└── HackMatrix.xcodeproj/    # Xcode project
```

---

## Project Conventions

### Python Scripts
- **Always activate venv** before running Python scripts
- Command pattern: `cd python && source venv/bin/activate && python <script>`

### Building

**Hybrid build approach:**
| Build | Command | Output | Use Case |
|-------|---------|--------|----------|
| SPM | `swift build` | `.build/debug/HackMatrix` | Training (headless) |
| Xcode | `xcodebuild -scheme HackMatrix -configuration Debug build` | `DerivedData/.../HackMatrix.app` | GUI app |

Python automatically selects the correct binary based on `visual` parameter.

### Git Workflow
- Create branches for non-trivial work: `git checkout -b feature-name`
- Branch naming: descriptive kebab-case (e.g., `reward-system-refactor`)
- Plan files go in `plans/` directory

---

## Game Mechanics

### Overview
- **Grid**: 6×6 cells
- **Stages**: 8 total (complete all to win)
- **Resources**: Credits, Energy, Data Siphons
- **Starting bonus**: Random (10 credits OR 11 energy OR 1 data siphon)

### Turn Structure

**Player's Turn:**
- **Move** → Turn ends
- **Attack** → Turn ends
- **Siphon** → Turn ends
- **Execute Program** → Turn does NOT end (can chain)
  - Exception: **Wait** program ends turn

**Enemy's Turn (after player's turn ends):**
1. Transmissions spawn (convert to enemies based on timer)
2. Enemies move/attack
3. Scheduled tasks execute
4. Enemy status resets

### Enemy Types

| Type | HP | Speed | Special |
|------|-----|-------|---------|
| Virus 🦠 | 2 | 2 cells/turn | Fast movement |
| Daemon 👹 | 3 | 1 | High HP |
| Glitch ⚡️ | 2 | 1 | Can move on blocks |
| Cryptog 👻 | 2 | 1 | Invisible (unless in same row/col or SHOW used) |

### Programs (23 total)

Programs are acquired by siphoning program blocks. Each has credit/energy cost.

| Index | Program | Cost (C/E) | Effect |
|-------|---------|-----------|--------|
| 5 | PUSH | 0/2 | Push enemies away 1 cell |
| 6 | PULL | 0/2 | Pull enemies toward 1 cell |
| 7 | CRASH | 3/2 | Clear 8 surrounding cells |
| 8 | WARP | 2/2 | Warp to random enemy |
| 9 | POLY | 1/1 | Randomize enemy types |
| 10 | WAIT | 0/1 | Skip turn (ends turn) |
| 11 | DEBUG | 3/0 | Damage enemies on blocks |
| 12 | ROW | 3/1 | Attack all in row |
| 13 | COL | 3/1 | Attack all in column |
| 14 | UNDO | 1/0 | Undo last turn |
| 15 | STEP | 0/3 | Enemies skip next turn |
| 16 | SIPH+ | 5/0 | Gain 1 data siphon |
| 17 | EXCH | 4/0 | Convert 4C to 4E |
| 18 | SHOW | 2/0 | Reveal Cryptogs |
| 19 | RESET | 0/4 | Restore to 3HP |
| 20 | CALM | 2/4 | Disable scheduled spawns |
| 21 | D_BOM | 3/0 | Destroy nearest Daemon |
| 22 | DELAY | 1/2 | Extend transmissions +3 turns |
| 23 | ANTI-V | 3/0 | Damage all Viruses |
| 24 | SCORE | 0/5 | Gain points = stages left |
| 25 | REDUC | 2/1 | Reduce block spawn counts |
| 26 | ATK+ | 4/4 | Increase damage to 2HP |
| 27 | HACK | 2/2 | Hack nearby enemies |

---

## Architecture

### Entry Points

| Flag | Purpose | GUI | Mode |
|------|---------|-----|------|
| (none) | Human plays game | Yes | Interactive |
| `--headless-cli` | ML training | No | Instant |
| `--visual-cli` | Watch ML play | Yes | Animated |
| `--debug-scenario` | Test specific scenario | Yes | Interactive |
| `--run-tests` | Run game logic tests | No | Test |
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

| Component | Responsibility |
|-----------|----------------|
| **GameState** | All game logic (single source of truth) |
| **GameScene** | Visual rendering, animations, input handling |
| **ObservationBuilder** | GameState → GameObservation for ML |
| **RewardCalculator** | Calculate RL rewards from state changes |
| **HeadlessGameCLI** | JSON stdin/stdout protocol |
| **StdinCommandReader** | Parse JSON commands, encode responses |

---

## Python-Swift Bridge (ML Training)

### Architecture

```
python/scripts/train.py
    └── hackmatrix/gym_env.py (Gymnasium environment)
            └── subprocess: HackMatrix --headless-cli
                    └── HeadlessGameCLI.swift (JSON stdin/stdout)
                            └── HeadlessGame.swift
                                    └── GameState.tryExecuteAction()
```

### JSON Protocol

**Commands (Python → Swift):**
```json
{"action": "reset"}
{"action": "step", "actionIndex": 0}
{"action": "getValidActions"}
```

**Responses (Swift → Python):**
```json
{"observation": {...}, "reward": 0.0, "done": false, "info": {}}
{"validActions": [0, 2, 4]}
```

### Action Space (28 actions)

| Index | Action |
|-------|--------|
| 0-3 | Move (up, down, left, right) |
| 4 | Siphon |
| 5-27 | Programs (23 total, in ProgramType.allCases order) |

### Observation Space

**Player state** (10 normalized values):
`[row, col, hp, credits, energy, stage, siphons, attack, showActivated, scheduledTasksDisabled]`

**Programs** (23 binary values): Which programs are owned

**Grid** (6×6×40 features per cell):
- Enemy: one-hot type (4) + hp + stunned = 6
- Block: one-hot type (3) + points + siphoned = 5
- Program: one-hot (23) + transmission spawn + turns = 25
- Resources: credits + energy = 2
- Special: is_data_siphon + is_exit = 2

---

## Reinforcement Learning

### Training Configuration

Located in `python/hackmatrix/training_config.py`:

```python
MODEL_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 4096,
    "batch_size": 64,
    "n_epochs": 20,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.3,        # High exploration to prevent entropy collapse
    "vf_coef": 1.0,
    "policy_kwargs": {
        "net_arch": {
            "pi": [256, 256, 128],
            "vf": [256, 256, 128],
        }
    },
}
```

### Reward Structure

**Stage Completion (exponential):**
| Stage | Reward |
|-------|--------|
| 1 | 1.0 |
| 2 | 2.0 |
| 3 | 4.0 |
| 4 | 8.0 |
| 5 | 16.0 |
| 6 | 32.0 |
| 7 | 64.0 |
| 8 | 100.0 |

**Other Rewards:**
- Score gain: `+0.5 × points`
- Enemy kills: `+0.3 per kill` (excludes scheduled task spawns)
- Data siphon collected: `+1.0`
- Distance to exit (closer): `+0.05 × delta`
- Victory: `+500 + score × 100`
- Resource gain: `+0.05 × credits/energy`
- HP recovery: `+1.0 per HP`

**Penalties:**
- Death: `-50%` of cumulative stage rewards
- Damage taken: `-1.0 per HP`
- Suboptimal siphon: `-0.5 × missed value`
- RESET at 2 HP: `-0.3`
- Death from siphon-spawned enemy: `-10.0`

### Training Commands

```bash
# Start training (fresh)
cd python && source venv/bin/activate
python scripts/train.py --timesteps 100000000

# Resume from checkpoint
python scripts/train.py --resume ./models/best_model.zip

# Monitor with TensorBoard
tensorboard --logdir logs/

# Watch trained agent
python scripts/watch_trained_agent.py
```

### Key Metrics to Monitor

| Metric | Healthy Range | Warning Sign |
|--------|---------------|--------------|
| `train/entropy_loss` | -1.0 to -1.5 | → 0 (collapse!) |
| `rollout/ep_rew_mean` | Climbing | Flatline >2M steps |
| `rollout/ep_len_mean` | Increasing | Decreasing |
| `train/approx_kl` | > 0 | = 0 (no updates) |

### Troubleshooting

**Entropy Collapsed:**
- Stop training immediately
- Increase `ent_coef` (try 0.15-0.2)
- Start fresh (don't resume from collapsed model)

**Training Too Slow:**
- Check FPS (should be >500)
- If CPU-bound, Swift subprocess is bottleneck
- If GPU available, add `device='cuda'` or `device='mps'`

---

## Testing

### Swift Tests
```bash
# Run game logic tests
.build/debug/HackMatrix --run-tests

# Or via Xcode build
DerivedData/.../HackMatrix.app/Contents/MacOS/HackMatrix --run-tests
```

### Python Tests
```bash
cd python && source venv/bin/activate

# Basic environment test
python scripts/random_test.py

# Validate observations match spec
python scripts/validate_observations.py

# Comprehensive observation validation
python scripts/validate_observations_comprehensive.py
```

---

## VS Code Integration

### Tasks (Cmd+Shift+B)
- **Build (SPM - Headless)**: Default build task
- **Build (Xcode - GUI)**: Build GUI app
- **Run GUI**: Build and launch GUI

### Launch Configurations (F5)
- **Debug Headless CLI (SPM)**: Debug headless mode
- **Debug GUI (Xcode)**: Debug full GUI
- **Debug GUI - Visual CLI**: Debug visual CLI mode
- **Debug GUI - Scenario**: Debug with `--debug-scenario`

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `HackMatrix/GameState.swift` | Core game logic, action execution |
| `HackMatrix/RewardCalculator.swift` | RL reward calculation |
| `HackMatrix/ObservationBuilder.swift` | State → ML observation |
| `HackMatrix/HeadlessGameCLI.swift` | JSON protocol for Python |
| `python/hackmatrix/gym_env.py` | Gymnasium environment |
| `python/hackmatrix/training_config.py` | PPO hyperparameters |
| `python/scripts/train.py` | Main training script |

---

## Dependencies

### Swift
- macOS 14+ (for GUI)
- SwiftUI, SpriteKit (GUI only)
- Foundation (all platforms)

### Python
```
gymnasium>=0.29.0
numpy>=1.24.0
stable-baselines3>=2.0.0
sb3-contrib>=2.0.0
torch>=2.0.0
tensorboard>=2.14.0
wandb>=0.15.0
```

---

## Linux/Docker Support

The headless CLI builds and runs on Linux for server-side training:

```bash
# Build in Docker
docker run --rm -v "$(pwd)":/workspace -w /workspace swift:6.0.3-jammy swift build

# Test
docker run --rm -v "$(pwd)":/workspace -w /workspace swift:6.0.3-jammy \
  bash -c 'echo "{\"action\": \"reset\"}" | .build/debug/HackMatrix'
```

SPM uses conditional compilation (`#if canImport(SwiftUI)`) to exclude GUI code on Linux.
