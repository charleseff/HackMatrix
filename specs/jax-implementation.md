# JAX Environment Implementation Spec

## Goal

Port the HackMatrix game environment to JAX to enable TPU-accelerated training via Google TPU Research Cloud.
NOTE: THIS IS NOT READY TO BE IMPLEMENTED YET. DO NOT IMPLEMENT THIS.

## Background

### Current Architecture

```
┌─────────────────┐      JSON/stdin       ┌─────────────────┐
│  Python (SB3)   │ ◄──────────────────► │   Swift Game    │
│  MaskablePPO    │                       │   (CPU)         │
│  PyTorch        │                       │                 │
└─────────────────┘                       └─────────────────┘
```

- **RL Library**: Stable Baselines 3 (sb3-contrib) with MaskablePPO
- **Framework**: PyTorch
- **Environment**: Swift binary communicating via subprocess stdin/stdout JSON protocol
- **Parallelization**: Multiple subprocess instances (~8-32 practical limit)

### Why TPUs?

With larger neural networks, compute becomes the bottleneck (not environment stepping). Google TPU Research Cloud provides access to TPUs, but the current stack cannot utilize them.

### Why Current Stack Won't Work on TPUs

| Library | TPU Support | Action Masking | Status |
|---------|-------------|----------------|--------|
| SB3 + torch_xla | Attempted, failed | Yes | Maintainers closed feature request |
| SBX (JAX) | Yes | No | Would need custom action masking |
| PureJaxRL | Yes | Unknown | Requires JAX-native environment |

**Key insight**: PureJaxRL offers the best TPU performance by JIT-compiling the entire training loop (including environment), but requires the environment itself to be written in JAX.

## Target Architecture

```
┌─────────────────────────────────────────────────────┐
│                    TPU / GPU                        │
│  ┌─────────────────┐    ┌────────────────────────┐  │
│  │  HackMatrix Env │───►│   PPO Policy Network   │  │
│  │     (JAX)       │◄───│        (JAX)           │  │
│  └─────────────────┘    └────────────────────────┘  │
│            (entire loop JIT-compiled)               │
└─────────────────────────────────────────────────────┘
```

**Benefits**:
- Run 1000+ environments in parallel on a single TPU
- Entire training loop JIT-compiled (no Python overhead)
- Native TPU support via JAX

## Implementation Plan

### Phase 1: SBX + Action Masking (Quick Win)

Get something running on TPU quickly while developing the full JAX env.

1. Fork or extend SBX's PPO implementation
2. Add action masking (~20 lines: mask logits before softmax)
3. Keep existing Swift subprocess environment
4. Validate TPU training works

### Phase 2: JAX Environment Port

Rewrite the Swift game logic in JAX.

#### Game Components to Port

| Component | Swift Location | JAX Difficulty | Notes |
|-----------|----------------|----------------|-------|
| Grid (6x6) | `Grid.swift` | Easy | Fixed-size `jnp.array` |
| Player state | `Player.swift` | Easy | Named tuple or dataclass |
| Enemy types (4) | `Enemy.swift` | Medium | One-hot encoding + stats array |
| Enemy list | `GameState.enemies` | Medium | Fixed-size array with mask |
| Transmissions | `GameState.transmissions` | Medium | Fixed-size array with mask |
| Programs (23) | `Program.swift` | Hard | `jax.lax.switch` for effects |
| Pathfinding | `Pathfinding.swift` | Medium-Hard | BFS in JAX |
| Stage generation | `GameState.initializeStage()` | Medium | Random with JAX PRNG |
| Action execution | `GameState.tryExecuteAction()` | Hard | Core game loop |

#### JAX Constraints

JAX requires functional programming patterns:

1. **Pure functions**: No mutation, no side effects
   ```python
   # Bad (mutation)
   state.player.hp -= damage

   # Good (return new state)
   new_player = player._replace(hp=player.hp - damage)
   ```

2. **Fixed-size arrays**: Variable-length lists become max-size arrays with masks
   ```python
   # Max 20 enemies, with active mask
   enemies = jnp.zeros((MAX_ENEMIES, ENEMY_FEATURES))
   enemy_mask = jnp.zeros(MAX_ENEMIES, dtype=bool)
   ```

3. **No Python control flow on traced values**: Use JAX primitives
   ```python
   # Bad
   if enemy.hp <= 0:
       remove_enemy()

   # Good
   enemy_alive = enemy.hp > 0
   enemy_mask = enemy_mask & enemy_alive
   ```

4. **Use `jax.lax` for control flow**:
   - `jax.lax.cond(pred, true_fn, false_fn)` for if/else
   - `jax.lax.switch(index, branches)` for switch/case
   - `jax.lax.fori_loop(start, stop, body_fn, init)` for loops
   - `jax.lax.scan` for sequential operations

#### Proposed JAX State Structure

```python
import jax.numpy as jnp
from typing import NamedTuple

class Player(NamedTuple):
    row: int
    col: int
    hp: int          # 0-3
    credits: int
    energy: int
    data_siphons: int
    base_attack: int

class GameState(NamedTuple):
    # Player
    player: Player

    # Grid: 6x6 cells, each with content type + resources
    grid_content: jnp.ndarray    # (6, 6) - content type enum
    grid_resources: jnp.ndarray  # (6, 6, 2) - credits, energy
    grid_block_data: jnp.ndarray # (6, 6, 3) - points, spawn_count, siphoned

    # Enemies: fixed-size array with mask
    enemies: jnp.ndarray         # (MAX_ENEMIES, ENEMY_FEATURES)
    enemy_mask: jnp.ndarray      # (MAX_ENEMIES,) bool

    # Transmissions: fixed-size array with mask
    transmissions: jnp.ndarray   # (MAX_TRANSMISSIONS, TRANS_FEATURES)
    trans_mask: jnp.ndarray      # (MAX_TRANSMISSIONS,) bool

    # Programs owned: binary vector
    owned_programs: jnp.ndarray  # (23,) bool

    # Game state
    stage: int
    turn: int
    show_activated: bool
    scheduled_tasks_disabled: bool

    # RNG key for stochastic operations
    rng_key: jnp.ndarray

# Constants
MAX_ENEMIES = 20
MAX_TRANSMISSIONS = 10
ENEMY_FEATURES = 5  # type, row, col, hp, stunned
TRANS_FEATURES = 4  # row, col, turns_remaining, enemy_type
```

#### Core Functions to Implement

```python
def reset(rng_key: jnp.ndarray) -> GameState:
    """Initialize a new game."""
    ...

def step(state: GameState, action: int) -> tuple[GameState, float, bool, dict]:
    """Execute one action, return (new_state, reward, done, info)."""
    ...

def get_valid_actions(state: GameState) -> jnp.ndarray:
    """Return boolean mask of valid actions (28,)."""
    ...

def get_observation(state: GameState) -> dict[str, jnp.ndarray]:
    """Convert state to observation dict matching current format."""
    ...
```

### Phase 3: PureJaxRL Integration

Once the JAX environment is complete:

1. Implement action-masked PPO (or find existing implementation)
2. Integrate with PureJaxRL's training loop
3. Benchmark on TPU
4. Tune hyperparameters for massively parallel training

## Development Workflow

1. **Local development**: macOS with JAX CPU backend
   ```bash
   pip install jax
   ```

2. **Testing**: Verify JAX env matches Swift env behavior
   - Run same action sequences through both
   - Compare observations, rewards, done flags

3. **TPU deployment**: Google TPU Research Cloud
   ```bash
   pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
   ```

## Estimated Effort

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1: SBX + masking | 2-3 days | None |
| Phase 2: JAX env port | 2-4 weeks | JAX familiarity |
| Phase 3: PureJaxRL | 1 week | Phase 2 complete |

## Open Questions

1. **Action masking in PureJaxRL**: Does it exist? May need custom implementation.
2. **Observation format**: Keep current Dict space or flatten for simplicity?
3. **Vectorization**: How many parallel envs fit on TPU v2-8 / v3-8?
4. **Mixed precision**: Use bfloat16 for better TPU performance?

## References

- [PureJaxRL](https://github.com/luchris429/purejaxrl)
- [SBX (Stable Baselines JAX)](https://github.com/araffin/sbx)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Action Masking in PPO](https://costa.sh/blog-a-closer-look-at-invalid-action-masking-in-policy-gradient-algorithms.html)
- [A Closer Look at Invalid Action Masking (paper)](https://arxiv.org/abs/2006.14171)
