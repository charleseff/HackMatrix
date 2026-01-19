# JAX Dummy Environment Implementation Plan

## Project Goal

Implement a minimal pure functional JAX environment (`jax_env.py`) as specified in `specs/jax-dummy-env.md` to enable TPU-accelerated training with PureJaxRL.

## Current State Assessment

### What Exists
- **Swift game**: Full game logic in Swift (`HackMatrix/*.swift`)
- **Python Gymnasium wrapper**: `python/hackmatrix/gym_env.py` - production-ready
- **Observation utilities**: `python/hackmatrix/observation_utils.py` - comprehensive parsing/display
- **Training pipeline**: `python/scripts/train.py` - MaskablePPO with W&B integration
- **Requirements**: `python/requirements.txt` - stable-baselines3, PyTorch dependencies

### What Now Exists
- [x] `python/jax_env.py` - Pure JAX environment (COMPLETED)
- [x] `python/test_env_parity.py` - Interface parity tests (COMPLETED)
- [x] `python/scripts/train_jax.py` - JAX training script sketch (COMPLETED)
- [x] JAX/Flax dependencies in requirements (COMPLETED)

## Specification Discrepancies (RESOLVED)

All discrepancies between spec and actual implementation have been resolved:

| Component | Resolution |
|-----------|-----------|
| Player state | Spec updated: (10,) with `showActivated` + `scheduledTasksDisabled` flags |
| Grid | Spec updated: (6, 6, 40) features per cell |
| Flags | Spec updated: no separate flags component, embedded in player state |

**Actual observation structure** (now matches specs/jax-dummy-env.md):
- **Player**: (10,) float32 - `[row, col, hp, credits, energy, stage, dataSiphons, baseAttack, showActivated, scheduledTasksDisabled]`
- **Programs**: (23,) int32 - binary vector of owned programs
- **Grid**: (6, 6, 40) float32 - 40 features per cell:
  - Channels 0-3: Enemy type one-hot (virus, daemon, glitch, cryptog)
  - Channel 4: Enemy HP
  - Channel 5: Enemy stunned
  - Channels 6-8: Block type one-hot (data, program, question)
  - Channel 9: Block points
  - Channel 10: Block siphoned
  - Channels 11-33: Program type one-hot (23 programs)
  - Channel 34: Transmission spawn count
  - Channel 35: Transmission turns
  - Channels 36-37: Credits, energy
  - Channels 38-39: Data siphon cell, exit cell

**Status**: JAX env matches actual `gym_env.py` - parity testing confirmed working.

## Implementation Tasks (Priority Order)

### Phase 1: Specification Alignment
- [x] **P1.1** Update `specs/jax-dummy-env.md` observation space to match `gym_env.py`: (COMPLETED)
  - Player state: (10,) with both `showActivated` and `scheduledTasksDisabled`
  - Grid: (6, 6, 40) features
  - Programs: (23,) int32 binary vector
  - Remove separate flags component
- [x] **P1.2** Update CLAUDE.md observation space documentation to match reality (COMPLETED)

### Phase 2: Dependencies
- [x] **P2.1** Add JAX dependencies to `python/requirements.txt`: (COMPLETED)
  ```
  jax>=0.4.20
  jaxlib>=0.4.20
  flax>=0.8.0
  ```

### Phase 3: Core JAX Environment
- [x] **P3.1** Create `python/jax_env.py` with: (COMPLETED)
  - `EnvState` dataclass using `flax.struct.dataclass`
  - `Observation` dataclass matching gym_env structure:
    - `player_state`: (10,) float32
    - `programs`: (23,) int32
    - `grid`: (6, 6, 40) float32
  - Constants: `NUM_ACTIONS=28`, `GRID_SIZE=6`, `GRID_FEATURES=40`, `PLAYER_STATE_SIZE=10`, `NUM_PROGRAMS=23`
  - `reset(key)` function returning zeroed observation
  - `step(state, action, key)` function with 10% termination probability
  - `get_valid_actions(state)` returning mask for actions 0-3 only
  - `_zero_observation()` helper
  - Vectorized versions: `batched_reset`, `batched_step`, `batched_get_valid_actions`
  - All functions marked with `@jax.jit`

### Phase 4: Parity Tests
- [x] **P4.1** Create `python/test_env_parity.py` with: (COMPLETED)
  - `EnvAdapter` protocol for common interface
  - `SwiftEnvAdapter` wrapping `HackEnv`
  - `JaxEnvAdapter` wrapping `jax_env` functions
  - `test_observation_shapes()` - verify shapes match
  - `test_observation_dtypes()` - verify dtypes match
  - `test_valid_actions_format()` - verify list of ints format
  - `test_step_return_types()` - verify reward (float), done (bool) types

### Phase 5: Training Script Sketch
- [x] **P5.1** Create `python/scripts/train_jax.py` with: (COMPLETED)
  - `Transition` NamedTuple for trajectory storage
  - `make_train(config)` factory returning JIT-compiled train function
  - `main()` with device detection and placeholder training loop
  - Comments indicating where PureJaxRL integration would go

### Phase 6: Testing & Verification
- [x] **P6.1** Manual verification script outputs: (COMPLETED)
  - JAX env observation shapes match spec
  - Valid actions mask has exactly 4 True values (0-3)
  - Episodes terminate ~10% of steps
  - JIT compilation works without errors
- [x] **P6.2** Run parity tests: `python test_env_parity.py` (COMPLETED - all 5 tests pass)
- [x] **P6.3** Test TPU/GPU detection: `python -c "import jax; print(jax.devices())"` (COMPLETED)

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `specs/jax-dummy-env.md` | Modify | Update observation space to match reality |
| `CLAUDE.md` | Modify | Update observation space documentation |
| `python/requirements.txt` | Modify | Add jax, jaxlib, flax |
| `python/jax_env.py` | Create | Pure JAX environment |
| `python/test_env_parity.py` | Create | Interface parity tests |
| `python/scripts/train_jax.py` | Create | Training script sketch |

## Success Criteria

- [x] `python/jax_env.py` exists and is JIT-compilable (VERIFIED)
- [x] `python -c "import jax_env; print('OK')"` succeeds (VERIFIED)
- [x] `python test_env_parity.py` passes all 5 tests (VERIFIED - all tests passing)
- [x] `python scripts/train_jax.py` runs without error (VERIFIED)
- [x] Manual verification shows correct observation shapes and action masking (VERIFIED)

**All success criteria have been met.**

## Notes

- The spec explicitly states this is a **dummy** environment - no real game logic
- `jax-implementation.md` (full JAX port) is marked "NOT READY" - do not implement
- Focus on establishing JAX patterns, not game correctness
- Model portability (JAX → Swift) is future work, not in this scope
