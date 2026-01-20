# Environment Parity Test Suite Implementation Plan

Based on analysis of `specs/env-parity-tests.md` and codebase exploration.

## Current State Assessment

### What Exists

| Component | Status | Location |
|-----------|--------|----------|
| Swift JSON protocol | Partial | `HackMatrix/GameCommandProtocol.swift` |
| `reset`, `step`, `getValidActions` commands | Complete | GameCommandProtocol.swift |
| `HackEnv` Gymnasium wrapper | Complete | `python/hackmatrix/gym_env.py` |
| JAX dummy environment | Complete | `python/hackmatrix/jax_env.py` |
| Basic parity tests (shapes/dtypes) | Complete | `python/scripts/test_env_parity.py` |
| Debug scenario mode | Complete | `--debug-scenario` flag in HackEnv |

### What's Missing

| Component | Priority | Notes |
|-----------|----------|-------|
| `set_state` JSON command | **P0** | Core blocker - needed for deterministic test scenarios |
| pytest test infrastructure | **P0** | No `tests/` directory, no `conftest.py` |
| `EnvInterface` Protocol | **P1** | Spec defines protocol, not yet implemented |
| `SwiftEnvWrapper` class | **P1** | Exists as adapter, needs `set_state` |
| `JaxEnvWrapper` skeleton | **P1** | Exists as adapter, needs `set_state` stub |
| Comprehensive test cases | **P2** | Movement, siphon, programs, enemies, turns, stages |
| Action mask verification tests | **P2** | Per spec requirements |

## Spec Discrepancies

1. **Interface location**: Spec specifies `python/tests/env_interface.py`, but current adapters are in `python/scripts/test_env_parity.py`
2. **Wrapper naming**: Spec uses `SwiftEnvWrapper`/`JaxEnvWrapper`, current code uses `SwiftEnvAdapter`/`JaxEnvAdapter`
3. **`set_state` missing**: Critical feature not yet in Swift JSON protocol

## Implementation Tasks

### Phase 1: Interface & Infrastructure

- [ ] **1.1** Create `python/tests/` directory structure
- [ ] **1.2** Create `python/tests/conftest.py` with pytest fixtures
- [ ] **1.3** Create `python/tests/env_interface.py` with `EnvInterface` Protocol and dataclasses:
  - `Observation` dataclass
  - `GameState` dataclass for `set_state`
  - `PlayerState`, `Enemy`, `Transmission`, `Block`, `Resource` dataclasses
- [ ] **1.4** Add `set_state` command to Swift JSON protocol (`GameCommandProtocol.swift`)
- [ ] **1.5** Implement `setState()` in `HeadlessGame.swift` to set arbitrary game state
- [ ] **1.6** Create `python/tests/swift_env_wrapper.py` implementing `EnvInterface`
- [ ] **1.7** Create `python/tests/jax_env_wrapper.py` skeleton implementing `EnvInterface`
- [ ] **1.8** Create `python/tests/test_interface_smoke.py` - basic interface compliance tests

### Phase 2: Comprehensive Test Cases

#### Movement Tests (`test_movement.py`)
- [ ] **2.1** Move to empty cell (all 4 directions)
- [ ] **2.2** Move blocked by wall/grid edge
- [ ] **2.3** Move blocked by block
- [ ] **2.4** Move onto cell with credits
- [ ] **2.5** Move onto cell with energy
- [ ] **2.6** Move onto cell with both resources
- [ ] **2.7** Move into enemy (attack, kill)
- [ ] **2.8** Move into enemy (attack, survives)
- [ ] **2.9** Move into transmission

#### Siphon Tests (`test_siphon.py`)
- [ ] **2.10** Siphon adjacent block (cross pattern)
- [ ] **2.11** Siphon with no adjacent block (invalid)
- [ ] **2.12** Siphon already-siphoned block
- [ ] **2.13** Siphon spawns transmissions based on spawn count
- [ ] **2.14** Siphon reveals resources
- [ ] **2.15** Siphon different block types (data, program, question)

#### Program Tests (`test_programs.py`)
Each of the 23 programs needs at least one test:

- [ ] **2.16** `push` (index 5): Push enemies away
- [ ] **2.17** `pull` (index 6): Pull enemies toward
- [ ] **2.18** `crash` (index 7): Clear 8 surrounding cells
- [ ] **2.19** `warp` (index 8): Warp to random enemy/transmission
- [ ] **2.20** `poly` (index 9): Randomize enemy types
- [ ] **2.21** `wait` (index 10): Skip turn, ends turn
- [ ] **2.22** `debug` (index 11): Damage enemies on blocks
- [ ] **2.23** `row` (index 12): Attack all in row
- [ ] **2.24** `col` (index 13): Attack all in column
- [ ] **2.25** `undo` (index 14): Restore previous state
- [ ] **2.26** `step` (index 15): Enemies don't move next turn
- [ ] **2.27** `siph+` (index 16): Gain data siphon
- [ ] **2.28** `exch` (index 17): Convert 4 credits to 4 energy
- [ ] **2.29** `show` (index 18): Reveal cryptogs/transmissions
- [ ] **2.30** `reset` (index 19): Restore to 3 HP
- [ ] **2.31** `calm` (index 20): Disable scheduled spawns
- [ ] **2.32** `d_bom` (index 21): Destroy nearest daemon
- [ ] **2.33** `delay` (index 22): Extend transmissions +3 turns
- [ ] **2.34** `anti-v` (index 23): Damage all viruses
- [ ] **2.35** `score` (index 24): Gain points = stages left
- [ ] **2.36** `reduc` (index 25): Reduce block spawn counts
- [ ] **2.37** `atk+` (index 26): Increase damage to 2 HP
- [ ] **2.38** `hack` (index 27): Damage enemies on siphoned cells

#### Enemy Tests (`test_enemies.py`)
- [ ] **2.39** Enemy spawns from transmission (timer reaches 0)
- [ ] **2.40** Enemy movement toward player (single step)
- [ ] **2.41** Virus double-move (2 steps per turn)
- [ ] **2.42** Glitch can move on blocks
- [ ] **2.43** Cryptog visibility (same row/col vs hidden)
- [ ] **2.44** Enemy attack when adjacent
- [ ] **2.45** Stunned enemy doesn't move
- [ ] **2.46** Disabled enemy behavior

#### Turn Tests (`test_turns.py`)
- [ ] **2.47** Move/attack/siphon ends player turn
- [ ] **2.48** Program execution does NOT end turn
- [ ] **2.49** Wait program ends turn
- [ ] **2.50** Turn counter increments on turn end
- [ ] **2.51** Enemy turn executes after player turn ends
- [ ] **2.52** Chain multiple programs before turn ends

#### Stage Tests (`test_stages.py`)
- [ ] **2.53** Stage completion trigger (reach exit)
- [ ] **2.54** New stage enemy count matches stage number
- [ ] **2.55** Enemies persist across stage transitions
- [ ] **2.56** Player state preserved on stage transition

#### Action Mask Tests (`test_action_mask.py`)
- [ ] **2.57** Movement masked by walls/edges
- [ ] **2.58** Movement masked by blocks
- [ ] **2.59** Siphon only valid adjacent to unsiphoned block
- [ ] **2.60** Programs masked when not owned
- [ ] **2.61** Programs masked when insufficient credits
- [ ] **2.62** Programs masked when insufficient energy
- [ ] **2.63** Mask updates correctly after state changes

#### Edge Case Tests (`test_edge_cases.py`)
- [ ] **2.64** Player death (HP reaches 0)
- [ ] **2.65** Win condition (complete stage 8)

## GameState Serialization Format

For `set_state`, use this JSON structure:

```json
{
  "action": "setState",
  "state": {
    "player": {
      "row": 3,
      "col": 3,
      "hp": 3,
      "credits": 5,
      "energy": 3,
      "dataSiphons": 0,
      "attackDamage": 1,
      "score": 0
    },
    "enemies": [
      {"type": "virus", "row": 1, "col": 1, "hp": 2, "stunned": false}
    ],
    "transmissions": [
      {"row": 2, "col": 2, "turnsRemaining": 3, "enemyType": "daemon"}
    ],
    "blocks": [
      {"row": 0, "col": 0, "type": "data", "points": 5, "spawnCount": 2, "siphoned": false}
    ],
    "resources": [
      {"row": 1, "col": 2, "credits": 2, "energy": 1}
    ],
    "ownedPrograms": [5, 10, 15],
    "stage": 1,
    "turn": 0,
    "showActivated": false,
    "scheduledTasksDisabled": false
  }
}
```

## File Structure

```
python/
├── hackmatrix/
│   ├── gym_env.py          # Existing - no changes needed
│   └── jax_env.py          # Existing - no changes needed
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # pytest fixtures
│   ├── env_interface.py    # EnvInterface protocol & dataclasses
│   ├── swift_env_wrapper.py
│   ├── jax_env_wrapper.py
│   ├── test_interface_smoke.py
│   ├── test_movement.py
│   ├── test_siphon.py
│   ├── test_programs.py
│   ├── test_enemies.py
│   ├── test_turns.py
│   ├── test_stages.py
│   ├── test_action_mask.py
│   └── test_edge_cases.py
└── scripts/
    └── test_env_parity.py  # Existing - keep for standalone parity checks
```

## Success Criteria

1. [ ] `EnvInterface` Protocol defined with all methods
2. [ ] `SwiftEnvWrapper` fully implements `EnvInterface` including `set_state`
3. [ ] `JaxEnvWrapper` skeleton implements `EnvInterface` (stub returns)
4. [ ] Interface smoke tests pass for both wrappers
5. [ ] `set_state` JSON command added to Swift protocol
6. [ ] All comprehensive tests pass against Swift environment
7. [ ] Test coverage includes all 23 programs, all action types, key edge cases
8. [ ] Tests runnable with: `cd python && source venv/bin/activate && pytest tests/`

## Running Tests

```bash
# Run all tests
cd python && source venv/bin/activate && pytest tests/ -v

# Run specific test file
pytest tests/test_movement.py -v

# Run only Swift tests (skip JAX)
pytest tests/ -v -k "swift"

# Run smoke tests only
pytest tests/test_interface_smoke.py -v
```

## Notes

- Tests should handle non-determinism (enemy movement ties) by asserting one of valid outcomes
- Stage generation randomness: only test deterministic properties (enemy counts, player preserved)
- JAX wrapper tests will fail until `jax-implementation.md` is complete (expected)
