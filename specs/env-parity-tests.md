# Environment Parity Test Suite Spec

## Goal

Create a comprehensive test suite that validates environment implementations through a common interface. The Swift environment serves as the reference implementation.

This spec covers:
1. Defining the `EnvInterface` contract
2. Implementing `SwiftEnvWrapper` (including new `set_state` capability)
3. Writing exhaustive tests validated against Swift

The JAX implementation will later use these same tests to verify parity (see `jax-implementation.md`).

## Principles

1. **Interface-only testing**: All tests interact with environments through a common interface—no implementation-specific code
2. **Swift as source of truth**: Tests are written/validated against Swift env first
3. **Deterministic scenarios**: Tests set specific game states before sending actions
4. **Flexible assertions**: Each test checks what matters for that scenario (not necessarily every observation bit)

## Interface Contract

Both Swift and JAX environments must implement this interface:

```python
class EnvInterface(Protocol):
    def reset(self) -> Observation:
        """Reset to initial game state, return observation."""
        ...

    def step(self, action: int) -> tuple[Observation, float, bool, dict]:
        """Execute action, return (observation, reward, done, info)."""
        ...

    def get_valid_actions(self) -> list[int]:
        """Return list of valid action indices."""
        ...

    def set_state(self, state: GameState) -> Observation:
        """Set complete game state for test setup, return observation."""
        ...
```

### Observation Structure

```python
@dataclass
class Observation:
    player: PlayerObs      # [row, col, hp, credits, energy, stage, dataSiphons, baseAttack, showActivated, scheduledTasksDisabled]
    programs: list[int]    # 23 binary values indicating owned programs
    grid: np.ndarray       # (6, 6, 40) cell features
```

### GameState Structure (for set_state)

```python
@dataclass
class GameState:
    player: PlayerState
    enemies: list[Enemy]
    transmissions: list[Transmission]
    blocks: list[Block]
    resources: list[Resource]  # credits/energy on cells
    owned_programs: list[int]  # program indices
    stage: int
    turn: int
    # ... other state fields
```

### Action Space (28 actions)

| Index | Action |
|-------|--------|
| 0 | Move up |
| 1 | Move down |
| 2 | Move left |
| 3 | Move right |
| 4 | Siphon |
| 5-27 | Programs (23 total) |

## Architecture

### Test Runner

- **Framework**: pytest
- **Fixtures**:
  - `swift_env` — provides `SwiftEnvWrapper` (for comprehensive tests)
  - `env` — parameterized for smoke tests (both Swift and JAX)

```python
@pytest.fixture
def swift_env():
    return SwiftEnvWrapper()

@pytest.fixture(params=["swift", "jax"])
def env(request):
    """Used for interface smoke tests only."""
    if request.param == "swift":
        return SwiftEnvWrapper()
    else:
        return JaxEnvWrapper()
```

Comprehensive tests use `swift_env`. Interface smoke tests use `env` to verify both wrappers.

### Swift Environment Wrapper

Wraps the existing Swift binary (communicates via JSON stdin/stdout protocol).

Must implement:
- Existing `reset()`, `step()`, `get_valid_actions()` — already supported
- New `set_state()` — needs to be added (extend `--debug-scenario` capability to the protocol)

### JAX Environment Wrapper (Skeleton)

A minimal `JaxEnvWrapper` that implements `EnvInterface` with stub/dummy logic. This verifies the interface contract is implementable by JAX.

- `reset()` → returns dummy observation
- `step(action)` → returns dummy (obs, reward, done, info)
- `get_valid_actions()` → returns dummy mask
- `set_state(state)` → accepts state, returns dummy observation

A smoke test verifies the skeleton implements the interface. Real tests will fail against it until `jax-implementation.md` is complete.

## Implementation Order

### Phase 1: Interface & Both Wrappers

1. Define the `EnvInterface` protocol in Python
2. Implement `SwiftEnvWrapper` with existing functionality
3. Add `set_state` command to Swift JSON protocol
4. Implement `set_state` in `SwiftEnvWrapper`
5. Implement `JaxEnvWrapper` skeleton (stub/dummy returns)
6. Write interface smoke tests that run against both wrappers (verifies interface contract)

### Phase 2: Comprehensive Test Cases

Enumerate and implement all test cases (see Test Cases section below). Use Swift env to validate tests are correct.

> **Note**: Full JAX implementation (making real tests pass) is deferred to `jax-implementation.md`. The skeleton just verifies interface compliance.

## Test Cases

> **Note**: This section to be expanded during planning phase. The planner should explore the Swift codebase to extract all game mechanics and enumerate exhaustive test cases.

### Categories

#### Movement Actions (0-3)

- Move to empty cell
- Move into wall (blocked)
- Move off grid edge (blocked)
- Move onto cell with credits
- Move onto cell with energy
- Move onto cell with both resources
- Move into enemy (attack, kill if damage >= HP)
- Move into enemy (attack, enemy survives)
- Move into block (blocked? or attack?)
- Move into transmission (?)

#### Siphon Action (4)

- Siphon adjacent block (gain resources based on block type)
- Siphon with no adjacent block (invalid action?)
- Siphon block that's already been siphoned
- Siphon effects on different block types

#### Programs (5-27)

Each of the 23 programs needs tests for their specific effects and edge cases.

> **TODO**: List all 23 programs and their expected behaviors

Note: We don't test "program not owned" or "insufficient energy" as invalid actions—action masking prevents these. Instead, we verify the action mask is correct (see Action Masking section).

#### Turn Mechanics

- Player action ends turn (move/attack/siphon)
- Program execution does NOT end turn (can chain)
- Wait program ends turn
- Turn counter increments correctly
- Enemy turn executes after player turn ends

#### Enemy Behavior

- Enemy spawning from transmissions
- Enemy movement (pathfinding toward player)
- Enemy movement with multiple equally-good options (assert one of valid outcomes)
- Enemy attack when adjacent to player
- Enemy status effects (stunned, disabled)

#### Stage Transitions

- Stage completion triggers
- New stage generation (test non-random parts):
  - Number of enemy spawns
  - Positions of enemies carried over
  - Player state preserved/reset appropriately

#### Edge Cases

- Player death (HP reaches 0)
- Win condition

#### Rewards

Reward verification is critical for RL training correctness. Every test should verify the reward returned by `step()` matches expected values.

Reward tests should cover:
- Killing an enemy → positive reward (verify exact value)
- Taking damage → negative reward (verify exact value)
- Collecting credits → reward (if applicable)
- Collecting energy → reward (if applicable)
- Completing a stage → reward
- Player death → terminal reward
- Neutral actions (moving to empty cell) → zero or baseline reward
- Program execution rewards (if any)
- Siphon rewards
- Compound scenarios (e.g., kill enemy but take damage in same turn)

> **TODO**: Extract exact reward values from Swift implementation during planning phase.

#### Action Masking

Action mask verification is a core part of every test. After each `step()`, tests should call `get_valid_actions()` and verify the mask is correct for the resulting state.

Action mask tests should cover:
- Movement blocked by walls/edges → those directions masked
- Movement blocked by blocks → masked
- Siphon only valid when adjacent to unsiphoned block
- Programs masked when not owned
- Programs masked when insufficient energy
- Programs masked based on other conditions (e.g., target requirements)
- All 4 directions independently calculated
- Mask changes correctly as state changes (e.g., gain energy → program becomes valid)

### Test Template

```python
def test_move_up_to_empty_cell(env):
    """Moving up to an empty cell should move the player."""
    # Arrange: Set up specific game state
    state = GameState(
        player=PlayerState(row=3, col=3, hp=3, ...),
        enemies=[],
        ...
    )
    obs = env.set_state(state)

    # Act: Send action
    obs, reward, done, info = env.step(0)  # Move up

    # Assert: Check observation
    assert obs.player.row == 2
    assert obs.player.col == 3
    assert done == False

    # Assert: Verify action mask is correct for new state
    valid_actions = env.get_valid_actions()
    assert 0 in valid_actions      # can move up (row 2 -> 1)
    assert 1 in valid_actions      # can move down (row 2 -> 3)
    # ... etc based on expected state
```

## Handling Non-Determinism

### Enemy Movement Ties

When an enemy has multiple equally-good movement options, both implementations may choose differently. Tests should:

```python
def test_enemy_moves_toward_player(env):
    # ... setup with enemy that has 2 equally good moves ...
    obs, _, _, _ = env.step(action)

    # Assert enemy is in ONE of the valid positions
    valid_positions = [(2, 3), (3, 2)]  # both equally good
    actual_position = (obs.enemies[0].row, obs.enemies[0].col)
    assert actual_position in valid_positions
```

### Stage Generation

New stages have randomness. Tests should check deterministic properties:

```python
def test_stage_transition_enemy_count(env):
    # ... trigger stage completion ...
    obs, _, _, _ = env.step(action)

    # Check non-random properties
    assert obs.player.stage == 2
    assert count_enemies(obs) == expected_new_enemy_count
    # Don't assert specific positions of newly spawned enemies
```

## Files

| File | Purpose |
|------|---------|
| `python/tests/conftest.py` | pytest fixtures |
| `python/tests/env_interface.py` | `EnvInterface` protocol definition |
| `python/tests/swift_env_wrapper.py` | Swift env wrapper implementation |
| `python/tests/jax_env_wrapper.py` | JAX env wrapper skeleton (stub implementation) |
| `python/tests/test_interface_smoke.py` | Smoke tests verifying both wrappers implement interface |
| `python/tests/test_movement.py` | Movement action tests |
| `python/tests/test_siphon.py` | Siphon action tests |
| `python/tests/test_programs.py` | Program execution tests |
| `python/tests/test_enemies.py` | Enemy behavior tests |
| `python/tests/test_turns.py` | Turn mechanics tests |
| `python/tests/test_stages.py` | Stage transition tests |
| `python/tests/test_action_mask.py` | Action mask verification tests |

## Open Questions

1. **State serialization format**: What format should `set_state` accept? JSON matching Swift's internal representation?
2. **Observation comparison helpers**: Should we build utilities for comparing observations with tolerance for non-deterministic parts?
3. **Test coverage tooling**: How do we ensure we've covered all game mechanics?

## Success Criteria

1. `EnvInterface` protocol defined
2. `SwiftEnvWrapper` fully implements `EnvInterface`
3. `JaxEnvWrapper` skeleton implements `EnvInterface` (stub returns)
4. Interface smoke tests pass for both wrappers
5. `set_state` functionality added to Swift JSON protocol
6. All comprehensive tests pass against Swift environment
7. Test coverage includes all programs, all action types, key edge cases
