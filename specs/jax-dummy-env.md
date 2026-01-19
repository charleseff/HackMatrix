# JAX Dummy Environment Spec

## Goal

Create a minimal **pure functional JAX environment** that can be JIT-compiled and run efficiently on TPUs. The environment uses real JAX primitives but returns dummy observations (no actual game logic yet).

This establishes the foundation for:

1. TPU-optimized training with PureJaxRL
2. Validation that JAX patterns work correctly (PRNG, JIT, vmap)
3. Interface parity testing against Swift implementation
4. Training models in JAX that can be deployed to play in Swift

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Training Paths (separate, not interchangeable)             │
│                                                             │
│  train.py ──► HackEnv (Gymnasium) ──► Swift subprocess      │
│               For local dev/debugging                       │
│                                                             │
│  train_jax.py ──► Pure JAX env ──► PureJaxRL                │
│                   For TPU training                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Interface Parity Testing                                   │
│                                                             │
│  test_env_parity.py                                         │
│       ├── SwiftEnvAdapter ──► HackEnv                       │
│       └── JaxEnvAdapter ──► Pure JAX env                    │
│       Compare outputs for same inputs                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Model Portability                                          │
│                                                             │
│  JAX training ──► save weights ──► Swift loads ──► plays    │
└─────────────────────────────────────────────────────────────┘
```

## Observation/Action Space

Must match Swift environment exactly for model portability and parity testing.

### Observation Space

| Component | Shape | Description |
|-----------|-------|-------------|
| Player state | (10,) | `[row, col, hp, credits, energy, stage, dataSiphons, baseAttack, showActivated, scheduledTasksDisabled]` |
| Programs | (23,) | Binary int32 vector of owned programs |
| Grid | (6, 6, 40) | 40 features per cell encoding enemies, blocks, transmissions, etc. |

### Action Space

28 discrete actions:

| Index | Action |
|-------|--------|
| 0 | Move up |
| 1 | Move down |
| 2 | Move left |
| 3 | Move right |
| 4 | Siphon |
| 5-27 | Programs (23 total) |

## Pure JAX Environment Design

### State Structure

All environment state is explicit (no `self` mutation). This enables JIT compilation.

```python
@struct.dataclass
class EnvState:
    """Immutable environment state."""
    step_count: int
    # Future: player position, grid state, enemies, etc.

@struct.dataclass
class Observation:
    """Observation returned to agent."""
    player_state: jax.Array  # (10,)
    programs: jax.Array      # (23,) int32
    grid: jax.Array          # (6, 6, 40)
```

### Functional Interface

```python
def reset(key: jax.Array) -> tuple[EnvState, Observation]:
    """Initialize environment state."""
    ...

def step(state: EnvState, action: int, key: jax.Array) -> tuple[EnvState, Observation, float, bool]:
    """Pure function: (state, action, key) -> (new_state, obs, reward, done)"""
    ...

def get_valid_actions(state: EnvState) -> jax.Array:
    """Return mask of valid actions for current state."""
    ...
```

### Dummy Behavior

For this initial implementation:

- `reset()`: Returns zeroed observation
- `step()`: Returns zeroed observation, `reward=0.0`, `done=True` with 10% probability
- `get_valid_actions()`: Returns `[0, 1, 2, 3]` (directional actions only)

## File Structure

```
python/
├── hack_env.py              # Swift-backed Gymnasium env (unchanged)
├── jax_env.py               # Pure JAX environment (this spec)
├── test_env_parity.py       # Interface comparison tests
├── scripts/
│   ├── train.py             # Swift + stable-baselines3 (unchanged)
│   └── train_jax.py         # Pure JAX + PureJaxRL (new)
```

## Implementation

### `jax_env.py`

```python
"""
Pure functional JAX environment for HackMatrix.
Designed for JIT compilation and TPU training with PureJaxRL.
"""

import jax
import jax.numpy as jnp
from flax import struct

# ---------------------------------------------------------------------------
# State and Observation dataclasses
# ---------------------------------------------------------------------------

@struct.dataclass
class EnvState:
    """Immutable environment state."""
    step_count: jnp.int32


@struct.dataclass
class Observation:
    """Observation returned to agent."""
    player_state: jax.Array  # (10,)
    programs: jax.Array      # (23,) int32
    grid: jax.Array          # (6, 6, 40)


# ---------------------------------------------------------------------------
# Environment constants
# ---------------------------------------------------------------------------

NUM_ACTIONS = 28
GRID_SIZE = 6
GRID_FEATURES = 40  # Changed from 20
PLAYER_STATE_SIZE = 10  # Changed from 9
NUM_PROGRAMS = 23  # Added


# ---------------------------------------------------------------------------
# Core environment functions (pure, JIT-compatible)
# ---------------------------------------------------------------------------

@jax.jit
def reset(key: jax.Array) -> tuple[EnvState, Observation]:
    """
    Initialize environment state.

    Args:
        key: JAX PRNG key

    Returns:
        (initial_state, initial_observation)
    """
    state = EnvState(step_count=jnp.int32(0))
    obs = _zero_observation()
    return state, obs


@jax.jit
def step(
    state: EnvState,
    action: jnp.int32,
    key: jax.Array
) -> tuple[EnvState, Observation, jnp.float32, jnp.bool_]:
    """
    Take one environment step.

    Args:
        state: Current environment state
        action: Action index (0-27)
        key: JAX PRNG key

    Returns:
        (new_state, observation, reward, done)
    """
    new_state = EnvState(step_count=state.step_count + 1)
    obs = _zero_observation()
    reward = jnp.float32(0.0)

    # 10% chance of termination
    done = jax.random.uniform(key) < 0.1

    return new_state, obs, reward, done


@jax.jit
def get_valid_actions(state: EnvState) -> jax.Array:
    """
    Return mask of valid actions.

    Args:
        state: Current environment state

    Returns:
        Boolean array of shape (NUM_ACTIONS,) where True = valid
    """
    # For dummy env: only directional actions (0-3) are valid
    mask = jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_)
    mask = mask.at[0:4].set(True)
    return mask


def _zero_observation() -> Observation:
    """Create zeroed observation."""
    return Observation(
        player_state=jnp.zeros(PLAYER_STATE_SIZE, dtype=jnp.float32),
        programs=jnp.zeros(NUM_PROGRAMS, dtype=jnp.int32),
        grid=jnp.zeros((GRID_SIZE, GRID_SIZE, GRID_FEATURES), dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Vectorized versions for batch training
# ---------------------------------------------------------------------------

# vmap over batch dimension for parallel environment execution
batched_reset = jax.vmap(reset)
batched_step = jax.vmap(step)
batched_get_valid_actions = jax.vmap(get_valid_actions)
```

### `test_env_parity.py`

**Note on parity testing scope:**

These tests verify **interface/structural compatibility**, not behavioral equivalence:

| What we CAN test | What we CAN'T test (yet) |
|------------------|--------------------------|
| Observation shapes match | Same seed → same observations |
| Action space size (28) | Same actions → same rewards |
| Return types (float, bool, etc.) | Deterministic replay |
| Valid actions format (list of ints) | Exact value comparisons |

Behavioral equivalence testing becomes relevant when JAX implements real game logic.
This would require adding seed support to Swift for deterministic comparisons.

```python
"""
Test that JAX and Swift environments have compatible interfaces.

NOTE: These tests verify structural compatibility (shapes, types, formats),
not behavioral equivalence. The Swift env doesn't support seeding yet, and
both implementations have inherent randomness, so exact value comparisons
are not possible at this stage.
"""

import numpy as np
from typing import Protocol

# ---------------------------------------------------------------------------
# Common test interface
# ---------------------------------------------------------------------------

class EnvAdapter(Protocol):
    """Minimal interface for parity testing."""

    def reset(self) -> tuple[dict, list[int]]:
        """Returns (observation_dict, valid_actions)"""
        ...

    def step(self, action: int) -> tuple[dict, float, bool, list[int]]:
        """Returns (observation_dict, reward, done, valid_actions)"""
        ...


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class SwiftEnvAdapter:
    """Adapter for Swift-backed HackEnv."""

    def __init__(self):
        from hack_env import HackEnv
        self.env = HackEnv()

    def reset(self) -> tuple[dict, list[int]]:
        obs, _ = self.env.reset()
        valid = self.env.get_valid_actions()
        return obs, valid

    def step(self, action: int) -> tuple[dict, float, bool, list[int]]:
        obs, reward, done, _, _ = self.env.step(action)
        valid = self.env.get_valid_actions()
        return obs, reward, done, valid


class JaxEnvAdapter:
    """Adapter for pure JAX environment."""

    def __init__(self):
        import jax
        import jax_env
        self.jax = jax
        self.env = jax_env
        self.state = None
        self.key = jax.random.PRNGKey(0)

    def reset(self) -> tuple[dict, list[int]]:
        self.key, subkey = self.jax.random.split(self.key)
        self.state, obs = self.env.reset(subkey)
        valid_mask = self.env.get_valid_actions(self.state)
        return self._obs_to_dict(obs), self._mask_to_list(valid_mask)

    def step(self, action: int) -> tuple[dict, float, bool, list[int]]:
        self.key, subkey = self.jax.random.split(self.key)
        self.state, obs, reward, done = self.env.step(self.state, action, subkey)
        valid_mask = self.env.get_valid_actions(self.state)
        return (
            self._obs_to_dict(obs),
            float(reward),
            bool(done),
            self._mask_to_list(valid_mask),
        )

    def _obs_to_dict(self, obs) -> dict:
        return {
            "player_state": np.asarray(obs.player_state),
            "programs": np.asarray(obs.programs),
            "grid": np.asarray(obs.grid),
        }

    def _mask_to_list(self, mask) -> list[int]:
        return [i for i, v in enumerate(np.asarray(mask)) if v]


# ---------------------------------------------------------------------------
# Interface/structural parity tests
# ---------------------------------------------------------------------------

def test_observation_shapes():
    """Verify both envs return same observation shapes."""
    swift = SwiftEnvAdapter()
    jax_adapter = JaxEnvAdapter()

    swift_obs, _ = swift.reset()
    jax_obs, _ = jax_adapter.reset()

    for key in ["player_state", "programs", "grid"]:
        assert swift_obs[key].shape == jax_obs[key].shape, \
            f"Shape mismatch for {key}: {swift_obs[key].shape} vs {jax_obs[key].shape}"

    print("Observation shapes match")


def test_observation_dtypes():
    """Verify both envs return same observation dtypes."""
    swift = SwiftEnvAdapter()
    jax_adapter = JaxEnvAdapter()

    swift_obs, _ = swift.reset()
    jax_obs, _ = jax_adapter.reset()

    for key in ["player_state", "programs", "grid"]:
        assert swift_obs[key].dtype == jax_obs[key].dtype, \
            f"Dtype mismatch for {key}: {swift_obs[key].dtype} vs {jax_obs[key].dtype}"

    print("Observation dtypes match")


def test_valid_actions_format():
    """Verify valid actions are returned in same format."""
    swift = SwiftEnvAdapter()
    jax_adapter = JaxEnvAdapter()

    _, swift_valid = swift.reset()
    _, jax_valid = jax_adapter.reset()

    assert isinstance(swift_valid, list), "Swift valid_actions should be list"
    assert isinstance(jax_valid, list), "JAX valid_actions should be list"
    assert all(isinstance(a, int) for a in swift_valid), "Swift actions should be ints"
    assert all(isinstance(a, int) for a in jax_valid), "JAX actions should be ints"
    assert all(0 <= a < 28 for a in swift_valid), "Swift actions should be in range 0-27"
    assert all(0 <= a < 28 for a in jax_valid), "JAX actions should be in range 0-27"

    print("Valid actions format matches")


def test_step_return_types():
    """Verify step() returns correct types."""
    swift = SwiftEnvAdapter()
    jax_adapter = JaxEnvAdapter()

    _, swift_valid = swift.reset()
    _, jax_valid = jax_adapter.reset()

    # Take a valid action in each
    swift_action = swift_valid[0] if swift_valid else 0
    jax_action = jax_valid[0] if jax_valid else 0

    swift_obs, swift_reward, swift_done, swift_valid = swift.step(swift_action)
    jax_obs, jax_reward, jax_done, jax_valid = jax_adapter.step(jax_action)

    # Check return types (not values)
    assert isinstance(swift_reward, float), f"Swift reward should be float, got {type(swift_reward)}"
    assert isinstance(jax_reward, float), f"JAX reward should be float, got {type(jax_reward)}"
    assert isinstance(swift_done, bool), f"Swift done should be bool, got {type(swift_done)}"
    assert isinstance(jax_done, bool), f"JAX done should be bool, got {type(jax_done)}"

    print("Step return types match")


if __name__ == "__main__":
    test_observation_shapes()
    test_observation_dtypes()
    test_valid_actions_format()
    test_step_return_types()
    print("\nAll interface parity tests passed!")
```

**Future: Behavioral equivalence testing**

When JAX implements real game logic, we may want to verify behavioral equivalence.
This would require:

1. Adding seed support to Swift env
2. Ensuring deterministic game logic (or accepting statistical equivalence)
3. Comparing exact observations/rewards for same seed + action sequence

This is out of scope for the dummy JAX env but noted here for future reference.

### `scripts/train_jax.py` (sketch)

```python
"""
TPU-optimized training using pure JAX environment + PureJaxRL.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple

import jax_env


class Transition(NamedTuple):
    obs: jax_env.Observation
    action: jnp.int32
    reward: jnp.float32
    done: jnp.bool_
    next_obs: jax_env.Observation


def make_train(config):
    """Create JIT-compiled training function."""

    def train(key: jax.Array):
        # Initialize
        key, env_key, policy_key = jax.random.split(key, 3)
        env_state, obs = jax_env.reset(env_key)

        # Training loop would go here
        # - Collect trajectories using jax_env.step
        # - Update policy
        # - All JIT-compiled together

        return {"placeholder": True}

    return train


def main():
    key = jax.random.PRNGKey(0)

    # Check available devices
    print(f"JAX devices: {jax.devices()}")

    # Create and run training
    train_fn = make_train(config={})
    train_fn = jax.jit(train_fn)

    result = train_fn(key)
    print(f"Training complete: {result}")


if __name__ == "__main__":
    main()
```

## Model Portability (Train JAX → Play Swift)

The trained policy network must be exportable so Swift can load it for gameplay.

### Requirements

1. **Same observation space**: Both envs produce observations with identical shapes
2. **Same action space**: Policy outputs action index 0-27
3. **Exportable weights**: Save policy weights in portable format

### Export Format Options

| Format | Pros | Cons |
|--------|------|------|
| `.npz` (numpy) | Simple, easy to load anywhere | Need to reimplement network in Swift |
| ONNX | Standard format, many runtimes | Adds dependency |
| CoreML | Native Apple, optimized | Apple-only |
| Raw JSON | Maximum portability | Verbose, slow |

### Recommended Approach

1. Save weights as `.npz` with network architecture metadata
2. Implement equivalent network in Swift using Accelerate/Metal
3. Load weights and run inference in Swift

```python
# Export from JAX
import numpy as np

def export_policy(params, path: str):
    """Export policy weights for Swift inference."""
    np.savez(
        path,
        # Flatten all params to numpy arrays
        **{k: np.asarray(v) for k, v in flatten_params(params).items()},
        # Include architecture metadata
        _architecture="mlp",
        _layer_sizes=[729, 256, 128, 28],  # input -> hidden -> output
    )
```

## Testing

### Manual Verification

```bash
cd python && source venv/bin/activate

# Test JAX env directly
python -c "
import jax
import jax_env

key = jax.random.PRNGKey(42)
state, obs = jax_env.reset(key)
print('Observation shapes:')
print(f'  player_state: {obs.player_state.shape}')
print(f'  programs: {obs.programs.shape}')
print(f'  grid: {obs.grid.shape}')

key, subkey = jax.random.split(key)
state, obs, reward, done = jax_env.step(state, 0, subkey)
print(f'Step result - reward: {reward}, done: {done}')

valid = jax_env.get_valid_actions(state)
print(f'Valid actions mask sum: {valid.sum()} (expect 4)')
"

# Run parity tests
python test_env_parity.py

# Test TPU/GPU detection
python -c "import jax; print('Devices:', jax.devices())"
```

### Expected Output

- Observation shapes: `player_state: (10,)`, `programs: (23,)`, `grid: (6, 6, 40)`
- Valid actions: 4 valid (indices 0-3)
- Episodes terminate ~10% of steps
- Parity tests pass

## Future Work

Once this dummy environment is working:

1. Replace dummy logic with actual JAX-based game state (see `jax-implementation.md`)
2. Implement full PureJaxRL training loop in `train_jax.py`
3. Add vectorized environments with `vmap` for batch training
4. Implement model export and Swift inference
5. Optimize for TPU (bfloat16, batch sizes, etc.)
