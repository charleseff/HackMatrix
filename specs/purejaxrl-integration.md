# PureJaxRL Integration Spec

## Goal

Integrate the HackMatrix JAX environment with PureJaxRL for TPU-accelerated training with action masking support.

## Background

### Completed Work

The JAX environment port is complete (`hackmatrix/jax_env/`):
- All 154 parity tests pass
- Environment supports `reset()`, `step()`, `get_valid_actions()`
- Batched versions available via `jax.vmap`
- Pure functional design compatible with JIT compilation

### Why PureJaxRL?
    
PureJaxRL JIT-compiles the entire training loop (environment + policy + updates), enabling:
- 1000+ parallel environments on a single TPU
- Zero Python overhead during training
- Native TPU/GPU utilization

### Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       TPU / GPU                              │
│  ┌─────────────────┐    ┌──────────────────────────────┐    │
│  │  HackMatrix Env │───►│  Action-Masked PPO Network   │    │
│  │  (JAX, vmapped) │◄───│  (Flax/JAX)                  │    │
│  └─────────────────┘    └──────────────────────────────┘    │
│           └── entire training loop JIT-compiled ──┘         │
└─────────────────────────────────────────────────────────────┘
```

## Success Criteria

1. **Training runs on CPU** - Development and testing without TPU
2. **Training runs on GPU** - Local GPU acceleration if available
3. **Training runs on TPU** - Production training on Google TRC
4. **Action masking works** - Invalid actions never selected
5. **Simple CLI interface** - Single script to run training

## Technical Design

### PureJaxRL Interface Requirements

PureJaxRL expects a Gymnax-style environment interface:

```python
# Gymnax interface (what PureJaxRL expects)
def reset(key, params) -> (obs, state)
def step(key, state, action, params) -> (obs, state, reward, done, info)

# Our current interface
def reset(key) -> (state, obs)
def step(state, action, key) -> (state, obs, reward, done)
```

Key differences:
1. **Argument order**: Gymnax puts `key` first, `params` last
2. **Return order**: Gymnax returns `(obs, state)`, we return `(state, obs)`
3. **Info dict**: Gymnax expects `info` dict in step output
4. **Params**: Gymnax has env params (we can use empty struct)

### Action Masking Strategy

PureJaxRL's vanilla PPO does not include action masking. Options:

**Option A: Masked Categorical Distribution (Recommended)**
Modify the policy to use a masked categorical distribution:

```python
def masked_categorical(logits, mask):
    """Sample from categorical with invalid actions masked."""
    # Set invalid action logits to -inf
    masked_logits = jnp.where(mask, logits, -1e9)
    return distrax.Categorical(logits=masked_logits)

# In actor network forward pass
action_mask = env.get_valid_actions(state)
pi = masked_categorical(logits, action_mask)
action = pi.sample(seed=key)
log_prob = pi.log_prob(action)
```

**Option B: Invalid Action Penalty**
Let invalid actions happen but give large negative reward. Not recommended - wastes samples and can destabilize training.

### Environment Wrapper

Create a Gymnax-compatible wrapper around the JAX environment:

```python
@struct.dataclass
class EnvParams:
    """Empty params (HackMatrix has no configurable params)."""
    pass

class HackMatrixGymnax:
    """Gymnax-compatible wrapper for HackMatrix JAX env."""

    def __init__(self):
        self.num_actions = 28
        self.obs_shape = self._compute_obs_shape()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def reset(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]:
        """Reset environment, return (obs, state)."""
        state, obs = jax_env.reset(key)
        flat_obs = self._flatten_obs(obs)
        return flat_obs, state

    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: jnp.int32,
        params: EnvParams
    ) -> tuple[jax.Array, EnvState, jnp.float32, jnp.bool_, dict]:
        """Step environment, return (obs, state, reward, done, info)."""
        state, obs, reward, done = jax_env.step(state, action, key)
        flat_obs = self._flatten_obs(obs)
        info = {"action_mask": jax_env.get_valid_actions(state)}
        return flat_obs, state, reward, done, info

    def get_action_mask(self, state: EnvState) -> jax.Array:
        """Get valid action mask for current state."""
        return jax_env.get_valid_actions(state)

    def _flatten_obs(self, obs: Observation) -> jax.Array:
        """Flatten structured observation to single array."""
        # player_state: (10,)
        # programs: (23,)
        # grid: (6, 6, 42) -> (1512,)
        # Total: 10 + 23 + 1512 = 1545
        return jnp.concatenate([
            obs.player_state,
            obs.programs.astype(jnp.float32),
            obs.grid.ravel(),
        ])

    def _compute_obs_shape(self) -> tuple[int]:
        return (10 + 23 + 6 * 6 * 42,)  # 1545
```

### Modified PPO Implementation

Copy PureJaxRL's `ppo.py` and modify for action masking:

```python
# Key modifications to PureJaxRL's PPO:

# 1. In ActorCritic network, return logits instead of distribution
class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # ... hidden layers ...
        logits = nn.Dense(self.action_dim)(x)  # Raw logits
        value = nn.Dense(1)(x)
        return logits, value

# 2. In _env_step, apply action mask
def _env_step(runner_state, _):
    state, env_state, last_obs, key = runner_state
    key, action_key, step_key = jax.random.split(key, 3)

    # Get action mask from env state
    action_mask = env.get_action_mask(env_state)

    # Get policy logits
    logits, value = network.apply(state.params, last_obs)

    # Apply mask and sample
    pi = masked_categorical(logits, action_mask)
    action = pi.sample(seed=action_key)
    log_prob = pi.log_prob(action)

    # Step environment
    obs, env_state, reward, done, info = env.step(step_key, env_state, action, env_params)

    # Store action_mask in transition for loss computation
    transition = Transition(
        obs=last_obs,
        action=action,
        reward=reward,
        done=done,
        log_prob=log_prob,
        value=value,
        action_mask=action_mask,  # NEW: store mask
    )

    return (state, env_state, obs, key), transition

# 3. In _loss_fn, use stored mask for log_prob computation
def _loss_fn(params, transitions, ...):
    logits, values = network.apply(params, transitions.obs)

    # Use stored action_mask for consistent log_prob
    pi = masked_categorical(logits, transitions.action_mask)
    log_prob = pi.log_prob(transitions.action)
    entropy = pi.entropy()

    # ... rest of PPO loss computation ...
```

### Device Configuration

Support CPU, GPU, and TPU transparently:

```python
import jax

def get_device_config():
    """Detect and configure available device."""
    devices = jax.devices()
    backend = jax.default_backend()

    if backend == "tpu":
        print(f"Running on TPU: {devices}")
        # TPU-specific settings
        num_envs = 2048  # TPUs handle many parallel envs
    elif backend == "gpu":
        print(f"Running on GPU: {devices}")
        num_envs = 256
    else:
        print(f"Running on CPU: {devices}")
        num_envs = 32  # CPU is slower, use fewer envs

    return {"num_envs": num_envs, "backend": backend}
```

### Training Configuration

```python
@dataclass
class TrainConfig:
    # Environment
    num_envs: int = 256          # Parallel environments
    num_steps: int = 128         # Steps per rollout

    # Training
    total_timesteps: int = 10_000_000
    learning_rate: float = 2.5e-4
    gamma: float = 0.99          # Discount factor
    gae_lambda: float = 0.95     # GAE lambda
    num_minibatches: int = 4
    update_epochs: int = 4

    # PPO
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Network
    hidden_dim: int = 256
    num_layers: int = 2

    # Logging
    log_interval: int = 10
    save_interval: int = 100

    def auto_tune_for_device(self):
        """Adjust settings based on available device."""
        config = get_device_config()
        if config["backend"] == "tpu":
            self.num_envs = 2048
            self.num_steps = 256
        elif config["backend"] == "gpu":
            self.num_envs = 512
            self.num_steps = 128
        else:  # CPU
            self.num_envs = 32
            self.num_steps = 64
```

### Training Script

Create `python/scripts/train_purejaxrl.py`:

```python
#!/usr/bin/env python3
"""
Train HackMatrix agent using PureJaxRL with action masking.

Usage:
    cd python && source venv/bin/activate
    python scripts/train_purejaxrl.py [--config config.yaml]

Devices:
    - Automatically detects CPU/GPU/TPU
    - Set JAX_PLATFORMS=cpu to force CPU
    - Set JAX_PLATFORMS=cuda to force GPU
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

from hackmatrix.purejaxrl import (
    HackMatrixGymnax,
    make_train,
    TrainConfig,
)

def main():
    # Config
    config = TrainConfig()
    config.auto_tune_for_device()

    # Create environment
    env = HackMatrixGymnax()

    # Create and run training
    print(f"Starting training on {jax.default_backend()}")
    print(f"  num_envs: {config.num_envs}")
    print(f"  num_steps: {config.num_steps}")
    print(f"  total_timesteps: {config.total_timesteps}")

    key = jax.random.PRNGKey(0)
    train_fn = make_train(config, env)
    train_fn = jax.jit(train_fn)

    result = train_fn(key)

    print(f"Training complete!")
    print(f"  Final return: {result['final_return']}")

if __name__ == "__main__":
    main()
```

## Implementation Plan

### Phase 1: Environment Wrapper

1. Create `hackmatrix/purejaxrl/__init__.py`
2. Create `hackmatrix/purejaxrl/env_wrapper.py` with Gymnax-compatible wrapper
3. Add `_flatten_obs()` for converting structured obs to flat array
4. Add tests to verify wrapper produces correct shapes

### Phase 2: Action-Masked PPO

1. Create `hackmatrix/purejaxrl/masked_ppo.py`
2. Implement `masked_categorical()` distribution helper
3. Copy PureJaxRL's PPO structure with mask modifications
4. Add `Transition` dataclass with `action_mask` field
5. Modify loss function to use stored masks

### Phase 3: Training Infrastructure

1. Create `hackmatrix/purejaxrl/config.py` with `TrainConfig`
2. Create `hackmatrix/purejaxrl/train.py` with `make_train()`
3. Implement device detection and auto-tuning
4. Add logging and checkpointing

### Phase 4: Training Script

1. Create `python/scripts/train_purejaxrl.py`
2. Add CLI argument parsing
3. Add config file support (optional)
4. Test on CPU locally

### Phase 5: TPU Deployment

1. Test on Google TPU Research Cloud
2. Tune hyperparameters for TPU scale
3. Benchmark training throughput
4. Document TPU-specific setup

## File Structure

```
python/hackmatrix/
├── jax_env/           # Existing JAX environment
│   ├── __init__.py
│   ├── state.py
│   ├── env.py
│   └── ...
└── purejaxrl/         # NEW: PureJaxRL integration
    ├── __init__.py
    ├── env_wrapper.py # Gymnax-compatible wrapper
    ├── masked_ppo.py  # Action-masked PPO implementation
    ├── config.py      # Training configuration
    └── train.py       # Training loop

python/scripts/
└── train_purejaxrl.py # Main training entry point
```

## Dependencies

Add to `requirements.txt`:

```
distrax>=0.1.3        # JAX probability distributions
flax>=0.7.0           # Neural network library
optax>=0.1.7          # Optimizers
chex>=0.1.7           # JAX testing utilities
```

For TPU:
```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Open Questions

1. **Checkpointing format**: Use Flax's checkpointing or custom?
2. **Hyperparameter search**: Optuna integration or manual tuning first?
3. **Multi-host TPU**: Do we need data parallelism across TPU hosts?
4. **Observation normalization**: Running mean/std or fixed normalization?

## References

- [PureJaxRL Repository](https://github.com/luchris429/purejaxrl)
- [PureJaxRL Paper](https://arxiv.org/abs/2307.01192)
- [Gymnax](https://github.com/RobertTLange/gymnax) - Reference environment interface
- [Distrax](https://github.com/google-deepmind/distrax) - JAX probability distributions
- [Action Masking in PPO](https://costa.sh/blog-a-closer-look-at-invalid-action-masking-in-policy-gradient-algorithms.html)
- [Google TPU Research Cloud](https://sites.research.google/trc/about/)
