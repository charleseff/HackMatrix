"""
Training configuration for PureJaxRL.

Provides TrainConfig dataclass with PPO hyperparameters and
device detection utilities for CPU/GPU/TPU.
"""

from dataclasses import dataclass
from typing import Optional
import jax


@dataclass
class TrainConfig:
    """Training configuration with PPO hyperparameters.

    Default values are tuned for HackMatrix environment.
    """
    # Environment
    num_envs: int = 256
    num_steps: int = 128  # Steps per rollout

    # Training duration
    total_timesteps: int = 10_000_000

    # PPO hyperparameters
    learning_rate: float = 2.5e-4
    gamma: float = 0.99           # Discount factor
    gae_lambda: float = 0.95      # GAE lambda
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_eps: float = 0.2         # PPO clipping epsilon
    vf_coef: float = 0.5          # Value function loss coefficient
    ent_coef: float = 0.1         # Entropy bonus (0.1+ prevents collapse)
    max_grad_norm: float = 0.5    # Gradient clipping

    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 2

    # Logging
    log_interval: int = 10        # Log every N updates
    eval_interval: int = 100      # Evaluate every N updates

    # Checkpointing
    save_interval: int = 1000     # Save every N updates
    checkpoint_dir: str = "checkpoints"

    # Random seed
    seed: int = 0

    # Derived values (computed)
    @property
    def num_updates(self) -> int:
        """Total number of PPO updates."""
        return self.total_timesteps // (self.num_envs * self.num_steps)

    @property
    def minibatch_size(self) -> int:
        """Size of each minibatch."""
        return (self.num_envs * self.num_steps) // self.num_minibatches

    @property
    def batch_size(self) -> int:
        """Total batch size per update."""
        return self.num_envs * self.num_steps


def get_device_config() -> dict:
    """Detect available devices and return configuration.

    Returns:
        Dictionary with device info:
        - device_type: "cpu", "gpu", or "tpu"
        - device_count: Number of devices
        - backend: JAX backend name
    """
    devices = jax.devices()
    device = devices[0]
    device_type = device.platform

    return {
        "device_type": device_type,
        "device_count": len(devices),
        "backend": jax.default_backend(),
        "devices": devices,
    }


def auto_tune_for_device(config: TrainConfig) -> TrainConfig:
    """Adjust config based on detected device.

    TPU: Increase parallelism
    GPU: Moderate parallelism
    CPU: Reduce parallelism for memory

    Args:
        config: Base configuration

    Returns:
        Tuned configuration for detected device
    """
    device_info = get_device_config()
    device_type = device_info["device_type"]
    device_count = device_info["device_count"]

    if device_type == "tpu":
        # TPU: maximize parallelism
        return TrainConfig(
            num_envs=config.num_envs * device_count,
            num_steps=config.num_steps,
            total_timesteps=config.total_timesteps,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            num_minibatches=config.num_minibatches * device_count,
            update_epochs=config.update_epochs,
            clip_eps=config.clip_eps,
            vf_coef=config.vf_coef,
            ent_coef=config.ent_coef,
            max_grad_norm=config.max_grad_norm,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            log_interval=config.log_interval,
            eval_interval=config.eval_interval,
            save_interval=config.save_interval,
            checkpoint_dir=config.checkpoint_dir,
            seed=config.seed,
        )
    elif device_type == "gpu":
        # GPU: moderate settings
        return config
    else:
        # CPU: reduce for memory
        return TrainConfig(
            num_envs=min(config.num_envs, 64),
            num_steps=config.num_steps,
            total_timesteps=config.total_timesteps,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            num_minibatches=min(config.num_minibatches, 2),
            update_epochs=config.update_epochs,
            clip_eps=config.clip_eps,
            vf_coef=config.vf_coef,
            ent_coef=config.ent_coef,
            max_grad_norm=config.max_grad_norm,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            log_interval=config.log_interval,
            eval_interval=config.eval_interval,
            save_interval=config.save_interval,
            checkpoint_dir=config.checkpoint_dir,
            seed=config.seed,
        )
