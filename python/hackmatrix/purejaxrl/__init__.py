"""
PureJaxRL Integration for HackMatrix.

This package provides Gymnax-compatible environment wrapper and
action-masked PPO implementation for JAX-based training.

Usage:
    from hackmatrix.purejaxrl import HackMatrixGymnax, make_train, TrainConfig

    config = TrainConfig()
    env = HackMatrixGymnax()
    train_fn = make_train(config, env)
    train_state = train_fn(jax.random.PRNGKey(0))
"""

from .config import TrainConfig, get_device_config
from .env_wrapper import EnvParams, HackMatrixGymnax
from .masked_ppo import ActorCritic, MaskedCategorical, Transition, masked_categorical
from .train import (
    RunnerState,
    init_runner_state,
    make_chunked_train,
    make_train,
    make_train_chunk,
)

__all__ = [
    # Environment
    "HackMatrixGymnax",
    "EnvParams",
    # PPO
    "ActorCritic",
    "Transition",
    "masked_categorical",
    "MaskedCategorical",
    # Training
    "TrainConfig",
    "get_device_config",
    "make_train",
    "make_train_chunk",
    "make_chunked_train",
    "init_runner_state",
    "RunnerState",
]
