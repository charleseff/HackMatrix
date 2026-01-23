#!/usr/bin/env python3
"""
PureJaxRL Training Script for HackMatrix.

This script trains an action-masked PPO agent on the HackMatrix environment
using pure JAX for maximum performance on TPU/GPU.

Usage:
    python scripts/train_purejaxrl.py
    python scripts/train_purejaxrl.py --num-envs 512 --total-timesteps 100000000
    python scripts/train_purejaxrl.py --wandb --project hackmatrix

Example TPU usage:
    python scripts/train_purejaxrl.py --num-envs 2048 --total-timesteps 1000000000
"""

import argparse
import os
import sys
import time

# Enable JAX compilation cache (must be set before importing jax)
# This caches compiled XLA programs to disk, speeding up subsequent runs
if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
    cache_dir = os.path.join(os.path.dirname(__file__), "..", ".jax_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir

import jax
import jax.numpy as jnp

# Add python directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hackmatrix.purejaxrl import (
    HackMatrixGymnax,
    TrainConfig,
    make_train,
    get_device_config,
)
from hackmatrix.purejaxrl.config import auto_tune_for_device
from hackmatrix.purejaxrl.logging import TrainingLogger, print_config
from hackmatrix.purejaxrl.checkpointing import save_checkpoint, save_params_npz


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HackMatrix agent with PureJaxRL"
    )

    # Environment
    parser.add_argument(
        "--num-envs",
        type=int,
        default=256,
        help="Number of parallel environments (default: 256)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=128,
        help="Steps per rollout (default: 128)",
    )

    # Training duration
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10_000_000,
        help="Total training timesteps (default: 10M)",
    )

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument(
        "--num-minibatches", type=int, default=4, help="Number of minibatches"
    )
    parser.add_argument(
        "--update-epochs", type=int, default=4, help="PPO update epochs"
    )
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coef")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coef")
    parser.add_argument(
        "--max-grad-norm", type=float, default=0.5, help="Max gradient norm"
    )

    # Network
    parser.add_argument(
        "--hidden-dim", type=int, default=256, help="Hidden layer dimension"
    )
    parser.add_argument(
        "--num-layers", type=int, default=2, help="Number of hidden layers"
    )

    # Logging
    parser.add_argument(
        "--log-interval", type=int, default=10, help="Log every N updates"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument(
        "--project", type=str, default="hackmatrix-purejaxrl", help="WandB project name"
    )
    parser.add_argument("--run-name", type=str, default=None, help="WandB run name")

    # Checkpointing
    parser.add_argument(
        "--save-interval", type=int, default=1000, help="Save every N updates"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )

    # Misc
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--auto-tune", action="store_true", help="Auto-tune config for device"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Print device info
    device_info = get_device_config()
    print(f"\nDevice: {device_info['device_type'].upper()}")
    print(f"Device count: {device_info['device_count']}")
    print(f"Backend: {device_info['backend']}")

    # Create config
    config = TrainConfig(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        clip_eps=args.clip_eps,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )

    # Auto-tune for device if requested
    if args.auto_tune:
        config = auto_tune_for_device(config)
        print("Config auto-tuned for device")

    # Check that total_timesteps is sufficient for meaningful training
    # Need at least 3 updates to see any learning signal
    min_updates = 3
    min_timesteps = config.batch_size * min_updates
    if config.total_timesteps < min_timesteps:
        print(f"\n⚠️  Warning: total_timesteps ({config.total_timesteps:,}) is very small")
        print(f"   batch_size = {config.batch_size:,} (num_envs × num_steps)")
        print(f"   Increasing to {min_timesteps:,} for at least {min_updates} updates")
        config = TrainConfig(
            **{**config.__dict__, "total_timesteps": min_timesteps}
        )

    print_config(config)

    # Initialize logger
    logger = TrainingLogger(
        use_wandb=args.wandb,
        project_name=args.project,
        run_name=args.run_name,
    )

    # Create environment and training function
    env = HackMatrixGymnax()
    train_fn = make_train(config, env)

    # Run training
    print("Compiling training function (this may take a moment)...")
    start_compile = time.time()

    key = jax.random.PRNGKey(args.seed)

    # The training function runs the full loop and returns all metrics
    # For real-time logging, we need a different approach
    # Here we just run and log at the end
    final_state, all_metrics = train_fn(key)

    compile_time = time.time() - start_compile
    print(f"Compilation + training completed in {compile_time:.1f}s")

    # Log final metrics
    total_steps = config.num_updates * config.batch_size
    final_metrics = {
        "total_loss": float(all_metrics["total_loss"][-1]),
        "pg_loss": float(all_metrics["pg_loss"][-1]),
        "vf_loss": float(all_metrics["vf_loss"][-1]),
        "entropy": float(all_metrics["entropy"][-1]),
        "mean_reward": float(all_metrics["mean_reward"][-1]),
    }

    print("\nFinal metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save final checkpoint
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    save_params_npz(
        final_state.train_state.params,
        os.path.join(config.checkpoint_dir, "final_params.npz"),
    )

    logger.finish()

    print(f"\nTraining complete! Total timesteps: {total_steps:,}")


if __name__ == "__main__":
    main()
