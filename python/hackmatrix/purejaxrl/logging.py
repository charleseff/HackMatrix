"""
Logging utilities for PureJaxRL training.

Provides console output and optional WandB integration.
"""

import time
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class TrainingLogger:
    """Logger for training progress.

    Handles console output and optional WandB integration.
    """
    use_wandb: bool = False
    project_name: str = "hackmatrix-purejaxrl"
    run_name: Optional[str] = None
    _wandb_run: Any = None
    _start_time: float = 0.0
    _last_log_time: float = 0.0

    def __post_init__(self):
        self._start_time = time.time()
        self._last_log_time = self._start_time

        if self.use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=self.project_name,
                    name=self.run_name,
                )
            except ImportError:
                print("Warning: wandb not installed, disabling wandb logging")
                self.use_wandb = False

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "train",
    ):
        """Log training metrics.

        Args:
            metrics: Dictionary of metric names to values
            step: Current training step
            prefix: Prefix for metric names
        """
        current_time = time.time()
        elapsed = current_time - self._start_time
        sps = step / elapsed if elapsed > 0 else 0

        # Console output
        metric_strs = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        print(f"[Step {step:6d}] [{elapsed:.1f}s] [{sps:.0f} steps/s] {', '.join(metric_strs)}")

        # WandB logging
        if self.use_wandb and self._wandb_run is not None:
            import wandb
            wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            wandb_metrics["steps_per_second"] = sps
            wandb.log(wandb_metrics, step=step)

        self._last_log_time = current_time

    def log_eval(self, metrics: Dict[str, float], step: int):
        """Log evaluation metrics.

        Args:
            metrics: Evaluation metrics
            step: Current training step
        """
        print(f"[Eval @ {step}] " + ", ".join(f"{k}: {v:.2f}" for k, v in metrics.items()))

        if self.use_wandb and self._wandb_run is not None:
            import wandb
            wandb_metrics = {f"eval/{k}": v for k, v in metrics.items()}
            wandb.log(wandb_metrics, step=step)

    def finish(self):
        """Clean up logging resources."""
        total_time = time.time() - self._start_time
        print(f"\nTraining completed in {total_time:.1f}s")

        if self.use_wandb and self._wandb_run is not None:
            import wandb
            wandb.finish()


def format_number(n: float) -> str:
    """Format large numbers with K/M suffixes."""
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return f"{n:.0f}"


def print_config(config):
    """Print training configuration."""
    print("\n" + "=" * 50)
    print("Training Configuration")
    print("=" * 50)
    print(f"  num_envs: {config.num_envs}")
    print(f"  num_steps: {config.num_steps}")
    print(f"  total_timesteps: {format_number(config.total_timesteps)}")
    print(f"  num_updates: {config.num_updates}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  minibatch_size: {config.minibatch_size}")
    print(f"  learning_rate: {config.learning_rate}")
    print(f"  gamma: {config.gamma}")
    print(f"  gae_lambda: {config.gae_lambda}")
    print(f"  clip_eps: {config.clip_eps}")
    print(f"  vf_coef: {config.vf_coef}")
    print(f"  ent_coef: {config.ent_coef}")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  num_layers: {config.num_layers}")
    print("=" * 50 + "\n")
