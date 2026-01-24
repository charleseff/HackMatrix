"""Shared training configuration for MaskablePPO models."""

# Production model configuration (from train.py)
MODEL_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 4096,  # Increased for better value estimates
    "batch_size": 64,
    "n_epochs": 20,  # Increased to train value function more
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.3,
    "vf_coef": 1.0,  # Prioritize value function learning
    "policy_kwargs": {
        "net_arch": {
            "pi": [256, 256, 128],  # Policy network: 3 layers
            "vf": [256, 256, 128],  # Value network: 3 layers
        }
    },
    # "device": "mps"
}

# For profiling/benchmarking where speed matters more than final performance
FAST_MODEL_CONFIG = {
    **MODEL_CONFIG,
    "n_steps": 2048,  # Reduce for faster iterations
    "n_epochs": 10,  # Reduce for faster iterations
}
