# Specs Directory

Design documents and implementation specifications for HackMatrix.

## Current Focus

**Active spec:** [jax-dummy-env.md](./jax-dummy-env.md)

Implement the minimal JAX dummy environment first. This establishes JAX patterns and enables parity testing before the full port.

## Specs Index

| Spec | Status | Description |
|------|--------|-------------|
| [jax-dummy-env.md](./jax-dummy-env.md) | **Active** | Minimal JAX dummy environment for plug-and-play testing with Swift env |
| [jax-implementation.md](./jax-implementation.md) | Deferred | Full JAX port of game logic (depends on dummy env) |

## Usage

Before starting a major feature or architectural change, create a spec document here.

**Status lifecycle:**
- **Active** - Currently being implemented
- **Deferred** - Planned but not yet started (may depend on other specs)
- **Complete** - Fully implemented and verified

When finishing a spec, update its status to **Complete** and set the next spec to **Active** in both the Current Focus section and the Specs Index table.
