# Implementation Plan: Test Reorganization

**Spec:** [test-reorganization.md](specs/test-reorganization.md)
**Status:** In Progress

## Overview

Reorganize and expand the test suite to:
1. Create proper pytest configuration in pyproject.toml
2. Reorganize tests into parity/ and implementation/ subdirectories
3. Add scheduled tasks testing
4. Add get_internal_state() for implementation-level testing of hidden state

## Phase 1: Test Infrastructure

### Tasks

- [ ] Create `python/pyproject.toml` with pytest configuration
- [ ] Create `python/tests/parity/` directory
- [ ] Move existing test files to `parity/`
- [ ] Create `python/tests/parity/__init__.py`
- [ ] Create `python/tests/implementation/` directory
- [ ] Create `python/tests/implementation/__init__.py`
- [ ] Update conftest.py imports (if needed)
- [ ] Delete `run_all_tests.py`
- [ ] Verify all existing tests pass with new structure

### New Directory Structure

```
python/tests/
├── parity/                    # Interface-level parity tests
│   ├── __init__.py
│   ├── test_movement.py
│   ├── test_siphon.py
│   ├── test_programs.py
│   ├── test_enemies.py
│   ├── test_turns.py
│   ├── test_stages.py
│   ├── test_action_mask.py
│   ├── test_edge_cases.py
│   ├── test_rewards.py
│   └── test_interface_smoke.py
├── implementation/            # Implementation-level tests
│   ├── __init__.py
│   ├── test_scheduled_tasks.py
│   ├── test_hidden_state.py
│   └── test_stage_generation.py
├── conftest.py               # Shared fixtures
├── env_interface.py          # Interface protocol
├── swift_env_wrapper.py      # Swift wrapper
└── jax_env_wrapper.py        # JAX wrapper
```

## Phase 2: Scheduled Tasks Parity Tests

Add tests that verify observable effects of scheduled tasks.

### Tasks

- [ ] Create `parity/test_scheduled_tasks.py`
- [ ] Add test: transmission spawns after interval
- [ ] Add test: siphon delays scheduled spawn
- [ ] Add test: CALM program disables scheduled tasks

## Phase 3: Implementation-Level Tests

### Tasks

- [ ] Add `get_internal_state()` to Swift protocol
- [ ] Add `getInternalState` command to HeadlessGameCLI
- [ ] Add `InternalState` dataclass to env_interface.py
- [ ] Add `get_internal_state()` to SwiftEnvWrapper
- [ ] Add `get_internal_state()` to JaxEnvWrapper (stub for now)
- [ ] Create `implementation/test_scheduled_tasks.py`
- [ ] Create `implementation/test_hidden_state.py`
- [ ] Create `implementation/test_stage_generation.py`

### InternalState Structure

```python
@dataclass
class InternalState:
    scheduled_task_interval: int
    next_scheduled_task_turn: int
    pending_siphon_transmissions: int
    enemies: list[EnemyInternalState]

@dataclass
class EnemyInternalState:
    row: int
    col: int
    hp: int
    disabled_turns: int
    is_stunned: bool
    spawned_from_siphon: bool
    is_from_scheduled_task: bool
```

## Success Criteria

1. [ ] `pyproject.toml` created with pytest config
2. [ ] `run_all_tests.py` deleted
3. [ ] Tests reorganized into `parity/` and `implementation/` subdirectories
4. [ ] All existing tests pass after reorganization
5. [ ] Scheduled task parity tests added (observable effects)
6. [ ] `get_internal_state()` added to interface
7. [ ] Implementation-level tests added for high-priority hidden state

## Implementation Order

1. Phase 1 - Test infrastructure (directory structure, pyproject.toml)
2. Phase 2 - Scheduled tasks parity tests
3. Phase 3 - Implementation-level tests with get_internal_state()
