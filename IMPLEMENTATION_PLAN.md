# Implementation Plan: Observation and Attack Fixes

**Spec:** [observation-and-attack-fixes.md](specs/observation-and-attack-fixes.md)
**Status:** Completed

## Completed

All phases have been successfully implemented:

1. **Phase 3 (ATK+)**: Changed `atkPlusUsedThisStage` from Bool to `atkPlusUsesThisStage` as Int, allowing 2 uses per stage. Attack damage now ranges 1→2→3.

2. **Phase 4 (set_state)**: Added `spawnedFromSiphon` and `isFromScheduledTask` fields to `SetStateEnemy` struct, enabling full enemy flag control in test scenarios.

3. **Phase 2 (spawnedFromSiphon)**: Added `spawnedFromSiphon` to `EnemyObservation` and grid encoding (new channel 6).

4. **Phase 1 (siphonCenter)**: Added `siphonCenter` to `CellObservation` and grid encoding (new channel 41).

**Observation Space Changes:**
- Grid features increased from 40 to 42 channels
- Channel 6: Enemy `spawnedFromSiphon` (NEW)
- Channel 41: Cell `siphonCenter` (NEW)
- Player `baseAttack` normalization updated: `(val - 1) / 2.0` (range 1-3)

## Current State Assessment

### Phase 1: Add siphonCenter to Observation Space

**Current State:** `siphonCenter` is tracked in `Grid.swift:44` but NOT exposed in the observation space.

| Component | Status | Details |
|-----------|--------|---------|
| `Cell.siphonCenter` property | Exists | `Grid.swift:44` - Set when player siphons at that cell |
| `CellObservation.siphonCenter` | **MISSING** | `Observation.swift:26-42` - Not included |
| `ObservationBuilder` | **MISSING** | `ObservationBuilder.swift:162-172` - Not passing siphonCenter |
| JSON encoding | **MISSING** | `GameCommandProtocol.swift:205-210` - Not encoding siphonCenter |
| Python grid encoding | **MISSING** | `observation_utils.py` - No channel for siphonCenter |

### Phase 2: Add spawnedFromSiphon to Enemy Observation Space

**Current State:** `spawnedFromSiphon` is tracked on enemies but NOT exposed in observations.

| Component | Status | Details |
|-----------|--------|---------|
| `Enemy.spawnedFromSiphon` property | Exists | `Enemy.swift:57` |
| `EnemyObservation.spawnedFromSiphon` | **MISSING** | `Observation.swift:44-48` - Only type, hp, isStunned |
| `ObservationBuilder` | **MISSING** | `ObservationBuilder.swift:60-75` - Not passing spawnedFromSiphon |
| JSON encoding | **MISSING** | `GameCommandProtocol.swift:213-217` - Not encoding spawnedFromSiphon |
| Python grid encoding | **MISSING** | `observation_utils.py` - No channel for spawnedFromSiphon |

### Phase 3: Allow ATK+ Program to be Used Twice Per Stage

**Current State:** ATK+ is a boolean flag, limited to one use per stage.

| Component | Status | Details |
|-----------|--------|---------|
| `atkPlusUsedThisStage` | Bool | `GameState.swift:55` - Needs to become Int |
| ATK+ execution | Sets to true | `GameState.swift:1420` - Should increment counter |
| ATK+ validity check | Checks bool + attackDamage < 2 | `GameState.swift:954` - Should check count < 2 |
| Stage reset | Resets to false | `GameState.swift:226` - Should reset to 0 |
| Player normalization | attackDamage (1-2) → (0-1) | `observation_utils.py:43` - Needs update for 1-3 range |

**Key Insight:** The current normalization `(baseAttack - 1) / 1.0` assumes attack ranges 1-2. If attack can go 1→2→3, this needs to change to `(baseAttack - 1) / 2.0`.

### Phase 4: Extend set_state for Enemy Flags

**Current State:** `set_state` does not support `spawnedFromSiphon` or `isFromScheduledTask` on enemies.

| Component | Status | Details |
|-----------|--------|---------|
| `SetStateEnemy` struct | Missing fields | `GameCommandProtocol.swift:50-56` |
| `HeadlessGame.setState` | Not setting flags | `HeadlessGame.swift:162-167` |
| Python test helpers | N/A | Will need updates after Swift changes |

## Implementation Tasks

### Phase 1: siphonCenter in Observation Space
- [x] **Swift: Add to CellObservation** (`Observation.swift`)
  - Add `let siphonCenter: Bool` field to `CellObservation` struct
- [x] **Swift: Update ObservationBuilder** (`ObservationBuilder.swift`)
  - Pass `siphonCenter: cell.siphonCenter` in return statement (line ~162)
- [x] **Swift: Update JSON encoding** (`GameCommandProtocol.swift`)
  - Add `siphonCenter` to cell dict (after line 210)
- [x] **Python: Add grid channel** (`observation_utils.py`)
  - Add channel for siphonCenter (increases grid features from 40 to 42)
  - Update observation space shape in `gym_env.py` if needed
- [x] **Test: Verify siphonCenter visible** after player siphons

### Phase 2: spawnedFromSiphon in Enemy Observation Space
- [x] **Swift: Add to EnemyObservation** (`Observation.swift`)
  - Add `let spawnedFromSiphon: Bool` field to `EnemyObservation` struct
- [x] **Swift: Update ObservationBuilder** (`ObservationBuilder.swift`)
  - Pass `spawnedFromSiphon: enemy.spawnedFromSiphon` in enemy observation
- [x] **Swift: Update JSON encoding** (`GameCommandProtocol.swift`)
  - Add `spawnedFromSiphon` to enemy dict (line ~216)
- [x] **Python: Add grid channel** (`observation_utils.py`)
  - Add channel for spawnedFromSiphon (increases enemy features from 6 to 7)
- [x] **Test: Verify spawnedFromSiphon visible** on enemies spawned from siphoning

### Phase 3: ATK+ Usable Twice Per Stage
- [x] **Swift: Change to counter** (`GameState.swift`)
  - Rename `atkPlusUsedThisStage: Bool` to `atkPlusUsesThisStage: Int = 0` (line 55)
- [x] **Swift: Update validity check** (`GameState.swift`)
  - Change line 954 from `!atkPlusUsedThisStage && player.attackDamage < 2`
  - To: `atkPlusUsesThisStage < 2`
- [x] **Swift: Update execution** (`GameState.swift`)
  - Change line 1419-1420 to increment attackDamage and counter
- [x] **Swift: Update stage reset** (`GameState.swift`)
  - Change line 226 to reset counter to 0
- [x] **Swift: Update snapshot/restore** (`GameState.swift`)
  - Update `GameStateSnapshot` struct (line ~2099)
  - Update snapshot creation (line ~1851)
  - Update restore (line ~1901)
- [x] **Python: Update normalization** (`observation_utils.py`)
  - Change line 43 from `(baseAttack - 1) / 1.0` to `(baseAttack - 1) / 2.0`
- [x] **Python: Update denormalization** (`observation_utils.py`)
  - Change line 156 from `* 1` to `* 2`
- [x] **Spec: Update game-mechanics.md** if needed
- [x] **Test: Verify ATK+ works twice** (attack: 1→2→3)
- [x] **Test: Verify ATK+ blocked on third** attempt

### Phase 4: set_state Enemy Flags
- [x] **Swift: Extend SetStateEnemy** (`GameCommandProtocol.swift`)
  - Add `let spawnedFromSiphon: Bool?`
  - Add `let isFromScheduledTask: Bool?`
- [x] **Swift: Update setState** (`HeadlessGame.swift`)
  - Pass new flags when creating Enemy (line ~162)
- [x] **Python: Update test helpers** if needed
- [x] **Test: Verify flags work in set_state**

## Observation Space Changes Summary

**Grid features change from 40 to 42:**

| Index | Current | Proposed |
|-------|---------|----------|
| 0-3 | Enemy type (one-hot) | Enemy type (one-hot) |
| 4 | Enemy HP | Enemy HP |
| 5 | Enemy stunned | Enemy stunned |
| **6** | - | **Enemy spawnedFromSiphon (NEW)** |
| 7-9 | Block type (one-hot) | Block type (one-hot) |
| 10 | Block points | Block points |
| 11 | Block siphoned | Block siphoned |
| **12** | - | **Cell siphonCenter (NEW)** |
| 13-35 | Program one-hot (23) | Program one-hot (23) |
| 36 | Transmission spawncount | Transmission spawncount |
| 37 | Transmission turns | Transmission turns |
| 38 | Credits | Credits |
| 39 | Energy | Energy |
| 40 | Data siphon cell | Data siphon cell |
| 41 | Exit cell | Exit cell |

**Note:** Adding 2 channels increases grid from (6,6,40) to (6,6,42). This affects:
- `gym_env.py` observation_space shape
- `observation_utils.py` feature encoding
- Any JAX/other environments

**Player normalization change:**
- `baseAttack`: Now normalized with `(val - 1) / 2.0` (range 1-3)

## Success Criteria

1. [x] `siphonCenter` visible in cell observation space after player siphons
2. [x] `spawnedFromSiphon` visible in enemy observation space
3. [x] ATK+ program can be used twice per stage (attack: 1→2→3)
4. [x] ATK+ blocked on third attempt in same stage
5. [x] `set_state` supports `spawnedFromSiphon` and `isFromScheduledTask` on enemies
6. [x] All existing tests pass
7. [x] `swift build` succeeds
8. [x] `swift test` passes

## Implementation Order

1. **Phase 3 (ATK+)** - Simplest, no observation space changes needed
2. **Phase 4 (set_state)** - Enables testing of phases 1-2
3. **Phase 2 (spawnedFromSiphon)** - Enemy observation addition
4. **Phase 1 (siphonCenter)** - Cell observation addition

This order minimizes risk by:
- Doing the simplest change first
- Enabling testing infrastructure before the features being tested
- Keeping observation space changes together at the end
