# Clean Architecture Refactor

## Goal

`processAction()` is THE function that advances the game until next user input. Both HeadlessGame and GUI call it. Enemy turn logic runs inside it. Returns data for GUI to animate.

**Key insight:** Game state advances synchronously, animation is just replay of what happened.

---

## Current Status

### Completed
- [x] Fix stepActive bug in beginAnimatedEnemyTurn
- [x] Move GameAction enum to GameState.swift (renamed `.move` → `.direction`)
- [x] Add `EnemyStepResult` struct
- [x] Add `ActionResult` struct with `playerDied` and `enemySteps`
- [x] Add `runEnemyTurn()` helper
- [x] Add `executeEnemyStepWithCapture()` method
- [x] Update `processAction()` to include enemy turn
- [x] Update `HeadlessGame.step()` to use `processAction()`
- [x] Refactor `moveEnemiesSimultaneously()` to filter enemies upfront

### Remaining
- [ ] Update GameScene to use `handleAction()` + `animateActionResult()`
  - This requires enriching `ActionResult` with player action details:
    - Whether player moved vs attacked
    - What was attacked (enemy/transmission position)
    - Player movement (from/to positions)
  - OR keep some pre-check logic in GameScene for player animations

---

## What's Done

### GameState.swift

**New structs:**
```swift
struct EnemyStepResult {
    let step: Int  // 0, 1, etc (virus moves twice per turn)
    let movements: [(enemyId: UUID, fromRow: Int, fromCol: Int, toRow: Int, toCol: Int)]
    let attacks: [(enemyId: UUID, damage: Int)]
}

struct ActionResult {
    let success: Bool
    let exitReached: Bool
    let playerDied: Bool
    let affectedPositions: [(row: Int, col: Int)]
    let enemySteps: [EnemyStepResult]
}
```

**processAction()** now handles player action + enemy turn:
```swift
func processAction(_ action: GameAction) -> ActionResult {
    // 1. Handle player action (direction/siphon/program)
    // 2. Run enemy turn if needed via runEnemyTurn()
    // 3. Return ActionResult with all data
}
```

**runEnemyTurn()** executes full enemy turn and captures step data for animation.

**executeEnemyStepWithCapture()** executes one enemy step and captures movements/attacks.

### HeadlessGame.swift

**step()** is now simple:
```swift
func step(action: GameAction) -> (GameObservation, Double, Bool, [String: Any]) {
    let result = gameState.processAction(action)
    // Handle result.success, result.exitReached, result.playerDied
    // Calculate reward
    return (observation, reward, isDone, info)
}
```

**Critical ML bug fixed:** Enemies now move during headless training!

---

## What Remains: GameScene Refactor

### Challenge

GameScene currently checks state *before* actions to determine animations:
- `handlePlayerMove()` calls `findTargetInLineOfFire()` before `tryMove()`
- Then animates attack or movement based on what was found

With `processAction()` doing everything at once, we need to either:

1. **Enrich ActionResult** with player action details:
   ```swift
   struct ActionResult {
       // ... existing fields ...
       let playerAction: PlayerActionResult?  // NEW
   }

   struct PlayerActionResult {
       let type: PlayerActionType  // .moved, .attacked, .siphoned, .program
       let fromPosition: (row: Int, col: Int)
       let toPosition: (row: Int, col: Int)?
       let targetPosition: (row: Int, col: Int)?  // for attacks
   }
   ```

2. **Keep pre-check logic** in GameScene for player animations, use `processAction()` only for enemy turn data

3. **Defer GameScene refactor** - HeadlessGame works correctly now, GUI can be updated later

### If We Continue

```swift
// GameScene simplified flow:
private func handleAction(_ action: GameAction) {
    let result = gameState.processAction(action)
    if !result.success { return }
    if result.exitReached { handleStageComplete(); return }
    animateActionResult(result)
}

private func animateActionResult(_ result: ActionResult) {
    // 1. Animate player action (from result.playerAction)
    // 2. Animate explosions (from result.affectedPositions)
    // 3. Animate enemy steps (from result.enemySteps)
}
```

---

## Benefits Achieved

- ✅ Single code path for game logic (`processAction()`)
- ✅ HeadlessGame is trivially simple
- ✅ Enemy turn bug fixed (enemies now move in ML training)
- ⏳ GUI animates from data (pending GameScene refactor)
