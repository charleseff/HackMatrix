# Unified Action Processing Refactor

## Goal

Single source of truth for game action processing. Both headless and GUI modes use the same `processAction()` method in GameState. No duplication.

## Current Problems

1. **HeadlessGame.step()** has switch statement on GameAction
2. **GameScene** has inline action handling in keyDown/mouseDown
3. **Different code paths** = potential bugs and maintenance burden
4. **HeadlessGame doesn't process turns** (critical bug)
5. **GameScene duplicates turn initiation** 5 times

## Solution Architecture

```
Input → GameAction → processAction() → ActionResult → Output
```

**GameState:**
- `processAction(_ action: GameAction) -> ActionResult` - single entry point
- Returns what happened, what needs to happen next
- Both modes call this, then handle output differently

**HeadlessGame:**
- Parse stdin → GameAction
- Call processAction()
- Call executeTurn() if needed
- Return JSON observation

**GameScene:**
- Parse keyboard/mouse → GameAction
- Call processAction()
- Animate based on ActionResult
- Call executeTurnWithAnimation() if needed

---

## Phase 1: Add processAction to GameState

### 1.1 Add ActionResult struct

**File:** GameState.swift
**Location:** After line 1567 (after GameStateSnapshot)

```swift
struct ActionResult {
    let success: Bool
    let exitReached: Bool
    let shouldAdvanceTurn: Bool  // false for undo, true for most actions
    let affectedPositions: [(Int, Int)]  // for explosion animations

    init(success: Bool = true, exitReached: Bool = false,
         shouldAdvanceTurn: Bool = true, affectedPositions: [(Int, Int)] = []) {
        self.success = success
        self.exitReached = exitReached
        self.shouldAdvanceTurn = shouldAdvanceTurn
        self.affectedPositions = affectedPositions
    }
}
```

### 1.2 Implement processAction method

**File:** GameState.swift
**Location:** After canExecuteProgram (around line 830)

```swift
/// Process a game action and return the result
/// This is the single entry point for all action processing
func processAction(_ action: GameAction) -> ActionResult {
    switch action {
    case .move(let direction):
        let result = tryMove(direction: direction)
        return ActionResult(
            success: !result.blocked,
            exitReached: result.exitReached,
            shouldAdvanceTurn: !result.exitReached
        )

    case .siphon:
        let success = performSiphon()
        return ActionResult(
            success: success,
            shouldAdvanceTurn: success
        )

    case .program(let programType):
        let execResult = executeProgram(programType)
        return ActionResult(
            success: execResult.success,
            shouldAdvanceTurn: programType != .undo && execResult.success,
            affectedPositions: execResult.affectedPositions
        )
    }
}
```

**Test:** Unit test that processAction returns expected results for various actions

---

## Phase 2: Update HeadlessGame

### 2.1 Add executeTurn to GameState

**File:** GameState.swift
**Location:** After processAction (around line 865)

```swift
/// Execute a full game turn (synchronous, no animations)
/// Returns turn progression info
func executeTurn() {
    advanceTurn()
}
```

**Note:** For now, just wrap advanceTurn(). Later we can enhance this to return TurnResult for animation info.

### 2.2 Simplify HeadlessGame.step()

**File:** HeadlessGame.swift
**Location:** Replace entire step() method (lines 19-76)

**Before:** 58 lines with switch statement
**After:** ~25 lines calling processAction

```swift
func step(action: GameAction) -> (GameObservation, Double, Bool, [String: Any]) {
    let oldScore = gameState.player.score
    var isDone = false
    var info: [String: Any] = [:]

    // Process the action
    let result = gameState.processAction(action)

    if !result.success {
        info["invalid_action"] = true
    }

    if result.exitReached {
        let continues = gameState.completeStage()
        isDone = !continues
        info["stage_complete"] = true
    }

    // Advance turn if action succeeded and requires it
    if result.shouldAdvanceTurn {
        gameState.executeTurn()
    }

    // Check if player died
    if gameState.player.health == .dead {
        isDone = true
        info["death"] = true
    }

    // Calculate reward
    let scoreDelta = Double(gameState.player.score - oldScore)
    var reward = scoreDelta * 0.01

    if isDone {
        if gameState.player.health == .dead {
            reward = 0.0
        } else {
            reward = Double(gameState.player.score) * 10.0
        }
    }

    let observation = getObservation()
    return (observation, reward, isDone, info)
}
```

**Tests:**
- Turn counter increments
- Enemies move
- Transmissions spawn
- Python wrapper still works (test_env.py)

---

## Phase 3: Update GameScene

### 3.1 Add helper methods

**File:** GameScene.swift
**Location:** After updateDisplay (around line 920)

```swift
/// Handle a game action and update display/animations
private func handleAction(_ action: GameAction) {
    let result = gameState.processAction(action)

    if !result.success {
        // Could add error feedback here
        return
    }

    if result.exitReached {
        handleStageComplete()
        return
    }

    // Handle explosion animations for affected positions
    if !result.affectedPositions.isEmpty {
        isAnimating = true
        animateExplosions(at: result.affectedPositions) { [weak self] in
            self?.updateDisplay()

            // After explosion animation, advance turn if needed
            if result.shouldAdvanceTurn {
                self?.executeTurnWithAnimation()
            } else {
                self?.isAnimating = false
            }
        }
        return
    }

    // No explosion animations needed
    if result.shouldAdvanceTurn {
        executeTurnWithAnimation()
    } else {
        // Just update display (e.g., for undo)
        updateDisplay()
    }
}

/// Execute a turn with animations
private func executeTurnWithAnimation() {
    isAnimating = true
    gameState.executeTurn()
    updateDisplay()

    // Animate enemy movements
    enemiesWhoAttacked = Set<UUID>()
    animateEnemySteps(currentStep: 0)

    // Note: animateEnemySteps eventually calls finalizeAnimatedTurn
    // and sets isAnimating = false when complete
}
```

### 3.2 Refactor keyDown

**File:** GameScene.swift
**Location:** Replace keyDown implementation (lines 634-723)

**Before:** 90 lines with inline action handling
**After:** ~30 lines converting input to GameAction

```swift
override func keyDown(with event: NSEvent) {
    // Block input during animations
    guard !isAnimating else { return }

    // Handle restart
    if isGameOver && event.keyCode == 15 { // R key
        restartGame()
        return
    }

    guard gameState.player.health != .dead else { return }

    // Convert keyboard input to GameAction
    var action: GameAction? = nil

    if event.keyCode == 1 { // S key
        action = .siphon
    } else if let programType = getProgramForKeyCode(event.keyCode) {
        action = .program(programType)
    } else {
        // Arrow keys
        switch event.keyCode {
        case 126: action = .move(.up)
        case 125: action = .move(.down)
        case 123: action = .move(.left)
        case 124: action = .move(.right)
        default: return
        }
    }

    if let action = action {
        handleAction(action)
    }
}
```

### 3.3 Refactor mouseDown

**File:** GameScene.swift
**Location:** Replace mouseDown implementation (lines 725-878)

**Before:** 154 lines with inline action handling
**After:** ~30 lines converting input to GameAction

```swift
override func mouseDown(with event: NSEvent) {
    guard !isAnimating && !isGameOver else { return }

    let location = event.location(in: self)
    let clickedNodes = nodes(at: location)

    // Check if a program button was clicked
    for node in clickedNodes {
        var checkNode: SKNode? = node

        // Check up to 2 levels (node or parent could be the button)
        for _ in 0..<2 {
            if let nodeName = checkNode?.name, nodeName.hasPrefix("program_") {
                let programName = String(nodeName.dropFirst("program_".count))

                if let programType = ProgramType.allCases.first(where: { $0.rawValue == programName }) {
                    handleAction(.program(programType))
                }
                return
            }
            checkNode = checkNode?.parent
        }
    }

    // Check if a grid cell was clicked
    for (rowIdx, row) in cellNodes.enumerated() {
        for (colIdx, cellNode) in row.enumerated() {
            if cellNode.contains(location) {
                handleCellClick(row: rowIdx, col: colIdx)
                return
            }
        }
    }
}
```

**Tests:**
- All keyboard shortcuts work
- All mouse clicks work
- Animations play correctly
- Turn advancement works
- No visual regressions

---

## Phase 4: Remove Old Turn Processing (Cleanup)

Once everything works with the new system:

### 4.1 Remove beginAnimatedTurn/finalizeAnimatedTurn

**File:** GameState.swift

These methods are only used by GameScene's animation system. We can keep them for now since executeTurnWithAnimation still needs step-by-step enemy animation.

**Action:** Keep for now, refactor later if needed

### 4.2 Verify advanceTurn still used

**File:** GameState.swift

`advanceTurn()` is now called by `executeTurn()`. Make sure no other direct calls remain.

---

## Benefits

1. **Single source of truth** - All game logic in GameState.processAction()
2. **No duplication** - HeadlessGame and GameScene share same code path
3. **Testable** - processAction() is a pure function with clear inputs/outputs
4. **Maintainable** - Game logic changes only touch one place
5. **Fixes critical bug** - HeadlessGame now processes turns correctly

---

## Migration Risk

**Low risk** - Each phase is independently testable:

1. Phase 1: Pure addition, no behavior changes
2. Phase 2: HeadlessGame changes, test with Python wrapper
3. Phase 3: GameScene changes, test GUI gameplay

Can rollback any phase if issues arise.

---

## Files Modified

1. **GameState.swift** (+40 lines)
   - ActionResult struct
   - processAction() method
   - executeTurn() method

2. **HeadlessGame.swift** (-35 lines)
   - Simplified step() using processAction

3. **GameScene.swift** (-120 lines)
   - handleAction() helper
   - executeTurnWithAnimation() helper
   - Simplified keyDown/mouseDown

**Net change:** -115 lines, massively improved architecture
