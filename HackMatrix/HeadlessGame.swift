import Foundation

// Simplified wrapper for ML training - no UI, instant turn processing
class HeadlessGame {
    var gameState: GameState  // Internal for observation building

    init() {
        self.gameState = GameState()
    }

    // Reset to new game
    func reset() -> GameObservation {
        gameState = GameState()
        return ObservationBuilder.build(from: gameState)
    }

    /// Build info dict from ActionResult - used by both HeadlessGame and VisualGameController
    static func buildInfoDict(from result: GameState.ActionResult) -> [String: Any] {
        var info: [String: Any] = [:]

        if !result.success {
            info["invalid_action"] = true
        }
        if result.stageAdvanced {
            info["stage_complete"] = true
        }
        if result.playerDied {
            info["death"] = true
        }

        info["reward_breakdown"] = [
            "stage": result.rewardBreakdown.stageCompletion,
            "score": result.rewardBreakdown.scoreGain,
            "kills": result.rewardBreakdown.kills,
            "dataSiphon": result.rewardBreakdown.dataSiphonCollected,
            "distance": result.rewardBreakdown.distanceShaping,
            "victory": result.rewardBreakdown.victory,
            "death": result.rewardBreakdown.deathPenalty,
            "resourceGain": result.rewardBreakdown.resourceGain,
            "resourceHolding": result.rewardBreakdown.resourceHolding,
            "damagePenalty": result.rewardBreakdown.damagePenalty,
            "hpRecovery": result.rewardBreakdown.hpRecovery,
            "siphonQuality": result.rewardBreakdown.siphonQuality,
            "programWaste": result.rewardBreakdown.programWaste,
            "siphonDeathPenalty": result.rewardBreakdown.siphonDeathPenalty
        ]

        return info
    }

    // Execute action and advance game state (including enemy turn)
    // Returns: (observation, reward, isDone, info)
    func step(actionIndex: Int) -> (GameObservation, Double, Bool, [String: Any]) {
        guard let action = GameAction.fromIndex(actionIndex) else {
            fatalError("Invalid action index: \(actionIndex)")
        }

        // Process action (handles player action + enemy turn)
        let result: GameState.ActionResult = gameState.tryExecuteAction(action)

        // Build info dict using shared helper
        let info = HeadlessGame.buildInfoDict(from: result)

        // Determine if episode is done
        var isDone = false
        if !result.success {
            isDone = true
            infoLog("HeadlessGame", "❌ Invalid action \(action) attempted - terminating episode")
        }
        if result.stageAdvanced {
            isDone = result.gameWon
            if isDone {
                infoLog("Completed the game! With points: \(gameState.player.score)")
            }
        }
        if result.playerDied {
            isDone = true
        }

        let observation = ObservationBuilder.build(from: gameState)
        let reward = result.rewardBreakdown.total

        if result.stageAdvanced {
            debugLog("Advanced to stage \(observation.stage)")
        } else {
            debugLog("Step \(String(describing: action)) -> reward: \(String(format: "%.3f", reward)), done: \(isDone), stage: \(observation.stage), credits: \(gameState.player.credits), energy: \(gameState.player.energy)")
        }

        return (observation, reward, isDone, info)
    }

    // Get valid actions based on current state
    func getValidActions() -> [GameAction] {
        let actions = gameState.getValidActions()
        let indices = actions.map { $0.toIndex() }
        debugLog(
            "HeadlessGame",
            "Valid actions: \(actions.map { String(describing: $0) }) → indices: \(indices)")
        return actions
    }

}
