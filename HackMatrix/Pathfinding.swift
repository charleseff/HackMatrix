import Foundation

struct Pathfinding {
    /// Find the next move for an enemy to get closer to the player using BFS
    /// When multiple moves are equally good, applies tie-breaking based on enemy type:
    /// - Cryptogs prefer moves that keep them hidden (not in same row/col as player)
    /// - Glitches prefer moves on top of blocks
    /// - Other enemies pick randomly
    static func findNextMove(
        from start: (row: Int, col: Int),
        to target: (row: Int, col: Int),
        grid: Grid,
        canMoveOnBlocks: Bool,
        occupiedPositions: Set<String>,
        enemyType: EnemyType? = nil
    ) -> (row: Int, col: Int)? {

        // Try pathfinding with enemy collision avoidance first
        let candidateMoves = findPath(from: start, to: target, grid: grid, canMoveOnBlocks: canMoveOnBlocks, occupiedPositions: occupiedPositions)
        if !candidateMoves.isEmpty {
            // Apply tie-breaking if multiple moves available
            let chosenKey = selectBestMove(from: candidateMoves, enemyType: enemyType, playerPosition: target, grid: grid)
            let parts = chosenKey.split(separator: ",")
            return (Int(parts[0])!, Int(parts[1])!)
        }

        // If no path found (likely blocked by enemies), try again ignoring enemy positions
        // This ensures enemies still try to move closer even when blocked
        let candidateMovesIgnoringEnemies = findPath(from: start, to: target, grid: grid, canMoveOnBlocks: canMoveOnBlocks, occupiedPositions: Set<String>())

        // Filter out moves that would actually place us on an enemy
        let validMoves = candidateMovesIgnoringEnemies.filter { moveKey in
            !occupiedPositions.contains(moveKey)
        }

        if !validMoves.isEmpty {
            // Apply tie-breaking if multiple moves available
            let chosenKey = selectBestMove(from: validMoves, enemyType: enemyType, playerPosition: target, grid: grid)
            let parts = chosenKey.split(separator: ",")
            return (Int(parts[0])!, Int(parts[1])!)
        }

        // No path found at all - stay in place
        return nil
    }

    /// Select best move from candidates based on enemy type preferences
    private static func selectBestMove(from candidates: Set<String>, enemyType: EnemyType?, playerPosition: (row: Int, col: Int), grid: Grid) -> String {
        guard candidates.count > 1, let type = enemyType else {
            // Only one option or no type info - return random
            return candidates.randomElement()!
        }

        let candidateArray = Array(candidates)

        switch type {
        case .cryptog:
            // Prefer moves that keep Cryptog hidden (not in same row/col as player)
            let hiddenMoves = candidateArray.filter { moveKey in
                let parts = moveKey.split(separator: ",")
                let row = Int(parts[0])!
                let col = Int(parts[1])!
                return row != playerPosition.row && col != playerPosition.col
            }
            return hiddenMoves.isEmpty ? candidates.randomElement()! : hiddenMoves.randomElement()!

        case .glitch:
            // Prefer moves on top of blocks
            let blockMoves = candidateArray.filter { moveKey in
                let parts = moveKey.split(separator: ",")
                let row = Int(parts[0])!
                let col = Int(parts[1])!
                return grid.cells[row][col].hasBlock
            }
            return blockMoves.isEmpty ? candidates.randomElement()! : blockMoves.randomElement()!

        default:
            // No special preference - random selection
            return candidates.randomElement()!
        }
    }

    /// Internal pathfinding implementation using BFS
    /// Returns all candidate first moves that lead to the shortest path
    private static func findPath(
        from start: (row: Int, col: Int),
        to target: (row: Int, col: Int),
        grid: Grid,
        canMoveOnBlocks: Bool,
        occupiedPositions: Set<String>
    ) -> Set<String> {

        // BFS to find shortest path(s)
        var queue: [(pos: (Int, Int), path: [(Int, Int)])] = [(start, [start])]
        var visited = Set<String>()
        visited.insert("\(start.0),\(start.1)")

        var shortestPathLength: Int?
        var candidateFirstMoves: Set<String> = []

        while !queue.isEmpty {
            let (currentPos, path) = queue.removeFirst()

            // If we've found paths and this one is longer, stop searching
            if let shortest = shortestPathLength, path.count > shortest {
                break
            }

            // Check if we reached the target
            if currentPos.0 == target.0 && currentPos.1 == target.1 {
                if path.count > 1 {
                    let firstMove = path[1]

                    if shortestPathLength == nil {
                        // First path found
                        shortestPathLength = path.count
                        candidateFirstMoves.insert("\(firstMove.0),\(firstMove.1)")
                    } else if path.count == shortestPathLength {
                        // Another path of same length - add its first move
                        candidateFirstMoves.insert("\(firstMove.0),\(firstMove.1)")
                    }
                }
                continue // Don't expand from target
            }

            // Try all 4 directions
            for direction in Direction.allCases {
                let offset = direction.offset
                let newRow = currentPos.0 + offset.row
                let newCol = currentPos.1 + offset.col

                guard grid.isValidPosition(row: newRow, col: newCol) else { continue }

                let posKey = "\(newRow),\(newCol)"
                guard !visited.contains(posKey) else { continue }

                // Check if position is blocked
                let cell = grid.cells[newRow][newCol]
                if cell.hasBlock && !canMoveOnBlocks {
                    continue
                }

                // Check if occupied by another enemy (but allow moving to player's position)
                if !(newRow == target.0 && newCol == target.1) && occupiedPositions.contains(posKey) {
                    continue
                }

                visited.insert(posKey)
                var newPath = path
                newPath.append((newRow, newCol))
                queue.append(((newRow, newCol), newPath))
            }
        }

        // Return all candidate first moves
        return candidateFirstMoves
    }

    /// Get all occupied positions (for collision detection)
    static func getOccupiedPositions(enemies: [Enemy], excludingId: UUID? = nil) -> Set<String> {
        var occupied = Set<String>()
        for enemy in enemies {
            if let excludeId = excludingId, enemy.id == excludeId {
                continue
            }
            occupied.insert("\(enemy.row),\(enemy.col)")
        }
        return occupied
    }

    /// Calculate shortest path distance from a position to a target using BFS
    /// Returns nil if no path exists
    static func findDistance(
        from start: (row: Int, col: Int),
        to target: (row: Int, col: Int),
        grid: Grid
    ) -> Int? {
        if start.0 == target.0 && start.1 == target.1 {
            return 0
        }

        var queue: [(pos: (Int, Int), distance: Int)] = [(start, 0)]
        var visited = Set<String>()
        visited.insert("\(start.0),\(start.1)")

        while !queue.isEmpty {
            let (currentPos, distance) = queue.removeFirst()

            for direction in Direction.allCases {
                let offset = direction.offset
                let newRow = currentPos.0 + offset.row
                let newCol = currentPos.1 + offset.col

                guard grid.isValidPosition(row: newRow, col: newCol) else { continue }

                let posKey = "\(newRow),\(newCol)"
                guard !visited.contains(posKey) else { continue }

                if newRow == target.0 && newCol == target.1 {
                    return distance + 1
                }

                let cell = grid.cells[newRow][newCol]
                if cell.hasBlock { continue }

                visited.insert(posKey)
                queue.append(((newRow, newCol), distance + 1))
            }
        }

        return nil
    }
}
