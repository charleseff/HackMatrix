# Memory Leak Investigation - Headless CLI Mode

**Date:** December 2024
**Branch:** `memory-leak`
**Status:** RESOLVED

## Summary

A memory leak was discovered during ML training where the Swift process would grow ~1.2 MB per episode (~12 KB per step), eventually consuming gigabytes of memory during long training runs.

**Root Cause:** Foundation's JSON/String operations create Objective-C bridged objects that were not being released in the tight stdin reading loop.

**Fix:** Wrap the command handling loop in `autoreleasepool { }` in `GameCommandProtocol.swift`.

## Symptoms

- Memory growth of ~1.2 MB per episode during headless training
- Process RSS growing from ~8 MB to 100+ MB over 100 episodes
- No corresponding growth in Python process (isolated to Swift)

## Investigation Process

### Systematic Elimination

The investigation used a systematic elimination approach, adding skip flags to isolate components:

| Component Tested | Skip Flag | Result |
|-----------------|-----------|--------|
| Enemy turn processing | `GameState.skipEnemyTurn` | Still leaked |
| Observation building | `ObservationBuilder.skipBuild` | Still leaked |
| All game logic | `HeadlessGame.skipGameLogic` | Still leaked |
| JSON parsing | Manual string checks | Still leaked |
| Dictionary creation | Pre-created responses | Still leaked |
| JSONSerialization | Pre-serialized Data | Still leaked |
| readLine() | Raw `Darwin.read()` | Still leaked |
| Full executor | `StdinCommandReader.skipExecutor` | **NO LEAK** |

### Framework Isolation Tests

Standalone test programs were created to isolate which framework caused the leak:

| Test Program | Frameworks | Result |
|-------------|------------|--------|
| `test_memory_c` | Pure C (libc) | No leak |
| `test_memory_swift` | Swift + Darwin | No leak |
| `test_memory_foundation` | Swift + Foundation | No leak |
| `test_memory_swiftui` | Swift + SwiftUI | No leak |
| `test_memory_spritekit` | Swift + SwiftUI + SpriteKit | No leak |

These tests showed that the frameworks themselves don't leak - the issue was in how the application code used them.

### The Breakthrough

When `skipExecutor=true` (bypassing all Swift code with pure C-style I/O), no leak occurred. When `skipExecutor=false` (using the normal code path), the leak returned.

The critical difference was the **autoreleasepool**.

### Verification

```
Testing WITHOUT autoreleasepool...
Initial: 8.3 MB
Episode  10: 18.8 MB (growth: +10.5)
Episode  50: 54.8 MB (growth: +46.5)
Episode 100: 99.7 MB (growth: +91.4)
```

```
Testing WITH autoreleasepool...
Initial: 8.3 MB
Episode  10: 10.0 MB (growth: +1.7)
Episode  50: 10.2 MB (growth: +1.9)
Episode 100: 10.2 MB (growth: +1.9)
```

## Root Cause Explanation

Foundation's operations like `line.data(using: .utf8)` and `JSONDecoder().decode()` create Objective-C bridged objects. These objects are added to the current autorelease pool and released when that pool drains.

In a typical macOS application, the main runloop periodically drains the autorelease pool. However, in a headless CLI that runs a tight `while let line = readLine()` loop, the runloop never gets a chance to drain the pool. Objects accumulate indefinitely.

### The Fix

```swift
// GameCommandProtocol.swift - start() method

while let line = readLine() {
    autoreleasepool {  // <-- CRITICAL: Forces release of bridged objects
        guard let data = line.data(using: .utf8),
              let command = try? JSONDecoder().decode(Command.self, from: data) else {
            sendError("Invalid command")
            return
        }
        handleCommand(command)
    }
}
```

The `autoreleasepool { }` block creates a local autorelease pool that drains at the end of each iteration, releasing the temporary Objective-C objects.

## Files Changed

- `HackMatrix/GameCommandProtocol.swift` - Added autoreleasepool wrapper with explanatory comment

## Test Files

The following test files were created during investigation and are preserved in the repository root for reference:

- `test_memory_c.c` / `test_memory_c` - Pure C stdin/stdout loop
- `test_memory_swift.swift` / `test_memory_swift` - Swift with Darwin only
- `test_memory_foundation.swift` / `test_memory_foundation` - Swift with Foundation
- `test_memory_swiftui.swift` / `test_memory_swiftui` - Swift with SwiftUI
- `test_memory_spritekit.swift` / `test_memory_spritekit` - Swift with SpriteKit
- `python/scripts/memory_test_c.py` - Python test harness for memory measurement

## Lessons Learned

1. **Autorelease pools matter in tight loops**: When running Swift code in a tight loop without a runloop (common in CLI tools), you must manually manage autorelease pools.

2. **Foundation operations create Objective-C objects**: Even "pure Swift" code using Foundation creates bridged Objective-C objects that need autorelease pool management.

3. **Systematic elimination is effective**: By adding skip flags to isolate components, we could quickly narrow down the source of the leak.

4. **Standalone test programs help isolate issues**: Creating minimal test programs that reproduce (or don't reproduce) the issue helps identify whether the problem is in framework code or application code.

## References

- [Apple Documentation: Using Autorelease Pool Blocks](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/MemoryMgmt/Articles/mmAutoreleasePools.html)
- [Swift Forums: Autorelease pools in Swift](https://forums.swift.org/t/autorelease-pools-in-swift/19876)
