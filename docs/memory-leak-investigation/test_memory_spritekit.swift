// Swift program with SwiftUI + SpriteKit to test if SpriteKit causes the leak
import SwiftUI
import SpriteKit

// Static response as C string
let response: StaticString = "{\"observation\":{\"stage\":1}}\n"

@main
struct TestApp: App {
    init() {
        fputs("Swift+SwiftUI+SpriteKit test starting\n", stderr)
        fflush(stderr)

        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: 256)
        defer { buffer.deallocate() }

        while true {
            let bytesRead = read(STDIN_FILENO, buffer, 256)
            if bytesRead <= 0 { break }

            _ = write(STDOUT_FILENO, response.utf8Start, response.utf8CodeUnitCount)
        }

        fputs("Swift+SwiftUI+SpriteKit test done\n", stderr)
        fflush(stderr)
        exit(0)
    }

    var body: some Scene {
        WindowGroup {
            Text("Test")
        }
    }
}
