// Swift program with Foundation to test if Foundation causes the leak
import Foundation

// Static response as C string
let response: StaticString = "{\"observation\":{\"stage\":1}}\n"

func main() {
    fputs("Swift+Foundation test starting\n", stderr)

    let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: 256)
    defer { buffer.deallocate() }

    while true {
        let bytesRead = read(STDIN_FILENO, buffer, 256)
        if bytesRead <= 0 { break }

        _ = write(STDOUT_FILENO, response.utf8Start, response.utf8CodeUnitCount)
    }

    fputs("Swift+Foundation test done\n", stderr)
}

main()
