// Unified entry point for both macOS and Linux builds.
// - macOS: Launches SwiftUI GUI app
// - Linux: Runs headless CLI for ML training

import Foundation
import HackMatrixCore

#if canImport(SwiftUI)
import SwiftUI

// macOS: Launch the SwiftUI app
HackApp.main()
#else
// Linux: Run headless CLI only
let cli = HeadlessGameCLI()
cli.run()
#endif
