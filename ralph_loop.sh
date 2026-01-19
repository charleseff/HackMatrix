#!/bin/bash
# Usage: ./loop.sh [mode] [max_iterations] --completion-promise TEXT
# Examples:
#   ./loop.sh --completion-promise "ALL_DONE"                              # Build mode, unlimited iterations
#   ./loop.sh 20 --completion-promise "ALL_DONE"                           # Build mode, max 20 iterations
#   ./loop.sh plan --completion-promise "PLAN_DONE"                        # Plan mode, unlimited iterations
#   ./loop.sh plan 5 --completion-promise "PLAN_DONE"                      # Plan mode, max 5 iterations

# Parse arguments
MODE="build"
PROMPT_FILE="PROMPT_build.md"
MAX_ITERATIONS=0
COMPLETION_PROMISE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        plan)
            MODE="plan"
            PROMPT_FILE="PROMPT_plan.md"
            shift
            ;;
        --completion-promise)
            COMPLETION_PROMISE="$2"
            shift 2
            ;;
        [0-9]*)
            MAX_ITERATIONS="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Verify --completion-promise is provided (required)
if [ -z "$COMPLETION_PROMISE" ]; then
    echo "Error: --completion-promise is required"
    echo ""
    echo "Usage: ./loop.sh [mode] [max_iterations] --completion-promise TEXT"
    echo ""
    echo "Examples:"
    echo "  ./loop.sh --completion-promise \"ALL_DONE\""
    echo "  ./loop.sh 20 --completion-promise \"ALL_DONE\""
    echo "  ./loop.sh plan --completion-promise \"PLAN_DONE\""
    echo "  ./loop.sh plan 5 --completion-promise \"PLAN_DONE\""
    exit 1
fi

ITERATION=0
CURRENT_BRANCH=$(git branch --show-current)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Mode:              $MODE"
echo "Prompt:            $PROMPT_FILE"
echo "Branch:            $CURRENT_BRANCH"
[ $MAX_ITERATIONS -gt 0 ] && echo "Max Iterations:    $MAX_ITERATIONS"
echo "Completion Promise: '$COMPLETION_PROMISE'"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Verify prompt file exists
if [ ! -f "$PROMPT_FILE" ]; then
    echo "Error: $PROMPT_FILE not found"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Export COMPLETION_PROMISE for envsubst
export COMPLETION_PROMISE

while true; do
    if [ $MAX_ITERATIONS -gt 0 ] && [ $ITERATION -ge $MAX_ITERATIONS ]; then
        echo "Reached max iterations: $MAX_ITERATIONS"
        break
    fi

    # Capture output to log file for completion detection
    OUTPUT_FILE="logs/claude_output_iter${ITERATION}_$(date +%s).log"

    # Run Ralph iteration with selected prompt
    # -p: Headless mode (non-interactive, reads from stdin)
    # --dangerously-skip-permissions: Auto-approve all tool calls (YOLO mode)
    # --output-format=text: Human-readable output
    # --model opus: Primary agent uses Opus for complex reasoning (task selection, prioritization)
    #               Can use 'sonnet' in build mode for speed if plan is clear and tasks well-defined
    # --verbose: Detailed execution logging
    # stdbuf -oL: Line-buffered output so we can see progress in real-time
    # Use envsubst to substitute COMPLETION_PROMISE variable into prompt
    envsubst < "$PROMPT_FILE" | stdbuf -oL claude -p \
        --dangerously-skip-permissions \
        --output-format=text \
        --model opus \
        --verbose 2>&1 | tee "$OUTPUT_FILE"

    # Check for completion promise in output
    if grep -q "$COMPLETION_PROMISE" "$OUTPUT_FILE"; then
        echo ""
        echo "✅ Completion promise detected: '$COMPLETION_PROMISE'"
        echo "Project work finished successfully."
        break
    fi

    # Push changes after each iteration
    git push origin "$CURRENT_BRANCH" || {
        echo "Failed to push. Creating remote branch..."
        git push -u origin "$CURRENT_BRANCH"
    }

    ITERATION=$((ITERATION + 1))
    echo -e "\n\n======================== LOOP $ITERATION ========================\n"
done