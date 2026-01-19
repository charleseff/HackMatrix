#!/bin/bash
# Filter stream-json output to readable text
# Usage: claude ... --output-format=stream-json | ./json2text.sh

while IFS= read -r line; do
    type=$(echo "$line" | jq -r '.type // empty' 2>/dev/null)

    case "$type" in
        system)
            subtype=$(echo "$line" | jq -r '.subtype // empty')
            if [ "$subtype" = "init" ]; then
                echo "=== Session started ==="
            fi
            ;;
        assistant)
            # Extract text content
            text=$(echo "$line" | jq -r '.message.content[]? | select(.type=="text") | .text // empty' 2>/dev/null)
            if [ -n "$text" ]; then
                echo "$text"
            fi

            # Extract tool use
            tool=$(echo "$line" | jq -r '.message.content[]? | select(.type=="tool_use") | "\(.name): \(.input | tostring | .[0:100])"' 2>/dev/null)
            if [ -n "$tool" ]; then
                echo "  → $tool"
            fi
            ;;
        result)
            # Final result
            result=$(echo "$line" | jq -r '.result // empty' 2>/dev/null)
            if [ -n "$result" ]; then
                echo ""
                echo "=== Result ==="
                echo "$result"
            fi
            ;;
    esac
done
