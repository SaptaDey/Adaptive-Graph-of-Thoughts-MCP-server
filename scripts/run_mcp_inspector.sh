#!/usr/bin/env bash
# Wrapper script to run Adaptive Graph of Thoughts MCP Inspector tests
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default test mode: all transports
MODE="${1:-all}"

# Ensure Node.js inspector is installed
echo "🔧 Installing MCP Inspector if necessary..."
cd "$PROJECT_ROOT"
npm install @modelcontextprotocol/inspector --no-save

# Add local node_modules to PATH
export PATH="$PROJECT_ROOT/node_modules/.bin:$PATH"

# Run the Python testing script with the chosen mode
echo "🚀 Running MCP Inspector tests with mode: $MODE"
python3 "$PROJECT_ROOT/scripts/mcp_inspector_executor.py" "$MODE"
