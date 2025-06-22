#!/usr/bin/env bash
# Enhanced wrapper script to run Adaptive Graph of Thoughts MCP Inspector tests
# Fixes timeout issues and improves process management for MCP testing
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
MODE="${1:-all}"
LOG_FILE="$PROJECT_ROOT/mcp_inspector_test.log"
MAX_RETRIES=2
STDIO_TIMEOUT=180
HTTP_TIMEOUT=120

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to validate environment
validate_environment() {
    log "🔍 Validating environment for MCP Inspector tests..."

    # Check for required commands
    local missing_commands=()

    if ! command_exists python3; then
        missing_commands+=("python3")
    fi
    if ! command_exists node; then
        missing_commands+=("node")
    fi
    if ! command_exists npm; then
        missing_commands+=("npm")
    fi
    if ! command_exists curl; then
        missing_commands+=("curl")
    fi

    if [ ${#missing_commands[@]} -ne 0 ]; then
        error "Missing required commands: ${missing_commands[*]}"
        exit 1
    fi

    # Check Node.js version
    local node_version
    node_version=$(node --version | sed 's/v//')
    local required_version="18.0.0"

    if ! command_exists npx; then
        error "npx not found. Please ensure Node.js is properly installed."
        exit 1
    fi

    # Check Python version
    local python_version
    python_version=$(python3 --version | awk '{print $2}')
    log "Python version: $python_version"
    log "Node.js version: $node_version"

    # Validate project structure
    if [[ ! -f "$PROJECT_ROOT/src/adaptive_graph_of_thoughts/main_stdio.py" ]]; then
        error "STDIO server file not found. Please ensure project is properly set up."
        exit 1
    fi
    if [[ ! -f "$PROJECT_ROOT/scripts/mcp_inspector_executor.py" ]]; then
        error "MCP inspector executor not found. Please ensure project is properly set up."
        exit 1
    fi

    success "Environment validation passed"
}

# Function to install MCP Inspector
install_mcp_inspector() {
    log "🔧 Installing/updating MCP Inspector..."
    cd "$PROJECT_ROOT"

    # Create package.json if it doesn't exist
    if [[ ! -f "package.json" ]]; then
        log "Creating package.json for MCP testing dependencies..."
        cat > package.json << 'EOF'
{
  "name": "adaptive-graph-of-thoughts-mcp-testing",
  "version": "1.0.0",
  "description": "MCP testing dependencies for Adaptive Graph of Thoughts",
  "private": true,
  "scripts": {
    "test:mcp:stdio": "python3 scripts/mcp_inspector_executor.py stdio",
    "test:mcp:http": "python3 scripts/mcp_inspector_executor.py http",
    "test:mcp:all": "python3 scripts/mcp_inspector_executor.py all"
  },
  "devDependencies": {
    "@modelcontextprotocol/inspector": "^0.1.0"
  },
  "engines": {
    "node": ">=18.0.0"
  }
}
EOF
    fi

    # Install MCP Inspector with retry logic
    local install_attempts=0
    local max_install_attempts=3

    while [ $install_attempts -lt $max_install_attempts ]; do
        if npm install @modelcontextprotocol/inspector --no-save --silent; then
            success "MCP Inspector installed successfully"
            break
        else
            install_attempts=$((install_attempts + 1))
            if [ $install_attempts -lt $max_install_attempts ]; then
                warning "Install attempt $install_attempts failed, retrying..."
                sleep 2
            else
                error "Failed to install MCP Inspector after $max_install_attempts attempts"
                exit 1
            fi
        fi
    done

    # Verify installation
    if [[ ! -f "node_modules/.bin/mcp-inspector" ]]; then
        error "MCP Inspector installation verification failed"
        exit 1
    fi

    # Add to PATH
    export PATH="$PROJECT_ROOT/node_modules/.bin:$PATH"
    log "Added local node_modules to PATH"
}

# Function to cleanup background processes
cleanup() {
    log "🧹 Cleaning up background processes..."

    # Kill any running uvicorn processes from this project
    pkill -f "uvicorn.*adaptive_graph_of_thoughts" || true

    # Kill any hanging Python MCP processes
    pkill -f "python.*main_stdio.py" || true

    # Wait a moment for processes to cleanup
    sleep 2
    log "Cleanup completed"
}

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 1  # Port is occupied
    else
        return 0  # Port is available
    fi
}

# Function to wait for server to be ready
wait_for_server() {
    local url=$1
    local timeout=$2
    local start_time
    start_time=$(date +%s)

    log "⏳ Waiting for server at $url to be ready (timeout: ${timeout}s)..."
    while true; do
        local current_time
        current_time=$(date +%s)
        local elapsed=$((current_time - start_time))

        if [ $elapsed -ge $timeout ]; then
            error "Server failed to start within $timeout seconds"
            return 1
        fi
        if curl -s --max-time 5 "$url" >/dev/null 2>&1; then
            success "Server is ready at $url"
            return 0
        fi
        sleep 2
    done
}

# Trap to ensure cleanup on exit
trap cleanup EXIT INT TERM

# Main execution function
main() {
    log "🚀 Starting Enhanced MCP Inspector Tests"
    log "Mode: $MODE"
    log "Project Root: $PROJECT_ROOT"
    log "Log File: $LOG_FILE"

    # Clear previous log
    > "$LOG_FILE"

    # Validate environment
    validate_environment

    # Install dependencies
    install_mcp_inspector

    # Check if HTTP port is available when testing HTTP mode
    if [[ "$MODE" == "http" || "$MODE" == "all" ]]; then
        if ! check_port 8000; then
            warning "Port 8000 is already in use. Attempting to free it..."
            cleanup
            sleep 3
            if ! check_port 8000; then
                error "Port 8000 is still occupied. Please free it manually."
                exit 1
            fi
        fi
    fi

    # Set environment variables for timeouts
    export MCP_STDIO_TIMEOUT=$STDIO_TIMEOUT
    export MCP_HTTP_TIMEOUT=$HTTP_TIMEOUT
    export MCP_INSPECTOR_RETRIES=$MAX_RETRIES

    log "Environment variables set:"
    log "  MCP_STDIO_TIMEOUT=$MCP_STDIO_TIMEOUT"
    log "  MCP_HTTP_TIMEOUT=$MCP_HTTP_TIMEOUT"
    log "  MCP_INSPECTOR_RETRIES=$MCP_INSPECTOR_RETRIES"

    # Run the enhanced Python testing script
    log "🔬 Executing MCP Inspector tests..."
    if python3 "$PROJECT_ROOT/scripts/mcp_inspector_executor.py" "$MODE" 2>&1 | tee -a "$LOG_FILE"; then
        success "🎉 All MCP Inspector tests completed successfully!"
        log "📋 Test logs saved to: $LOG_FILE"
        exit 0
    else
        error "❌ Some MCP Inspector tests failed!"
        log "📋 Check test logs at: $LOG_FILE"
        exit 1
    fi
}

# Validate script arguments
if [[ "$MODE" != "http" && "$MODE" != "stdio" && "$MODE" != "all" ]]; then
    error "Invalid mode: $MODE"
    echo "Usage: $0 [http|stdio|all]"
    echo "  http  - Test HTTP transport only"
    echo "  stdio - Test STDIO transport only"
    echo "  all   - Test both transports (default)"
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT"

# Run main function
main "$@"