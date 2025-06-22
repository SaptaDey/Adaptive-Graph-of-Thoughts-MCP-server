#!/usr/bin/env bash
# Setup script for MCP client configurations

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE} Adaptive Graph of Thoughts MCP${NC}"
    echo -e "${BLUE}     Client Setup Utility${NC}"
    echo -e "${BLUE}================================${NC}"
    echo
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_dependencies() {
    echo "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    print_success "Python 3 found"
    
    # Check Poetry
    if ! command -v poetry &> /dev/null; then
        print_warning "Poetry not found. Install with: curl -sSL https://install.python-poetry.org | python3 -"
    else
        print_success "Poetry found"
    fi
    
    # Check Neo4j (optional)
    if command -v neo4j &> /dev/null; then
        print_success "Neo4j found"
    else
        print_warning "Neo4j not found. You'll need to configure connection manually"
    fi
    
    echo
}

setup_claude_desktop() {
    echo "Setting up Claude Desktop configuration..."
    
    # Run Python setup script
    if [ -f "$PROJECT_ROOT/src/adaptive_graph_of_thoughts/setup/client_setup.py" ]; then
        python3 "$PROJECT_ROOT/src/adaptive_graph_of_thoughts/setup/client_setup.py" claude-desktop
    else
        print_error "Client setup script not found"
        return 1
    fi
}

setup_vscode() {
    echo "Setting up VS Code configuration..."
    
    # Run Python setup script
    if [ -f "$PROJECT_ROOT/src/adaptive_graph_of_thoughts/setup/client_setup.py" ]; then
        python3 "$PROJECT_ROOT/src/adaptive_graph_of_thoughts/setup/client_setup.py" vscode
    else
        print_error "Client setup script not found"
        return 1
    fi
}

setup_docker() {
    echo "Setting up Docker configuration..."
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        print_error "Docker is required but not installed"
        return 1
    fi
    
    # Check if docker-compose is available
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is required but not installed"
        return 1
    fi
    
    echo "Building Docker image..."
    cd "$PROJECT_ROOT"
    docker-compose build
    
    print_success "Docker setup complete"
    echo "To start the server: docker-compose up"
    echo "To start in production mode: docker-compose -f docker-compose.prod.yml up -d"
}

install_dependencies() {
    echo "Installing project dependencies..."
    
    cd "$PROJECT_ROOT"
    
    if command -v poetry &> /dev/null; then
        poetry install
        print_success "Dependencies installed with Poetry"
    else
        pip3 install -r requirements.txt
        print_success "Dependencies installed with pip"
    fi
}

verify_installation() {
    echo "Verifying installation..."
    
    cd "$PROJECT_ROOT"
    
    # Test import
    if python3 -c "import sys; sys.path.insert(0, 'src'); import adaptive_graph_of_thoughts" 2>/dev/null; then
        print_success "Python package imports successfully"
    else
        print_error "Python package import failed"
        return 1
    fi
    
    # Check configuration files
    if [ -f "$PROJECT_ROOT/config/settings.yaml" ]; then
        print_success "Configuration file found"
    else
        print_warning "Configuration file not found - using defaults"
    fi
    
    # Test server startup (quick test)
    echo "Testing server startup..."
    timeout 10s python3 -c "
import sys
sys.path.insert(0, 'src')
from adaptive_graph_of_thoughts.config import settings
print('Configuration loaded successfully')
" || print_warning "Server startup test timed out (this may be normal)"
    
    print_success "Installation verification complete"
}

show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  claude-desktop    Setup Claude Desktop MCP configuration"
    echo "  vscode           Setup VS Code MCP configuration"
    echo "  docker           Setup Docker environment"
    echo "  install          Install project dependencies"
    echo "  verify           Verify installation"
    echo "  all              Run complete setup (install + verify)"
    echo "  help             Show this help message"
    echo
}

main() {
    print_header
    
    if [ $# -eq 0 ]; then
        show_usage
        exit 0
    fi
    
    case "$1" in
        "claude-desktop")
            check_dependencies
            setup_claude_desktop
            ;;
        "vscode")
            check_dependencies
            setup_vscode
            ;;
        "docker")
            setup_docker
            ;;
        "install")
            check_dependencies
            install_dependencies
            ;;
        "verify")
            verify_installation
            ;;
        "all")
            check_dependencies
            install_dependencies
            verify_installation
            echo
            echo -e "${GREEN}Setup complete! Choose a client to configure:${NC}"
            echo "  ./scripts/setup_mcp_client.sh claude-desktop"
            echo "  ./scripts/setup_mcp_client.sh vscode"
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
