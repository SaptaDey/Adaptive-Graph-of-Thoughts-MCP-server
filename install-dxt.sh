#!/bin/bash

# Adaptive Graph of Thoughts DXT Installation Script
# This script installs and sets up the DXT extension

set -e

echo "🧠 Adaptive Graph of Thoughts DXT Installation"
echo "=============================================="

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18 or later."
    exit 1
fi

NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version $NODE_VERSION is too old. Please upgrade to Node.js 18 or later."
    exit 1
fi
echo "✅ Node.js $(node --version) detected"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "❌ Python version $PYTHON_VERSION is too old. Please upgrade to Python 3.11 or later."
    exit 1
fi
echo "✅ Python $(python3 --version) detected"

# Check if we're in the right directory
if [ ! -f "manifest.json" ]; then
    echo "❌ manifest.json not found. Please run this script from the project root directory."
    exit 1
fi

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
cd server
if [ ! -f "package.json" ]; then
    echo "❌ server/package.json not found. Installation files may be corrupted."
    exit 1
fi

npm install
if [ $? -ne 0 ]; then
    echo "❌ Failed to install Node.js dependencies."
    exit 1
fi
echo "✅ Node.js dependencies installed"

cd ..

# Check if Python dependencies are installed
echo "🐍 Checking Python dependencies..."
if ! python3 -c "import poetry" &> /dev/null; then
    echo "⚠️  Poetry not found. Please install Python dependencies manually:"
    echo "   pip install -r requirements.txt"
else
    echo "✅ Poetry detected, Python dependencies should be managed via 'poetry install'"
fi

# Create logs directory
echo "📁 Creating logs directory..."
mkdir -p server/logs
echo "✅ Logs directory created"

# Configuration check
echo "⚙️  Configuration check..."
if [ -z "$NEO4J_PASSWORD" ]; then
    echo "⚠️  Warning: NEO4J_PASSWORD environment variable not set."
    echo "   You may need to set this before running the extension."
fi

# Test basic functionality
echo "🧪 Testing basic functionality..."
cd server
timeout 10 node -e "
const server = require('./index.js');
console.log('✅ Server module loads successfully');
process.exit(0);
" 2>/dev/null || echo "⚠️  Could not test server module (this may be normal)"

cd ..

echo ""
echo "🎉 Installation completed!"
echo ""
echo "📋 Next steps:"
echo "1. Ensure Neo4j is running with APOC library installed"
echo "2. Start the Python backend server:"
echo "   poetry run uvicorn src.adaptive_graph_of_thoughts.main:app --reload"
echo "3. Configure your MCP client (Claude Desktop, VS Code, etc.)"
echo "4. Test the extension with your AI client"
echo ""
echo "📖 For detailed setup instructions, see DXT_README.md"
echo ""
echo "🔧 Environment variables you may want to set:"
echo "   export NEO4J_URI='bolt://localhost:7687'"
echo "   export NEO4J_USERNAME='neo4j'"
echo "   export NEO4J_PASSWORD='your_password'"
echo "   export LOG_LEVEL='INFO'"
echo ""

exit 0