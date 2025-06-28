#!/bin/bash

# DXT Package Creation Script
# Creates a distributable DXT package for the Adaptive Graph of Thoughts extension

set -e

echo "📦 Creating Adaptive Graph of Thoughts DXT Package"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "manifest.json" ]; then
    echo "❌ manifest.json not found. Please run this script from the project root directory."
    exit 1
fi

# Read version from manifest
VERSION=$(python3 -c "import json; print(json.load(open('manifest.json'))['version'])")
PACKAGE_NAME="adaptive-graph-of-thoughts-dxt-v${VERSION}"

echo "📋 Package Info:"
echo "   Name: ${PACKAGE_NAME}"
echo "   Version: ${VERSION}"
echo ""

# Create temporary package directory
TEMP_DIR=$(mktemp -d)
PACKAGE_DIR="${TEMP_DIR}/${PACKAGE_NAME}"

echo "📁 Creating package structure..."
mkdir -p "${PACKAGE_DIR}"

# Copy core files
echo "📄 Copying core files..."
cp manifest.json "${PACKAGE_DIR}/"
cp DXT_README.md "${PACKAGE_DIR}/README.md"
cp LICENSE "${PACKAGE_DIR}/" 2>/dev/null || echo "⚠️ LICENSE file not found, skipping"

# Copy server directory
echo "📂 Copying server directory..."
cp -r server "${PACKAGE_DIR}/"

# Copy assets
echo "🖼️ Copying assets..."
cp -r assets "${PACKAGE_DIR}/" 2>/dev/null || echo "⚠️ Assets directory not found, skipping"

# Install production dependencies
echo "📦 Installing production dependencies..."
cd "${PACKAGE_DIR}/server"
npm install --production --no-optional

# Remove development files
echo "🧹 Cleaning up development files..."
rm -rf node_modules/.cache 2>/dev/null || true
rm -rf logs 2>/dev/null || true
rm -f npm-debug.log* 2>/dev/null || true
rm -f .nyc_output 2>/dev/null || true

cd "${TEMP_DIR}"

# Create exclusion list for zip
cat > .zipignore << EOF
*.log
*.tmp
.DS_Store
Thumbs.db
node_modules/.cache/*
logs/*
coverage/*
.nyc_output/*
*.test.js
test/*
tests/*
spec/*
__tests__/*
*.spec.js
.git/*
.github/*
.vscode/*
.idea/*
*.orig
*.rej
*~
.env
.env.local
.env.development.local
.env.test.local
.env.production.local
EOF

# Create ZIP package
echo "🗜️ Creating ZIP package..."
PACKAGE_FILE="${PACKAGE_NAME}.zip"
cd "${PACKAGE_NAME}"

# Create the zip file
zip -r "../${PACKAGE_FILE}" . -x@../.zipignore
cd ..

# Move package to original directory
ORIGINAL_DIR=$(dirname "${TEMP_DIR}")
mv "${PACKAGE_FILE}" "${ORIGINAL_DIR}/"

# Calculate package size
PACKAGE_SIZE=$(du -h "${ORIGINAL_DIR}/${PACKAGE_FILE}" | cut -f1)

echo ""
echo "✅ Package created successfully!"
echo "📍 Location: ${ORIGINAL_DIR}/${PACKAGE_FILE}"
echo "📏 Size: ${PACKAGE_SIZE}"
echo ""

# Validate package contents
echo "🔍 Package contents:"
unzip -l "${ORIGINAL_DIR}/${PACKAGE_FILE}" | head -20
echo ""

# Cleanup temporary directory
rm -rf "${TEMP_DIR}"

echo "📋 Package validation checklist:"
echo "  ✅ manifest.json included"
echo "  ✅ Server code included"
echo "  ✅ Dependencies installed"
echo "  ✅ Development files removed"
echo "  ✅ README documentation included"
echo ""

echo "🚀 Next steps:"
echo "1. Test the package by extracting it to a test directory"
echo "2. Run the validation script: node validate-dxt.js"
echo "3. Test integration with your MCP client"
echo "4. Distribute the package file: ${PACKAGE_FILE}"
echo ""

echo "📖 Installation instructions for users:"
echo "1. Extract ${PACKAGE_FILE} to desired location"
echo "2. Run 'npm install' in the server directory (if needed)"
echo "3. Configure environment variables"
echo "4. Add to MCP client configuration"
echo ""

exit 0