#!/bin/bash
# Code quality check script
# Runs formatting, import sorting, linting, and type checking

set -e

echo "🔍 Running code quality checks..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if directories exist
if [ ! -d "semantica" ]; then
    echo "❌ semantica/ directory not found"
    exit 1
fi

# Check formatting
echo "📝 Checking code formatting (black)..."
if black --check semantica/ 2>/dev/null; then
    echo "✅ Black check passed"
else
    echo "❌ Code formatting issues found"
    echo "   Run: black semantica/"
    exit 1
fi

# Check import sorting
echo "📦 Checking import sorting (isort)..."
if isort --check-only semantica/ 2>/dev/null; then
    echo "✅ isort check passed"
else
    echo "❌ Import sorting issues found"
    echo "   Run: isort semantica/"
    exit 1
fi

# Lint with flake8
echo "🔎 Linting with flake8..."
if flake8 semantica/; then
    echo "✅ flake8 check passed"
else
    echo "❌ Linting issues found"
    exit 1
fi

# Type check with mypy
echo "🔬 Type checking with mypy..."
if mypy semantica/; then
    echo "✅ mypy check passed"
else
    echo "❌ Type checking issues found"
    exit 1
fi

echo ""
echo "✅ All code quality checks passed!"

