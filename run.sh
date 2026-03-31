#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# ResumeIQ — Setup & Run Script
# Usage: bash run.sh
# ─────────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║          ResumeIQ — AI Resume Analyzer           ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi
echo "✅ Python found: $(python3 --version)"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies (this may take a few minutes on first run)..."
pip install -r requirements.txt --quiet

echo ""
echo "🚀 Starting ResumeIQ server..."
echo "   → Open http://localhost:5000 in your browser"
echo "   → Press Ctrl+C to stop"
echo ""

python3 app.py
