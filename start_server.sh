#!/bin/bash

echo "🚀 AI Trader - Quick Start"
echo "=========================="
echo ""

# Check Python version
echo "Checking Python version..."
python --version

# Kill existing server
echo "Stopping any existing servers..."
pkill -f "uvicorn src.main:app" || true
sleep 2

# Create minimal environment
echo "Setting up environment..."
export PYTHONPATH=/home/user/ai_trader:$PYTHONPATH
export APP_ENV=development
export LOG_LEVEL=INFO
export FEATURE_OHLCV=true
export FEATURE_SIGNALS=true
export FEATURE_PAPER=true
export FEATURE_UI=true
export DISABLE_DOCS=false

# Start server
echo ""
echo "🔥 Starting AI Trader server on http://0.0.0.0:8001"
echo "📊 Dashboard: http://localhost:8001/dashboard"
echo "📖 API Docs: http://localhost:8001/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd /home/user/ai_trader
python -m uvicorn src.main:app --host 0.0.0.0 --port 8001 --log-level info
