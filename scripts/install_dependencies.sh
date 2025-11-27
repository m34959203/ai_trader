#!/bin/bash
# Installation script for AI Trader dependencies

set -e

echo "🚀 Installing AI Trader Dependencies..."
echo "========================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}📦 Installing base dependencies...${NC}"
pip install -q numpy pandas scikit-learn

echo -e "${YELLOW}🧠 Installing TensorFlow for LSTM...${NC}"
pip install -q tensorflow>=2.13.0

echo -e "${YELLOW}💬 Installing Telegram bot library...${NC}"
pip install -q python-telegram-bot>=20.7

echo -e "${YELLOW}📊 Installing additional ML libraries...${NC}"
pip install -q imbalanced-learn  # For SMOTE if needed

echo -e "${GREEN}✅ All dependencies installed successfully!${NC}"
echo ""
echo "Next steps:"
echo "1. Configure .env file: cp configs/.env.example configs/.env"
echo "2. Start services: docker-compose up -d"
echo "3. Check health: curl http://localhost:8000/health"
echo ""
echo "📖 See QUICK_START.md for detailed instructions"
