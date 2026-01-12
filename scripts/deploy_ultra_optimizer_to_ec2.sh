#!/bin/bash
#
# Deploy Ultra-Comprehensive Optimizer to EC2
# Tests all symbols (EURUSD, GBPUSD, USDJPY, AUDUSD, XAUUSD) across all timeframes (M1-D)
#

set -e

EC2_HOST="ec2-3-218-55-57.compute-1.amazonaws.com"
EC2_USER="ubuntu"
PEM_FILE="/home/chris/expert-advisor/docs/ea.pem"
PROJECT_DIR="/home/chris/expert-advisor"

echo "=========================================="
echo "Ultra-Comprehensive Optimizer Deployment"
echo "=========================================="
echo ""

# Step 1: Create deployment package
echo "📦 Creating deployment package..."
cd "$PROJECT_DIR"

# Create a temporary directory for deployment files
DEPLOY_DIR=$(mktemp -d)
echo "Using temp directory: $DEPLOY_DIR"

# Copy essential files only
mkdir -p "$DEPLOY_DIR/expert-advisor"
cp -r src "$DEPLOY_DIR/expert-advisor/"
cp -r scripts "$DEPLOY_DIR/expert-advisor/"
cp -r configs "$DEPLOY_DIR/expert-advisor/"
mkdir -p "$DEPLOY_DIR/expert-advisor/data/raw"
mkdir -p "$DEPLOY_DIR/expert-advisor/reports"

# Copy all available data files
echo "📊 Copying data files..."
cp data/raw/*.parquet "$DEPLOY_DIR/expert-advisor/data/raw/" 2>/dev/null || echo "  Warning: Some data files not found"

echo "  Copied $(ls -1 $DEPLOY_DIR/expert-advisor/data/raw/*.parquet 2>/dev/null | wc -l) data files"

# Create requirements.txt
cat > "$DEPLOY_DIR/expert-advisor/requirements.txt" << 'EOF'
pandas>=2.0.0
numpy>=1.24.0
optuna>=3.0.0
loguru>=0.7.0
pyarrow>=12.0.0
scikit-learn>=1.3.0
EOF

# Create tarball
echo "🗜️  Creating tarball..."
cd "$DEPLOY_DIR"
tar -czf expert-advisor.tar.gz expert-advisor/
echo "Package size: $(du -h expert-advisor.tar.gz | cut -f1)"

# Step 2: Upload to EC2
echo ""
echo "📤 Uploading to EC2..."
scp -i "$PEM_FILE" expert-advisor.tar.gz "${EC2_USER}@${EC2_HOST}:~/"

# Step 3: Setup and run on EC2
echo ""
echo "🚀 Setting up EC2..."
ssh -i "$PEM_FILE" "${EC2_USER}@${EC2_HOST}" << 'ENDSSH'
set -e

echo "=========================================="
echo "EC2 Setup"
echo "=========================================="

# Stop any running optimizers
pkill -f optimizer.py || true

# Extract files
cd ~
rm -rf expert-advisor  # Clean previous installation
tar -xzf expert-advisor.tar.gz
cd expert-advisor

# Update system
echo "📦 Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-venv htop -qq

# Create virtual environment
echo "🐍 Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "📚 Installing Python packages..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "System Info:"
echo "  CPU: $(nproc) cores"
echo "  RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo "  Disk: $(df -h ~ | tail -1 | awk '{print $4}') free"
echo "  Python: $(python --version)"
echo ""

# List available data
echo "📊 Available data files:"
ls -1 data/raw/*.parquet 2>/dev/null | while read f; do
    basename "$f" | sed 's/\.parquet//'
done | column -c 80
echo ""

ENDSSH

# Step 4: Start ultra-comprehensive optimizer
echo ""
echo "=========================================="
echo "Starting Ultra-Comprehensive Optimizer"
echo "=========================================="
echo ""
echo "This will test:"
echo "  • 5 symbols: EURUSD, GBPUSD, USDJPY, AUDUSD, XAUUSD (GOLD)"
echo "  • 7 timeframes: M1, M5, M15, M30, H1, H4, D"
echo "  • 500 trials per combination"
echo "  • Total: 35 optimizations"
echo "  • Estimated time: 24-48 hours"
echo ""

ssh -i "$PEM_FILE" "${EC2_USER}@${EC2_HOST}" << 'ENDSSH'
cd ~/expert-advisor
source .venv/bin/activate

# Start optimizer in background with nohup
nohup python scripts/ultra_comprehensive_optimizer.py > ultra_optimizer.log 2>&1 &
PID=$!

echo "🔥 Ultra-Comprehensive Optimizer started!"
echo "   PID: $PID"
echo "   Log: ~/expert-advisor/ultra_optimizer.log"
echo ""
echo "📊 Progress files will be saved to: ~/expert-advisor/reports/"
echo ""
echo "Monitor progress:"
echo "   ssh -i docs/ea.pem ubuntu@ec2-3-218-55-57.compute-1.amazonaws.com"
echo "   cd expert-advisor && tail -f ultra_optimizer.log"
echo ""
echo "Check results periodically:"
echo "   ls -lh reports/ultra_progress_*.json"
echo ""
echo "Check if running:"
echo "   ps aux | grep ultra_comprehensive_optimizer"
echo ""

# Show first few lines
echo "Initial output:"
sleep 3
tail -30 ultra_optimizer.log || true

ENDSSH

# Cleanup
rm -rf "$DEPLOY_DIR"

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "The ultra-comprehensive optimizer is now running on EC2!"
echo ""
echo "Estimated completion: 24-48 hours"
echo ""
echo "Current data coverage:"
echo "  ✅ H1, H4, D timeframes (15 files)"
echo "  ⏳ M1, M5, M15, M30 timeframes (need Windows export)"
echo ""
echo "To get COMPLETE coverage (35 optimizations):"
echo "  1. Run on Windows: python scripts/export_mt5_data_windows.py"
echo "  2. This will export M1, M5, M15, M30 data (20 more files)"
echo "  3. Re-run this deployment script"
echo ""
echo "To check progress now:"
echo "  ssh -i $PEM_FILE ${EC2_USER}@${EC2_HOST}"
echo "  cd expert-advisor && tail -f ultra_optimizer.log"
echo ""
echo "To download results:"
echo "  scp -i $PEM_FILE ${EC2_USER}@${EC2_HOST}:~/expert-advisor/reports/*.json reports/"
echo ""
echo "To stop:"
echo "  ssh -i $PEM_FILE ${EC2_USER}@${EC2_HOST}"
echo "  pkill -f ultra_comprehensive_optimizer"
echo ""
