#!/bin/bash
#
# Deploy project to EC2 and start overnight optimizer
#

set -e

EC2_HOST="ec2-3-218-55-57.compute-1.amazonaws.com"
EC2_USER="ubuntu"
PEM_FILE="/home/chris/expert-advisor/docs/ea.pem"
PROJECT_DIR="/home/chris/expert-advisor"

echo "=========================================="
echo "Deploying to EC2"
echo "=========================================="
echo ""

# Step 1: Create deployment package (exclude heavy files)
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

# Copy only the data files we need (H1, H4, D for EURUSD)
echo "📊 Copying data files..."
cp data/raw/EURUSDm_H1.parquet "$DEPLOY_DIR/expert-advisor/data/raw/" 2>/dev/null || echo "  Warning: EURUSDm_H1.parquet not found"
cp data/raw/EURUSDm_H4.parquet "$DEPLOY_DIR/expert-advisor/data/raw/" 2>/dev/null || echo "  Warning: EURUSDm_H4.parquet not found"
cp data/raw/EURUSDm_D.parquet "$DEPLOY_DIR/expert-advisor/data/raw/" 2>/dev/null || echo "  Warning: EURUSDm_D.parquet not found"

# Copy GOLD data if available
cp data/raw/XAUUSDm_H1.parquet "$DEPLOY_DIR/expert-advisor/data/raw/" 2>/dev/null || echo "  Note: GOLD data not yet exported"
cp data/raw/XAUUSDm_H4.parquet "$DEPLOY_DIR/expert-advisor/data/raw/" 2>/dev/null || true
cp data/raw/XAUUSDm_D.parquet "$DEPLOY_DIR/expert-advisor/data/raw/" 2>/dev/null || true

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

# Extract files
cd ~
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
ls -lh data/raw/*.parquet 2>/dev/null || echo "  No data files found"
echo ""

ENDSSH

# Step 4: Start optimizer
echo ""
echo "=========================================="
echo "Starting Overnight Optimizer"
echo "=========================================="
echo ""

ssh -i "$PEM_FILE" "${EC2_USER}@${EC2_HOST}" << 'ENDSSH'
cd ~/expert-advisor
source .venv/bin/activate

# Start optimizer in background with nohup
nohup python scripts/overnight_optimizer.py > overnight.log 2>&1 &
PID=$!

echo "🔥 Optimizer started!"
echo "   PID: $PID"
echo "   Log: ~/expert-advisor/overnight.log"
echo ""
echo "Monitor progress:"
echo "   ssh -i docs/ea.pem ubuntu@ec2-3-218-55-57.compute-1.amazonaws.com"
echo "   cd expert-advisor && tail -f overnight.log"
echo ""
echo "Check if running:"
echo "   ps aux | grep overnight_optimizer"
echo ""

# Show first few lines
echo "Initial output:"
sleep 2
tail -20 overnight.log || true

ENDSSH

# Cleanup
rm -rf "$DEPLOY_DIR"

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "The optimizer is now running on EC2!"
echo ""
echo "Estimated completion: 12-16 hours (t2.micro)"
echo ""
echo "To check progress:"
echo "  ssh -i $PEM_FILE ${EC2_USER}@${EC2_HOST}"
echo "  cd expert-advisor && tail -f overnight.log"
echo ""
echo "To stop:"
echo "  ssh -i $PEM_FILE ${EC2_USER}@${EC2_HOST}"
echo "  pkill -f overnight_optimizer"
echo ""
