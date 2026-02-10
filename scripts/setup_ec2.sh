#!/bin/bash
#
# EC2 Setup Script for Overnight Optimizer
# This script prepares the EC2 instance and runs the optimizer
#

set -e  # Exit on error

echo "=========================================="
echo "EC2 Setup for Expert Advisor"
echo "=========================================="
echo ""

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-venv git htop -qq

# Create project directory
echo "📁 Setting up project directory..."
cd ~
mkdir -p expert-advisor
cd expert-advisor

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install Python packages
echo "📚 Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q pandas numpy optuna loguru pyarrow scikit-learn

echo ""
echo "✅ EC2 Setup Complete!"
echo ""
echo "System Info:"
echo "  CPU: $(nproc) cores"
echo "  RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo "  Python: $(python --version)"
echo ""
echo "Ready to run optimizer!"
