#!/bin/bash
# Prime Intellect Pod Setup Script for ChatBERT
#
# This script is run ONCE when the pod first starts.
# It clones the repo, installs deps, and kicks off training.
#
# Usage from your local machine:
#   prime pods ssh <pod-id> -- 'bash -s' < cloud/setup_prime.sh

set -e

echo "============================================"
echo "  ChatBERT - Prime Intellect Pod Setup"
echo "============================================"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo ""

# GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
echo ""

# Create workspace
WORKSPACE="/workspace/chatbert"
if [ -d "$WORKSPACE" ]; then
    echo "Workspace exists, pulling latest..."
    cd "$WORKSPACE"
    git pull 2>/dev/null || true
else
    echo "Setting up workspace..."
    mkdir -p /workspace
    cd /workspace

    # If the code was rsync'd, use it; otherwise note to push code
    if [ -d "/workspace/chatbert" ]; then
        cd /workspace/chatbert
    else
        echo "ERROR: Code not found at /workspace/chatbert"
        echo "Push code first with:"
        echo "  rsync -avz --exclude '.git' --exclude '__pycache__' . <pod>:/workspace/chatbert/"
        exit 1
    fi
fi

# Install Python deps
echo ""
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install wandb -q

echo ""
echo "Setup complete! Ready to train."
echo ""
echo "To start training:"
echo "  cd /workspace/chatbert"
echo "  bash cloud/train_cloud.sh both small"
echo ""
echo "Or run individual models:"
echo "  bash cloud/train_cloud.sh ed small"
echo "  bash cloud/train_cloud.sh imr small"
