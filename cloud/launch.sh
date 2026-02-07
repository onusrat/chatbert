#!/bin/bash
# ChatBERT - Launch Training on Prime Intellect
#
# This script creates a pod, pushes code, and starts training.
# Run from the project root: bash cloud/launch.sh
#
# Prerequisites:
#   - prime CLI authenticated (prime login)
#   - SSH key configured (~/.ssh/primeintellect_ed25519)

set -e

# Configuration
GPU_TYPE="${GPU_TYPE:-RTX4090_24GB}"
GPU_ID="${GPU_ID:-005e17}"  # Norway RTX 4090, cheapest
POD_NAME="chatbert-training"
IMAGE="pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"
DISK_SIZE=100
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "  ChatBERT - Launch Training"
echo "============================================"
echo "GPU: ${GPU_TYPE}"
echo "Instance: ${GPU_ID}"
echo "Project: ${PROJECT_ROOT}"
echo ""

# Step 1: Create pod
echo "Step 1: Creating pod..."
echo "Running: prime pods create --id ${GPU_ID} --name ${POD_NAME} --image ${IMAGE} --disk-size ${DISK_SIZE} -y"
echo ""

prime pods create \
    --id "${GPU_ID}" \
    --name "${POD_NAME}" \
    --image "${IMAGE}" \
    --disk-size "${DISK_SIZE}" \
    -y

echo ""
echo "Waiting for pod to be ready..."
sleep 15

# Get pod info
echo ""
echo "Pod status:"
prime pods list

echo ""
echo "============================================"
echo "  Pod created! Next steps:"
echo "============================================"
echo ""
echo "1. Get the pod ID from above, then SSH in:"
echo "   prime pods ssh <POD_ID>"
echo ""
echo "2. From ANOTHER terminal, push code to the pod:"
echo "   prime pods ssh <POD_ID> -- 'mkdir -p /workspace/chatbert'"
echo "   rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \\"
echo "     --exclude 'checkpoints' --exclude 'data' --exclude 'site' \\"
echo "     -e 'ssh -i ~/.ssh/primeintellect_ed25519' \\"
echo "     ${PROJECT_ROOT}/ <USER>@<HOST>:/workspace/chatbert/"
echo ""
echo "3. On the pod, start training:"
echo "   cd /workspace/chatbert"
echo "   bash cloud/train_cloud.sh both small"
echo ""
echo "4. When done, pull checkpoints back:"
echo "   rsync -avz -e 'ssh -i ~/.ssh/primeintellect_ed25519' \\"
echo "     <USER>@<HOST>:/workspace/chatbert/checkpoints/ ${PROJECT_ROOT}/checkpoints/"
echo ""
echo "5. Terminate the pod when done:"
echo "   prime pods terminate <POD_ID>"
echo ""
echo "Estimated cost: ~\$3-5 for both models on RTX 4090"
