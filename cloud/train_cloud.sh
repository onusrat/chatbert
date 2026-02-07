#!/bin/bash
# ChatBERT Cloud Training Script for Prime Intellect
# Trains both ChatBERT-ED (encoder-decoder) and ChatBERT-IMR (iterative MLM)
#
# Usage:
#   bash cloud/train_cloud.sh                    # Train both models
#   bash cloud/train_cloud.sh ed                 # Train ED only
#   bash cloud/train_cloud.sh imr                # Train IMR only
#   bash cloud/train_cloud.sh ed base            # Train ED base config

set -e

MODEL="${1:-both}"
SIZE="${2:-small}"
OUTPUT_DIR="${3:-./checkpoints}"

echo "============================================"
echo "  ChatBERT Training on Prime Intellect"
echo "============================================"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
else
    echo "WARNING: No GPU detected! Training will be very slow."
fi

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q wandb gradio

# Download datasets
echo ""
echo "Downloading datasets..."
python scripts/download_data.py --datasets daily_dialog personachat

# Train ChatBERT-ED
train_ed() {
    local config="configs/ed_${SIZE}.yaml"
    echo ""
    echo "============================================"
    echo "  Training ChatBERT-ED (${SIZE})"
    echo "  Config: ${config}"
    echo "============================================"

    python scripts/train.py \
        --config "${config}" \
        --output_dir "${OUTPUT_DIR}"

    echo "ChatBERT-ED training complete!"
    echo "Model saved to: ${OUTPUT_DIR}/chatbert-ed-${SIZE}/final"
}

# Train ChatBERT-IMR
train_imr() {
    local config="configs/imr_${SIZE}.yaml"
    echo ""
    echo "============================================"
    echo "  Training ChatBERT-IMR (${SIZE})"
    echo "  Config: ${config}"
    echo "============================================"

    python scripts/train.py \
        --config "${config}" \
        --output_dir "${OUTPUT_DIR}"

    echo "ChatBERT-IMR training complete!"
    echo "Model saved to: ${OUTPUT_DIR}/chatbert-imr-${SIZE}/final"
}

case "${MODEL}" in
    ed)
        train_ed
        ;;
    imr)
        train_imr
        ;;
    both)
        train_ed
        train_imr
        ;;
    *)
        echo "Unknown model: ${MODEL}. Use 'ed', 'imr', or 'both'."
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  All training complete!"
echo "  Checkpoints: ${OUTPUT_DIR}"
echo "============================================"

# Run demo to verify models work
echo ""
echo "Running quick generation test..."
python -c "
import sys
sys.path.insert(0, 'src')
from chatbert.inference.generator import ChatBERTGenerator
from pathlib import Path

checkpoints = Path('${OUTPUT_DIR}')
for model_dir in sorted(checkpoints.glob('*/final')):
    model_type = 'encoder_decoder' if 'ed' in model_dir.parent.name else 'iterative_mlm'
    print(f'Testing {model_dir.parent.name} ({model_type})...')
    try:
        gen = ChatBERTGenerator.from_pretrained(str(model_dir), model_type=model_type)
        response = gen.generate('Hello, how are you?')
        print(f'  Input: Hello, how are you?')
        print(f'  Response: {response}')
        print(f'  OK')
    except Exception as e:
        print(f'  Error: {e}')
    print()
"

echo "Done! Models are ready."
