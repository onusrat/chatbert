#!/bin/bash
# ============================================================
# ChatBERT: Full training & evaluation pipeline
# Run on GPU pod (A100). Total: ~7.5 hrs
# ============================================================
set -e

WORKSPACE="/workspace/chatbert"
cd "$WORKSPACE"

# Install/update
pip install -e ".[all]" 2>/dev/null

echo "============================================================"
echo "PHASE 1: Evaluate existing models (~30 min)"
echo "============================================================"

# Evaluate ChatBERT-ED
echo ">>> Evaluating ChatBERT-ED..."
python scripts/evaluate.py \
    --model_path checkpoints/chatbert-ed-small/final \
    --model_type encoder_decoder \
    --output_path results/chatbert_ed_small.json \
    --max_samples 500

# Evaluate ChatBERT-IMR
echo ">>> Evaluating ChatBERT-IMR..."
python scripts/evaluate.py \
    --model_path checkpoints/chatbert-imr-small/final \
    --model_type iterative_mlm \
    --output_path results/chatbert_imr_small.json \
    --max_samples 500 \
    --max_length 10 \
    --num_iterations 25

# IMR Analysis
echo ">>> Running IMR analysis..."
python scripts/analyze_imr.py \
    --model_path checkpoints/chatbert-imr-small/final \
    --run_full_analysis \
    --output_dir results/imr_analysis \
    --max_samples 100

echo "============================================================"
echo "PHASE 2: Train GPT-2 baseline (~45 min)"
echo "============================================================"

python scripts/train_gpt2_baseline.py \
    --config configs/gpt2_baseline.yaml \
    --output_dir checkpoints

echo ">>> Evaluating GPT-2 baseline..."
python scripts/evaluate.py \
    --model_path checkpoints/gpt2-baseline/final \
    --model_type gpt2_baseline \
    --output_path results/gpt2_baseline.json \
    --max_samples 500

echo "============================================================"
echo "PHASE 3: Run ablations (~3.5 hrs)"
echo "============================================================"

# Priority ablations (most informative)
PRIORITY_ABLATIONS=(
    "configs/ablations/ed_frozen_encoder.yaml"
    "configs/ablations/ed_decoder_depth_2.yaml"
    "configs/ablations/ed_decoder_depth_6.yaml"
    "configs/ablations/ed_dailydialog_only.yaml"
)

for config in "${PRIORITY_ABLATIONS[@]}"; do
    name=$(basename "$config" .yaml)
    echo ">>> Ablation: $name"

    python scripts/train.py \
        --config "$config" \
        --output_dir checkpoints

    python scripts/evaluate.py \
        --model_path "checkpoints/$name/final" \
        --model_type encoder_decoder \
        --output_path "results/ablations/$name.json" \
        --max_samples 500
done

# LR ablations (if time permits)
for config in configs/ablations/ed_lr_*.yaml; do
    name=$(basename "$config" .yaml)
    echo ">>> Ablation: $name"

    python scripts/train.py \
        --config "$config" \
        --output_dir checkpoints

    python scripts/evaluate.py \
        --model_path "checkpoints/$name/final" \
        --model_type encoder_decoder \
        --output_path "results/ablations/$name.json" \
        --max_samples 500
done

echo "============================================================"
echo "PHASE 4: SmolTalk training (~2.5 hrs)"
echo "============================================================"

# Download SmolTalk
python scripts/download_data.py --datasets smoltalk

# Train ED with SmolTalk
echo ">>> Training ED + SmolTalk..."
python scripts/train.py \
    --config configs/ed_small_smoltalk.yaml \
    --output_dir checkpoints

python scripts/evaluate.py \
    --model_path checkpoints/chatbert-ed-small-smoltalk/final \
    --model_type encoder_decoder \
    --output_path results/chatbert_ed_small_smoltalk.json \
    --max_samples 500

# Train IMR with SmolTalk
echo ">>> Training IMR + SmolTalk..."
python scripts/train.py \
    --config configs/imr_small_smoltalk.yaml \
    --output_dir checkpoints

python scripts/evaluate.py \
    --model_path checkpoints/chatbert-imr-small-smoltalk/final \
    --model_type iterative_mlm \
    --output_path results/chatbert_imr_small_smoltalk.json \
    --max_samples 500 \
    --max_length 10 \
    --num_iterations 25

echo "============================================================"
echo "PHASE 5: Generate comparison tables"
echo "============================================================"

python scripts/compare_results.py \
    results/chatbert_ed_small.json \
    results/chatbert_imr_small.json \
    results/gpt2_baseline.json \
    --output_dir results/comparison

echo "============================================================"
echo "ALL DONE!"
echo "============================================================"
echo "Results in: results/"
echo "Checkpoints in: checkpoints/"
