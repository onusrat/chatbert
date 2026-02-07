#!/bin/bash
set -e
cd /workspace/chatbert

echo "============================================================"
echo "FIX SCRIPT START: $(date)"
echo "============================================================"

echo ">>> Phase 3 FIX: Evaluate ablation models (already trained)"
echo "============================================================"

# Map config names to actual checkpoint directory names
declare -A CKPT_NAMES
CKPT_NAMES[ed_frozen_encoder]="chatbert-ed-frozen-encoder"
CKPT_NAMES[ed_decoder_depth_2]="chatbert-ed-decoder-2"
CKPT_NAMES[ed_decoder_depth_6]="chatbert-ed-decoder-6"
CKPT_NAMES[ed_dailydialog_only]="chatbert-ed-dailydialog-only"
CKPT_NAMES[ed_lr_1e4]="chatbert-ed-lr-1e4"
CKPT_NAMES[ed_lr_1e5]="chatbert-ed-lr-1e5"

mkdir -p results/ablations

for config_name in ed_frozen_encoder ed_decoder_depth_2 ed_decoder_depth_6 ed_dailydialog_only ed_lr_1e4 ed_lr_1e5; do
    ckpt_name="${CKPT_NAMES[$config_name]}"
    model_path="checkpoints/${ckpt_name}/final"

    if [ ! -d "$model_path" ]; then
        echo "SKIP: $model_path does not exist"
        continue
    fi

    echo ">>> Evaluating $config_name (from $model_path)..."
    python3 scripts/evaluate.py \
        --model_path "$model_path" \
        --model_type encoder_decoder \
        --output_path "results/ablations/${config_name}.json" \
        --max_samples 500 || echo "FAILED: $config_name eval"
done

echo "============================================================"
echo ">>> Phase 4 FIX: SmolTalk training"
echo "============================================================"

echo ">>> Downloading SmolTalk data..."
python3 scripts/download_data.py --datasets smoltalk

echo ">>> Training ED + SmolTalk..."
python3 scripts/train.py --config configs/ed_small_smoltalk.yaml --output_dir checkpoints

echo ">>> Evaluating ED + SmolTalk..."
python3 scripts/evaluate.py \
    --model_path checkpoints/chatbert-ed-small-smoltalk/final \
    --model_type encoder_decoder \
    --output_path results/chatbert_ed_small_smoltalk.json \
    --max_samples 500

echo ">>> Training IMR + SmolTalk..."
python3 scripts/train.py --config configs/imr_small_smoltalk.yaml --output_dir checkpoints

echo ">>> Evaluating IMR + SmolTalk..."
python3 scripts/evaluate.py \
    --model_path checkpoints/chatbert-imr-small-smoltalk/final \
    --model_type iterative_mlm \
    --output_path results/chatbert_imr_small_smoltalk.json \
    --max_samples 500 --max_length 10 --num_iterations 25

echo "============================================================"
echo ">>> Phase 5: Generate comparison"
echo "============================================================"
python3 scripts/compare_results.py \
    results/chatbert_ed_small.json \
    results/chatbert_imr_small.json \
    results/gpt2_baseline.json \
    --output_dir results/comparison || echo "comparison failed"

echo "============================================================"
echo "ALL DONE: $(date)"
echo "============================================================"
