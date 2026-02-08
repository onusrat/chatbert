#!/bin/bash
set -e
cd /workspace/chatbert

echo "=== ED + empathetic_dialogues ==="
python3 scripts/evaluate.py \
    --model_path checkpoints/chatbert-ed-small/final \
    --model_type encoder_decoder \
    --datasets empathetic_dialogues \
    --output_path results/chatbert_ed_small_empathetic.json \
    --max_samples 500

echo "=== ED + topical_chat ==="
python3 scripts/evaluate.py \
    --model_path checkpoints/chatbert-ed-small/final \
    --model_type encoder_decoder \
    --datasets topical_chat \
    --output_path results/chatbert_ed_small_topical.json \
    --max_samples 500

echo "=== ED + wizard_of_wikipedia ==="
python3 scripts/evaluate.py \
    --model_path checkpoints/chatbert-ed-small/final \
    --model_type encoder_decoder \
    --datasets wizard_of_wikipedia \
    --output_path results/chatbert_ed_small_wow.json \
    --max_samples 500

echo "=== IMR + empathetic_dialogues ==="
python3 scripts/evaluate.py \
    --model_path checkpoints/chatbert-imr-small/final \
    --model_type iterative_mlm \
    --datasets empathetic_dialogues \
    --output_path results/chatbert_imr_small_empathetic.json \
    --max_samples 500 --max_length 10 --num_iterations 25

echo "=== IMR + topical_chat ==="
python3 scripts/evaluate.py \
    --model_path checkpoints/chatbert-imr-small/final \
    --model_type iterative_mlm \
    --datasets topical_chat \
    --output_path results/chatbert_imr_small_topical.json \
    --max_samples 500 --max_length 10 --num_iterations 25

echo "=== IMR + wizard_of_wikipedia ==="
python3 scripts/evaluate.py \
    --model_path checkpoints/chatbert-imr-small/final \
    --model_type iterative_mlm \
    --datasets wizard_of_wikipedia \
    --output_path results/chatbert_imr_small_wow.json \
    --max_samples 500 --max_length 10 --num_iterations 25

echo "=== GPT2 + empathetic_dialogues ==="
python3 scripts/evaluate.py \
    --model_path checkpoints/gpt2-baseline/final \
    --model_type gpt2_baseline \
    --datasets empathetic_dialogues \
    --output_path results/gpt2_baseline_empathetic.json \
    --max_samples 500

echo "=== GPT2 + topical_chat ==="
python3 scripts/evaluate.py \
    --model_path checkpoints/gpt2-baseline/final \
    --model_type gpt2_baseline \
    --datasets topical_chat \
    --output_path results/gpt2_baseline_topical.json \
    --max_samples 500

echo "=== GPT2 + wizard_of_wikipedia ==="
python3 scripts/evaluate.py \
    --model_path checkpoints/gpt2-baseline/final \
    --model_type gpt2_baseline \
    --datasets wizard_of_wikipedia \
    --output_path results/gpt2_baseline_wow.json \
    --max_samples 500

echo "=== ALL EVALUATIONS COMPLETE ==="
