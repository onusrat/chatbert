#!/usr/bin/env python3
"""IMR (Iterative MLM Refinement) analysis script.

Captures per-iteration state during generation, produces:
- Iteration trace CSV
- Quality vs num_iterations curve
- Unmasking order visualization
- Confidence evolution stats
- Mask schedule comparison

Usage:
    python scripts/analyze_imr.py \
        --model_path checkpoints/chatbert-imr-small/final \
        --prompt "Hello, how are you?"

    python scripts/analyze_imr.py \
        --model_path checkpoints/chatbert-imr-small/final \
        --run_full_analysis \
        --output_dir results/imr_analysis \
        --max_samples 100
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chatbert.models.iterative_mlm import ChatBERTIterativeMLM
from chatbert.data.datasets import load_dailydialog, load_personachat


def generate_with_trace(model, tokenizer, context, max_length=10, num_iterations=25,
                        temperature=0.4, mask_schedule="confidence", device="cpu"):
    """Generate a response while recording per-iteration state.

    Returns:
        response_text: Final generated response.
        trace: List of dicts, one per iteration, with keys:
            iteration, tokens, confidence_scores, newly_revealed, mean_confidence, num_masked
    """
    model.eval()
    ctx_enc = tokenizer(
        context, max_length=256, truncation=True, padding="max_length", return_tensors="pt",
    )
    input_ids = ctx_enc["input_ids"].to(device)
    attention_mask = ctx_enc["attention_mask"].to(device)

    mask_id = tokenizer.mask_token_id
    sep_id = tokenizer.sep_token_id
    batch_size = 1

    # Initialize all-mask response
    response_ids = torch.full((1, max_length), mask_id, device=device, dtype=input_ids.dtype)
    is_masked = torch.ones(1, max_length, dtype=torch.bool, device=device)
    sep_token = torch.tensor([[sep_id]], device=device, dtype=input_ids.dtype)

    trace = []

    # Record initial state (t=0)
    initial_tokens = [tokenizer.decode([t]) for t in response_ids[0].tolist()]
    trace.append({
        "iteration": 0,
        "tokens": initial_tokens,
        "confidence_scores": [0.0] * max_length,
        "newly_revealed": [],
        "mean_confidence": 0.0,
        "num_masked": max_length,
    })

    for iteration in range(num_iterations):
        t = iteration / max(num_iterations - 1, 1)
        current_temp = model.config.initial_temperature + t * (
            model.config.final_temperature - model.config.initial_temperature
        )

        combined_ids = torch.cat([input_ids, sep_token, response_ids], dim=1)
        sep_mask = torch.ones(1, 1, device=device)
        resp_mask = torch.ones(1, max_length, device=device)
        combined_mask = torch.cat([attention_mask.float(), sep_mask, resp_mask], dim=1)

        with torch.no_grad():
            outputs = model.bert(input_ids=combined_ids, attention_mask=combined_mask, return_dict=True)

        context_len = input_ids.size(1) + 1
        response_logits = outputs.logits[:, context_len:context_len + max_length]
        probs = F.softmax(response_logits / max(current_temp, 1e-7), dim=-1)
        predictions = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values

        # Determine unmask count
        if mask_schedule == "linear":
            unmask_ratio = (iteration + 1) / num_iterations
        elif mask_schedule == "cosine":
            unmask_ratio = 1 - np.cos((iteration + 1) / num_iterations * np.pi / 2)
        else:  # confidence
            unmask_ratio = (iteration + 1) / num_iterations

        target_unmasked = int(max_length * unmask_ratio)

        prev_masked = is_masked[0].clone()
        masked_indices = torch.where(is_masked[0])[0]

        if len(masked_indices) > 0:
            masked_confidence = confidence[0, masked_indices]
            current_unmasked = (~is_masked[0]).sum().item()
            num_to_unmask = min(len(masked_indices), max(1, target_unmasked - current_unmasked))
            _, top_indices = masked_confidence.topk(num_to_unmask)
            unmask_positions = masked_indices[top_indices]
            response_ids[0, unmask_positions] = predictions[0, unmask_positions]
            is_masked[0, unmask_positions] = False

        # Record which positions were newly revealed
        newly_revealed = []
        for pos in range(max_length):
            if prev_masked[pos] and not is_masked[0, pos]:
                newly_revealed.append(pos)

        current_tokens = []
        for pos in range(max_length):
            if is_masked[0, pos]:
                current_tokens.append("[MASK]")
            else:
                current_tokens.append(tokenizer.decode([response_ids[0, pos].item()]))

        trace.append({
            "iteration": iteration + 1,
            "tokens": current_tokens,
            "confidence_scores": confidence[0].cpu().tolist(),
            "newly_revealed": newly_revealed,
            "mean_confidence": confidence[0].mean().item(),
            "num_masked": is_masked[0].sum().item(),
        })

    # Final pass for any remaining masks
    if is_masked.any():
        combined_ids = torch.cat([input_ids, sep_token, response_ids], dim=1)
        with torch.no_grad():
            outputs = model.bert(input_ids=combined_ids, return_dict=True)
        response_logits = outputs.logits[:, context_len:context_len + max_length]
        final_pred = response_logits.argmax(dim=-1)
        response_ids = torch.where(is_masked, final_pred, response_ids)

    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response_text, trace


def save_trace_csv(trace, output_path):
    """Save iteration trace as CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "tokens", "newly_revealed", "mean_confidence", "num_masked"])
        for step in trace:
            writer.writerow([
                step["iteration"],
                " ".join(step["tokens"]),
                ",".join(str(p) for p in step["newly_revealed"]),
                f"{step['mean_confidence']:.4f}",
                step["num_masked"],
            ])


def print_unmasking_visualization(trace):
    """Print text-based visualization of unmasking order."""
    print("\n=== Unmasking Order Visualization ===")
    for step in trace:
        it = step["iteration"]
        tokens_str = " ".join(
            f"\033[96m{t}\033[0m" if i in step["newly_revealed"] else t
            for i, t in enumerate(step["tokens"])
        )
        masked_count = step["num_masked"]
        conf = step["mean_confidence"]
        print(f"  t={it:2d}  [{masked_count:2d} masked, conf={conf:.3f}]  {tokens_str}")


def run_quality_vs_iterations(model, tokenizer, test_data, max_length=10,
                              temperature=0.4, device="cpu"):
    """Measure response quality at different iteration counts.

    Returns:
        Dict mapping num_iterations to metrics dict.
    """
    from chatbert.utils.metrics import compute_metrics, compute_distinct_n

    iteration_counts = [1, 3, 5, 10, 15, 25]
    results = {}

    for n_iter in iteration_counts:
        print(f"  Evaluating with {n_iter} iterations...")
        predictions = []
        references = []

        for ex in test_data:
            ctx = ex["context"]
            if isinstance(ctx, list):
                ctx = " [SEP] ".join(ctx[-5:])
            ref = ex["response"]

            resp, _ = generate_with_trace(
                model, tokenizer, ctx,
                max_length=max_length, num_iterations=n_iter,
                temperature=temperature, device=device,
            )
            predictions.append(resp)
            references.append(ref)

        metrics = compute_metrics(predictions, references)
        metrics.update(compute_distinct_n(predictions))
        results[n_iter] = metrics
        print(f"    BLEU={metrics.get('bleu', 0):.4f}  ROUGE-L={metrics.get('rougeL', 0):.4f}")

    return results


def run_mask_schedule_comparison(model, tokenizer, test_data, max_length=10,
                                num_iterations=25, temperature=0.4, device="cpu"):
    """Compare confidence vs linear vs cosine mask schedules.

    Returns:
        Dict mapping schedule name to metrics dict.
    """
    from chatbert.utils.metrics import compute_metrics, compute_distinct_n

    schedules = ["confidence", "linear", "cosine"]
    results = {}

    for schedule in schedules:
        print(f"  Schedule: {schedule}...")
        predictions = []
        references = []

        for ex in test_data:
            ctx = ex["context"]
            if isinstance(ctx, list):
                ctx = " [SEP] ".join(ctx[-5:])
            ref = ex["response"]

            resp, _ = generate_with_trace(
                model, tokenizer, ctx,
                max_length=max_length, num_iterations=num_iterations,
                temperature=temperature, mask_schedule=schedule, device=device,
            )
            predictions.append(resp)
            references.append(ref)

        metrics = compute_metrics(predictions, references)
        metrics.update(compute_distinct_n(predictions))
        results[schedule] = metrics
        print(f"    BLEU={metrics.get('bleu', 0):.4f}  ROUGE-L={metrics.get('rougeL', 0):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze ChatBERT-IMR generation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to IMR checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt for trace")
    parser.add_argument("--max_length", type=int, default=10, help="Response length")
    parser.add_argument("--num_iterations", type=int, default=25, help="Refinement iterations")
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature")
    parser.add_argument("--run_full_analysis", action="store_true", help="Run full analysis suite")
    parser.add_argument("--output_dir", type=str, default="results/imr_analysis", help="Output dir")
    parser.add_argument("--max_samples", type=int, default=100, help="Samples for full analysis")
    parser.add_argument("--device", type=str, default=None, help="Device")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading IMR from {args.model_path}...")
    model = ChatBERTIterativeMLM.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single prompt trace
    if args.prompt:
        print(f"\nGenerating for: \"{args.prompt}\"")
        response, trace = generate_with_trace(
            model, tokenizer, args.prompt,
            max_length=args.max_length, num_iterations=args.num_iterations,
            temperature=args.temperature, device=device,
        )
        print(f"Response: {response}")
        print_unmasking_visualization(trace)
        save_trace_csv(trace, output_dir / "trace.csv")
        print(f"\nTrace saved to {output_dir / 'trace.csv'}")

    # Full analysis
    if args.run_full_analysis:
        print("\n=== Loading test data ===")
        test_data = []
        for loader, split in [(load_dailydialog, "test"), (load_personachat, "validation")]:
            test_data.extend(loader(split=split))
        import random
        rng = random.Random(42)
        rng.shuffle(test_data)
        test_data = test_data[:args.max_samples]
        print(f"Test samples: {len(test_data)}")

        # Run traces for a few examples
        print("\n=== Sample Traces ===")
        sample_prompts = [
            "Hello, how are you?",
            "What do you like to do for fun?",
            "I had a terrible day today.",
            "Do you have any pets?",
            "Tell me about yourself.",
        ]
        all_traces = {}
        for prompt in sample_prompts:
            resp, trace = generate_with_trace(
                model, tokenizer, prompt,
                max_length=args.max_length, num_iterations=args.num_iterations,
                temperature=args.temperature, device=device,
            )
            print(f"\n\"{prompt}\" -> \"{resp}\"")
            print_unmasking_visualization(trace)
            all_traces[prompt] = {"response": resp, "trace": trace}
            save_trace_csv(trace, output_dir / f"trace_{prompt[:20].replace(' ', '_')}.csv")

        # Quality vs iterations
        print("\n=== Quality vs Iterations ===")
        iter_results = run_quality_vs_iterations(
            model, tokenizer, test_data,
            max_length=args.max_length, temperature=args.temperature, device=device,
        )
        with open(output_dir / "quality_vs_iterations.json", "w") as f:
            json.dump(iter_results, f, indent=2)

        # Mask schedule comparison
        print("\n=== Mask Schedule Comparison ===")
        schedule_results = run_mask_schedule_comparison(
            model, tokenizer, test_data,
            max_length=args.max_length, num_iterations=args.num_iterations,
            temperature=args.temperature, device=device,
        )
        with open(output_dir / "mask_schedule_comparison.json", "w") as f:
            json.dump(schedule_results, f, indent=2)

        # Confidence evolution (aggregate across traces)
        print("\n=== Confidence Evolution ===")
        conf_evolution = []
        for prompt_data in all_traces.values():
            for step in prompt_data["trace"]:
                conf_evolution.append({
                    "iteration": step["iteration"],
                    "mean_confidence": step["mean_confidence"],
                    "num_masked": step["num_masked"],
                })

        with open(output_dir / "confidence_evolution.json", "w") as f:
            json.dump(conf_evolution, f, indent=2)

        # Summary
        summary = {
            "quality_vs_iterations": iter_results,
            "mask_schedule_comparison": schedule_results,
            "sample_traces": {
                k: {"response": v["response"], "num_steps": len(v["trace"])}
                for k, v in all_traces.items()
            },
        }
        with open(output_dir / "analysis_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
