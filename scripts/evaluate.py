#!/usr/bin/env python3
"""Evaluation script for ChatBERT models.

Generates responses on test splits, computes BLEU, ROUGE, BERTScore,
Distinct-N, perplexity, and response length stats. Saves results as JSON.

Usage:
    python scripts/evaluate.py \
        --model_path checkpoints/chatbert-ed-small/final \
        --model_type encoder_decoder \
        --output_path results/ed_small.json \
        --max_samples 500
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chatbert.inference.generator import ChatBERTGenerator
from chatbert.data.datasets import (
    load_dailydialog,
    load_personachat,
    load_empathetic_dialogues,
    load_topical_chat,
    load_wizard_of_wikipedia,
)
from chatbert.utils.metrics import (
    compute_metrics,
    compute_distinct_n,
    compute_response_length_stats,
    compute_perplexity_ed,
    compute_perplexity_imr,
)


def load_test_data(datasets, max_samples=500, max_turns=5):
    """Load test splits from specified datasets.

    Args:
        datasets: List of dataset names.
        max_samples: Max total samples.
        max_turns: Max context turns.

    Returns:
        List of dicts with 'context' and 'response' keys.
    """
    loaders = {
        "daily_dialog": ("test", load_dailydialog),
        "personachat": ("validation", load_personachat),  # no test split
        "empathetic_dialogues": ("test", load_empathetic_dialogues),
        "topical_chat": ("test", load_topical_chat),
        "wizard_of_wikipedia": ("test", load_wizard_of_wikipedia),
    }

    examples = []
    for name in datasets:
        if name not in loaders:
            print(f"Warning: no test loader for {name}, skipping")
            continue
        split, loader = loaders[name]
        print(f"Loading {name} ({split})...")
        data = loader(split=split)
        examples.extend(data)
        print(f"  {len(data)} examples")

    # Shuffle deterministically and truncate
    import random
    rng = random.Random(42)
    rng.shuffle(examples)
    examples = examples[:max_samples]

    # Format contexts
    formatted = []
    for ex in examples:
        ctx = ex["context"]
        if isinstance(ctx, list):
            ctx = ctx[-max_turns:]
            ctx = " [SEP] ".join(ctx)
        formatted.append({"context": ctx, "response": ex["response"]})

    return formatted


def evaluate_model(generator, model, tokenizer, test_data, model_type, device, gen_kwargs=None):
    """Run full evaluation.

    Args:
        generator: ChatBERTGenerator instance.
        model: Raw model (for perplexity).
        tokenizer: Tokenizer.
        test_data: List of dicts.
        model_type: 'encoder_decoder' or 'iterative_mlm'.
        device: Device string.
        gen_kwargs: Extra kwargs for generate().

    Returns:
        Dict with all metrics + example outputs.
    """
    gen_kwargs = gen_kwargs or {}
    contexts = [d["context"] for d in test_data]
    references = [d["response"] for d in test_data]

    # Generate responses
    print(f"Generating {len(contexts)} responses...")
    predictions = []
    start = time.time()
    for i, ctx in enumerate(contexts):
        pred = generator.generate(ctx, **gen_kwargs)
        predictions.append(pred)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  {i+1}/{len(contexts)} ({elapsed:.1f}s)")
    gen_time = time.time() - start
    print(f"Generation complete in {gen_time:.1f}s")

    # Compute text metrics
    print("Computing metrics...")
    metrics = compute_metrics(predictions, references)
    metrics.update(compute_response_length_stats(predictions))

    # Compute perplexity
    print("Computing perplexity...")
    if model_type == "encoder_decoder":
        ppl = compute_perplexity_ed(model, tokenizer, contexts, references, device=device)
    elif model_type == "iterative_mlm":
        ppl = compute_perplexity_imr(model, tokenizer, contexts, references, device=device)
    else:
        ppl = None

    if ppl is not None:
        metrics["perplexity"] = ppl

    metrics["generation_time_s"] = gen_time
    metrics["samples"] = len(contexts)

    # Collect examples (first 50)
    examples = []
    for i in range(min(50, len(contexts))):
        examples.append({
            "context": contexts[i],
            "reference": references[i],
            "prediction": predictions[i],
        })

    return {"metrics": metrics, "examples": examples}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a ChatBERT model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--model_type", type=str, required=True,
        choices=["encoder_decoder", "iterative_mlm", "gpt2_baseline"],
        help="Model type",
    )
    parser.add_argument("--output_path", type=str, default=None, help="Output JSON path")
    parser.add_argument("--max_samples", type=int, default=500, help="Max test samples")
    parser.add_argument("--max_length", type=int, default=None, help="Override max response length")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--num_iterations", type=int, default=None, help="IMR iterations")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=["daily_dialog", "personachat"],
        help="Datasets to evaluate on (default: daily_dialog personachat)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading {args.model_type} from {args.model_path}...")

    gen_kwargs = {}

    if args.model_type == "gpt2_baseline":
        from chatbert.baselines.gpt2_generator import GPT2Generator
        gen = GPT2Generator.from_pretrained(args.model_path, device=device)
        model = gen.model
        tokenizer = gen.tokenizer
        if args.max_length:
            gen.max_length = args.max_length
        if args.temperature:
            gen.temperature = args.temperature
    else:
        kwargs = {}
        if args.max_length:
            kwargs["max_length"] = args.max_length
        if args.temperature:
            kwargs["temperature"] = args.temperature
        gen = ChatBERTGenerator.from_pretrained(
            args.model_path, model_type=args.model_type, device=device, **kwargs,
        )
        model = gen.model
        tokenizer = gen.tokenizer

        if args.model_type == "iterative_mlm" and args.num_iterations:
            gen_kwargs["num_iterations"] = args.num_iterations

    # Load test data
    test_data = load_test_data(
        datasets=args.datasets,
        max_samples=args.max_samples,
    )
    print(f"Test samples: {len(test_data)}")

    # Evaluate
    results = evaluate_model(
        generator=gen,
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        model_type=args.model_type,
        device=device,
        gen_kwargs=gen_kwargs,
    )

    # Add metadata
    results["model"] = {
        "path": args.model_path,
        "type": args.model_type,
        "params": sum(p.numel() for p in model.parameters()),
    }

    # Save
    if args.output_path is None:
        model_name = Path(args.model_path).stem
        args.output_path = f"results/{model_name}.json"

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to {output_path}")
    print("\nMetrics:")
    for k, v in results["metrics"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
