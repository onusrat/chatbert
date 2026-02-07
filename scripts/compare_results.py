#!/usr/bin/env python3
"""Compare evaluation results from multiple models.

Reads JSON result files from evaluate.py and produces:
- Markdown table for README/website
- Bar chart (matplotlib PNG)

Usage:
    python scripts/compare_results.py \
        results/ed_small.json results/imr_small.json results/gpt2_baseline.json \
        --output_dir results/comparison
"""

import argparse
import json
import sys
from pathlib import Path


# Metrics to include in comparison (in display order)
DISPLAY_METRICS = [
    ("bleu", "BLEU"),
    ("rouge1", "ROUGE-1"),
    ("rouge2", "ROUGE-2"),
    ("rougeL", "ROUGE-L"),
    ("bertscore_f1", "BERTScore F1"),
    ("distinct_1", "Distinct-1"),
    ("distinct_2", "Distinct-2"),
    ("perplexity", "Perplexity"),
    ("avg_length", "Avg Length"),
]


def load_results(paths):
    """Load result JSON files.

    Returns:
        List of (name, metrics_dict) tuples.
    """
    results = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        name = data.get("model", {}).get("type", Path(p).stem)
        # Use filename if type is generic
        if name in ("encoder_decoder", "iterative_mlm", "gpt2_baseline"):
            name = Path(p).stem
        results.append((name, data["metrics"]))
    return results


def generate_markdown_table(results):
    """Generate a markdown comparison table.

    Args:
        results: List of (name, metrics) tuples.

    Returns:
        Markdown string.
    """
    names = [r[0] for r in results]

    # Header
    header = "| Metric | " + " | ".join(names) + " |"
    separator = "|--------|" + "|".join(["--------"] * len(names)) + "|"

    rows = [header, separator]
    for key, label in DISPLAY_METRICS:
        vals = []
        for _, metrics in results:
            v = metrics.get(key)
            if v is None:
                vals.append("--")
            elif key == "perplexity":
                vals.append(f"{v:.1f}")
            elif key == "avg_length":
                vals.append(f"{v:.1f}")
            else:
                vals.append(f"{v:.4f}")
        row = f"| {label} | " + " | ".join(vals) + " |"
        rows.append(row)

    return "\n".join(rows)


def generate_bar_chart(results, output_path):
    """Generate a bar chart comparing metrics across models.

    Args:
        results: List of (name, metrics) tuples.
        output_path: Path to save PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping chart generation")
        return

    # Select metrics suitable for bar chart (0-1 scale)
    chart_metrics = [
        ("bleu", "BLEU"),
        ("rouge1", "ROUGE-1"),
        ("rougeL", "ROUGE-L"),
        ("bertscore_f1", "BERTScore F1"),
        ("distinct_1", "Distinct-1"),
        ("distinct_2", "Distinct-2"),
    ]

    names = [r[0] for r in results]
    n_metrics = len(chart_metrics)
    n_models = len(results)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = range(n_metrics)
    width = 0.8 / n_models
    colors = ["#007AFF", "#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77"]

    for i, (name, metrics) in enumerate(results):
        values = [metrics.get(k, 0) for k, _ in chart_metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], values, width,
                      label=name, color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in chart_metrics], rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("ChatBERT Model Comparison")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Chart saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare ChatBERT evaluation results")
    parser.add_argument("result_files", nargs="+", help="JSON result files from evaluate.py")
    parser.add_argument("--output_dir", type=str, default="results/comparison", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results(args.result_files)
    print(f"Loaded {len(results)} result files")

    # Generate markdown table
    table = generate_markdown_table(results)
    print("\n" + table)

    table_path = output_dir / "comparison_table.md"
    with open(table_path, "w") as f:
        f.write(table + "\n")
    print(f"\nTable saved to {table_path}")

    # Generate bar chart
    chart_path = output_dir / "comparison_chart.png"
    generate_bar_chart(results, chart_path)


if __name__ == "__main__":
    main()
