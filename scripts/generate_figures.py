#!/usr/bin/env python3
"""Generate publication-quality figures from ChatBERT evaluation results.

Reads existing JSON/CSV results and produces 4 PNG figures:
1. Quality vs Iterations (IMR)
2. Confidence Evolution (IMR)
3. Ablation Comparison (ED)
4. IMR Unmasking Visualization

Usage:
    python scripts/generate_figures.py --output_dir results/figures
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Consistent color palette (matches compare_results.py)
COLORS = ["#007AFF", "#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77"]
RESULTS_DIR = Path(__file__).parent.parent / "results"


def fig_quality_vs_iterations(output_dir: Path):
    """Figure 1: Quality vs Iterations line plot.

    Shows how IMR response quality changes with more refinement steps.
    """
    data_path = RESULTS_DIR / "imr_analysis" / "quality_vs_iterations.json"
    with open(data_path) as f:
        data = json.load(f)

    iters = sorted(data.keys(), key=int)
    x = [int(i) for i in iters]
    rougeL = [data[i]["rougeL"] for i in iters]
    bertscore = [data[i]["bertscore_f1"] for i in iters]
    distinct2 = [data[i]["distinct_2"] for i in iters]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, rougeL, "o-", color=COLORS[0], linewidth=2, markersize=7, label="ROUGE-L")
    ax.plot(x, bertscore, "s-", color=COLORS[1], linewidth=2, markersize=7, label="BERTScore F1")
    ax.plot(x, distinct2, "^-", color=COLORS[2], linewidth=2, markersize=7, label="Distinct-2")

    ax.set_xlabel("Number of Iterations", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("IMR: Quality vs Number of Iterations", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    out = output_dir / "quality_vs_iterations.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def fig_confidence_evolution(output_dir: Path):
    """Figure 2: Confidence Evolution dual-axis plot.

    Shows unmasking dynamics averaged across 5 traces.
    """
    data_path = RESULTS_DIR / "imr_analysis" / "confidence_evolution.json"
    with open(data_path) as f:
        data = json.load(f)

    # Group by iteration and average
    by_iter = defaultdict(lambda: {"confidence": [], "masked": []})
    for entry in data:
        it = entry["iteration"]
        by_iter[it]["confidence"].append(entry["mean_confidence"])
        by_iter[it]["masked"].append(entry["num_masked"])

    iters = sorted(by_iter.keys())
    avg_conf = [np.mean(by_iter[i]["confidence"]) for i in iters]
    avg_masked = [np.mean(by_iter[i]["masked"]) for i in iters]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(iters, avg_conf, "o-", color=COLORS[0], linewidth=2, markersize=5, label="Mean Confidence")
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Mean Confidence", fontsize=12, color=COLORS[0])
    ax1.tick_params(axis="y", labelcolor=COLORS[0])
    ax1.set_ylim(-0.05, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(iters, avg_masked, "s-", color=COLORS[2], linewidth=2, markersize=5, label="Tokens Masked")
    ax2.set_ylabel("Tokens Still Masked", fontsize=12, color=COLORS[2])
    ax2.tick_params(axis="y", labelcolor=COLORS[2])
    ax2.set_ylim(-0.5, 11)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=11)

    ax1.set_title("IMR: Confidence Evolution During Generation", fontsize=14, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = output_dir / "confidence_evolution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def fig_ablation_comparison(output_dir: Path):
    """Figure 3: Ablation Comparison two-panel bar chart."""
    # Load default ED results
    with open(RESULTS_DIR / "chatbert_ed_small.json") as f:
        default_metrics = json.load(f)["metrics"]

    # Load ablation results
    ablation_files = [
        ("Frozen Encoder", "ed_frozen_encoder.json"),
        ("2-Layer Dec", "ed_decoder_depth_2.json"),
        ("6-Layer Dec", "ed_decoder_depth_6.json"),
        ("LR=1e-4", "ed_lr_1e4.json"),
        ("LR=1e-5", "ed_lr_1e5.json"),
        ("DD Only", "ed_dailydialog_only.json"),
    ]

    configs = [("Default", default_metrics)]
    for label, fname in ablation_files:
        with open(RESULTS_DIR / "ablations" / fname) as f:
            configs.append((label, json.load(f)["metrics"]))

    names = [c[0] for c in configs]
    n = len(names)

    # Left panel: 0-1 scale metrics
    score_metrics = [
        ("rougeL", "ROUGE-L"),
        ("bertscore_f1", "BERTScore F1"),
        ("distinct_2", "Distinct-2"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 1]})

    x = np.arange(len(score_metrics))
    width = 0.8 / n
    bar_colors = [COLORS[0], COLORS[1], COLORS[2], COLORS[3], COLORS[4], "#a17fe0", "#ff9a3c"]

    for i, (name, metrics) in enumerate(configs):
        values = [metrics.get(k, 0) for k, _ in score_metrics]
        offset = (i - n / 2 + 0.5) * width
        ax1.bar(x + offset, values, width, label=name, color=bar_colors[i], alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels([label for _, label in score_metrics], fontsize=11)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("Quality Metrics", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=8, ncol=2)
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y", alpha=0.3)

    # Right panel: Perplexity
    ppls = [m.get("perplexity", 0) for _, m in configs]
    bars = ax2.bar(range(n), ppls, color=bar_colors[:n], alpha=0.85)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Perplexity", fontsize=12)
    ax2.set_title("Perplexity", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("ChatBERT-ED Ablation Study", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = output_dir / "ablation_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def fig_imr_unmasking(output_dir: Path):
    """Figure 4: IMR Unmasking Visualization.

    Renders iterations 0-10 as styled monospace text showing token reveal.
    """
    trace_path = RESULTS_DIR / "imr_analysis" / "trace.csv"

    rows = []
    with open(trace_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Only show iterations 0-10 (the interesting part)
    rows = [r for r in rows if int(r["iteration"]) <= 10]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    # Header
    ax.text(0.02, 0.95, "IMR Unmasking: \"Hello, how are you?\"",
            fontsize=14, fontweight="bold", transform=ax.transAxes,
            fontfamily="monospace")

    # Parse which tokens are newly revealed each iteration
    prev_tokens = None
    y = 0.87
    for row in rows:
        iteration = int(row["iteration"])
        tokens_str = row["tokens"]
        tokens = tokens_str.split()
        newly_revealed = row.get("newly_revealed", "")
        newly_set = set()
        if newly_revealed:
            for idx in newly_revealed.split(","):
                idx = idx.strip()
                if idx:
                    newly_set.add(int(idx))

        # Build colored text using individual ax.text calls
        x_pos = 0.12
        iter_label = f"t={iteration:>2d}  "
        ax.text(0.02, y, iter_label, fontsize=11, fontfamily="monospace",
                color="#888888", transform=ax.transAxes, verticalalignment="top")

        for j, tok in enumerate(tokens):
            if tok == "[MASK]":
                color = "#999999"
            elif j in newly_set:
                color = "#00d4ff"  # cyan for newly revealed
            else:
                color = "#007AFF"  # blue for already revealed

            txt = ax.text(x_pos, y, tok + " ", fontsize=11, fontfamily="monospace",
                          color=color, transform=ax.transAxes, verticalalignment="top",
                          fontweight="bold" if j in newly_set else "normal")

            # Estimate width for next token positioning
            renderer = fig.canvas.get_renderer()
            bbox = txt.get_window_extent(renderer=renderer)
            bbox_axes = bbox.transformed(ax.transAxes.inverted())
            x_pos = bbox_axes.x1

        y -= 0.072

    # Legend
    ax.text(0.02, y - 0.02, "[MASK]", fontsize=10, fontfamily="monospace",
            color="#999999", transform=ax.transAxes)
    ax.text(0.12, y - 0.02, "= masked", fontsize=10, fontfamily="monospace",
            color="#666666", transform=ax.transAxes)
    ax.text(0.25, y - 0.02, "token", fontsize=10, fontfamily="monospace",
            color="#00d4ff", fontweight="bold", transform=ax.transAxes)
    ax.text(0.33, y - 0.02, "= newly revealed", fontsize=10, fontfamily="monospace",
            color="#666666", transform=ax.transAxes)
    ax.text(0.53, y - 0.02, "token", fontsize=10, fontfamily="monospace",
            color="#007AFF", transform=ax.transAxes)
    ax.text(0.61, y - 0.02, "= already revealed", fontsize=10, fontfamily="monospace",
            color="#666666", transform=ax.transAxes)

    plt.tight_layout()
    out = output_dir / "imr_unmasking.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Generate ChatBERT figures")
    parser.add_argument("--output_dir", type=str, default="results/figures",
                        help="Output directory for PNG files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    fig_quality_vs_iterations(output_dir)
    fig_confidence_evolution(output_dir)
    fig_ablation_comparison(output_dir)
    fig_imr_unmasking(output_dir)
    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
