#!/usr/bin/env python3
"""Run ablation studies for ChatBERT-ED.

Trains each ablation config, then evaluates the result. Collects all
results into a single JSON file for comparison.

Usage:
    # Run all ablations in configs/ablations/
    python scripts/run_ablations.py --all

    # Run specific configs
    python scripts/run_ablations.py --configs configs/ablations/ed_frozen_encoder.yaml configs/ablations/ed_decoder_depth_2.yaml

    # Dry run (just list what would be done)
    python scripts/run_ablations.py --all --dry_run
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def find_ablation_configs(configs_dir):
    """Find all YAML configs in the ablations directory."""
    configs_dir = Path(configs_dir)
    configs = sorted(configs_dir.glob("*.yaml"))
    return configs


def run_command(cmd, dry_run=False):
    """Run a shell command, printing it first."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n{'[DRY RUN] ' if dry_run else ''}$ {cmd_str}")

    if dry_run:
        return 0

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run ChatBERT ablation studies")
    parser.add_argument("--configs", nargs="*", default=None, help="Specific config files")
    parser.add_argument("--configs_dir", type=str, default="configs/ablations", help="Ablation configs dir")
    parser.add_argument("--all", action="store_true", help="Run all configs in --configs_dir")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Training output dir")
    parser.add_argument("--results_dir", type=str, default="results/ablations", help="Eval results dir")
    parser.add_argument("--max_samples", type=int, default=500, help="Eval samples per ablation")
    parser.add_argument("--skip_train", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation, only train")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    # Determine configs to run
    if args.all:
        configs = find_ablation_configs(args.configs_dir)
    elif args.configs:
        configs = [Path(c) for c in args.configs]
    else:
        print("Specify --all or --configs. Use --dry_run to preview.")
        return

    if not configs:
        print(f"No configs found in {args.configs_dir}")
        return

    print(f"Ablation configs ({len(configs)}):")
    for c in configs:
        print(f"  {c}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    scripts_dir = Path(__file__).parent

    for config_path in configs:
        config_name = config_path.stem
        print(f"\n{'='*60}")
        print(f"ABLATION: {config_name}")
        print(f"{'='*60}")

        model_dir = Path(args.output_dir) / config_name / "final"

        # Train
        if not args.skip_train:
            train_cmd = [
                sys.executable, str(scripts_dir / "train.py"),
                "--config", str(config_path),
                "--output_dir", args.output_dir,
            ]
            rc = run_command(train_cmd, dry_run=args.dry_run)
            if rc != 0 and not args.dry_run:
                print(f"Training failed for {config_name}, skipping eval")
                continue

        # Evaluate
        if not args.skip_eval:
            result_path = results_dir / f"{config_name}.json"
            eval_cmd = [
                sys.executable, str(scripts_dir / "evaluate.py"),
                "--model_path", str(model_dir),
                "--model_type", "encoder_decoder",
                "--output_path", str(result_path),
                "--max_samples", str(args.max_samples),
            ]
            rc = run_command(eval_cmd, dry_run=args.dry_run)

            if rc == 0 and not args.dry_run and result_path.exists():
                with open(result_path) as f:
                    all_results[config_name] = json.load(f)

    # Save combined results
    if all_results:
        combined_path = results_dir / "ablations.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined results saved to {combined_path}")

    # Print summary
    if all_results:
        print(f"\n{'='*60}")
        print("ABLATION SUMMARY")
        print(f"{'='*60}")
        print(f"{'Config':<35} {'BLEU':>8} {'ROUGE-L':>8} {'PPL':>8}")
        print("-" * 65)
        for name, data in all_results.items():
            m = data.get("metrics", {})
            bleu = m.get("bleu", 0)
            rougeL = m.get("rougeL", 0)
            ppl = m.get("perplexity", 0)
            print(f"{name:<35} {bleu:>8.4f} {rougeL:>8.4f} {ppl:>8.1f}")


if __name__ == "__main__":
    main()
