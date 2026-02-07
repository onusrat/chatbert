#!/usr/bin/env python3
"""Download and prepare datasets for ChatBERT training."""

import argparse
from pathlib import Path

from datasets import load_dataset


SUPPORTED_DATASETS = [
    "daily_dialog",
    "personachat",
    "empathetic_dialogues",
    "smoltalk",
]


def download_dataset(name: str, cache_dir: Path) -> None:
    """Download a single dataset.

    Args:
        name: Dataset name.
        cache_dir: Cache directory.
    """
    print(f"Downloading {name}...")

    if name == "daily_dialog":
        dataset = load_dataset(
            "roskoN/dailydialog",
            cache_dir=str(cache_dir),
        )
    elif name == "personachat":
        dataset = load_dataset(
            "AlekseyKorshuk/persona-chat",
            cache_dir=str(cache_dir),
        )
    elif name == "empathetic_dialogues":
        dataset = load_dataset(
            "empathetic_dialogues",
            cache_dir=str(cache_dir),
            trust_remote_code=True,
        )
    elif name == "smoltalk":
        dataset = load_dataset(
            "HuggingFaceTB/smoltalk",
            "everyday-conversations",
            cache_dir=str(cache_dir),
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Print info
    print(f"  Splits: {list(dataset.keys())}")
    for split, data in dataset.items():
        print(f"  {split}: {len(data)} examples")

    print(f"  Saved to: {cache_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download ChatBERT datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["daily_dialog", "personachat"],
        choices=SUPPORTED_DATASETS,
        help="Datasets to download",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./data/cache",
        help="Cache directory for datasets",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all supported datasets",
    )

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    datasets = SUPPORTED_DATASETS if args.all else args.datasets

    print(f"Downloading {len(datasets)} dataset(s) to {cache_dir}")
    print("=" * 50)

    for name in datasets:
        try:
            download_dataset(name, cache_dir)
            print()
        except Exception as e:
            print(f"  Error downloading {name}: {e}")
            print()

    print("=" * 50)
    print("Download complete!")


if __name__ == "__main__":
    main()
