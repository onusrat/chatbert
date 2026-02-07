#!/usr/bin/env python3
"""Push ChatBERT models to HuggingFace Hub.

Generates a model card with architecture, training details, eval metrics,
usage code, and limitations, then pushes model + tokenizer + card.

Usage:
    # Dry run (preview model card)
    python scripts/push_to_hub.py \
        --model_path checkpoints/chatbert-ed-small/final \
        --model_type encoder_decoder \
        --repo_name onusrat/chatbert-ed-small \
        --dry_run

    # Actually push
    python scripts/push_to_hub.py \
        --model_path checkpoints/chatbert-ed-small/final \
        --model_type encoder_decoder \
        --repo_name onusrat/chatbert-ed-small \
        --results_json results/ed_small.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


MODEL_CARDS = {
    "encoder_decoder": {
        "title": "ChatBERT-ED (Encoder-Decoder)",
        "architecture": (
            "DistilBERT encoder (6 layers, 768d) with a 4-layer GPT-style decoder (512d, 8 heads) "
            "and cross-attention at every decoder layer. Generates responses autoregressively. "
            "~100M total parameters."
        ),
        "usage_code": '''```python
from chatbert import ChatBERTEncoderDecoder
from chatbert.inference import ChatBERTGenerator

model = ChatBERTEncoderDecoder.from_pretrained("{repo_name}")
generator = ChatBERTGenerator(model)

response = generator.generate("Hello, how are you today?")
print(response)
# i am doing okay. how are you?
```''',
    },
    "iterative_mlm": {
        "title": "ChatBERT-IMR (Iterative MLM Refinement)",
        "architecture": (
            "DistilBERT (6 layers, 768d) with its MLM head. Generates responses by starting with "
            "all [MASK] tokens and iteratively unmasking the most confident predictions. "
            "Non-autoregressive 'deliberative' generation. ~66M parameters."
        ),
        "usage_code": '''```python
from chatbert import ChatBERTIterativeMLM
from chatbert.inference import ChatBERTGenerator

model = ChatBERTIterativeMLM.from_pretrained("{repo_name}")
generator = ChatBERTGenerator(model, max_length=10, temperature=0.4)

response = generator.generate("Hello, how are you today?", num_iterations=25)
print(response)
# i'm doing well, just got off work
```''',
    },
}


def generate_model_card(model_type, repo_name, results=None):
    """Generate a HuggingFace model card.

    Args:
        model_type: 'encoder_decoder' or 'iterative_mlm'.
        repo_name: HuggingFace repo name (e.g. 'onusrat/chatbert-ed-small').
        results: Optional dict from evaluate.py JSON.

    Returns:
        Model card string (markdown).
    """
    info = MODEL_CARDS[model_type]

    # YAML frontmatter
    card = f"""---
language: en
license: mit
library_name: transformers
tags:
  - chatbert
  - dialogue
  - conversational
  - bert
  - distilbert
datasets:
  - roskoN/dailydialog
  - AlekseyKorshuk/persona-chat
pipeline_tag: text-generation
---

# {info['title']}

## Overview

ChatBERT explores whether BERT's bidirectional attention can enable "deliberative" conversational
response generation. This model is part of the ChatBERT project:
[GitHub](https://github.com/onusrat/chatbert) | [Demo](https://onusrat.github.io/chatbert/)

## Architecture

{info['architecture']}

## Training

- **Data**: DailyDialog + PersonaChat (~207k dialogue examples)
- **Hardware**: 1x NVIDIA A100 (via [Prime Intellect](https://www.primeintellect.ai/))
- **Optimizer**: AdamW, linear schedule with 10% warmup
- **Precision**: FP16
- **Training time**: ~67 minutes

## Usage

{info['usage_code'].format(repo_name=repo_name)}

## Installation

```bash
pip install git+https://github.com/onusrat/chatbert.git
```

Or clone and install:

```bash
git clone https://github.com/onusrat/chatbert.git
cd chatbert
pip install -e ".[all]"
```
"""

    # Add metrics table if results are provided
    if results and "metrics" in results:
        metrics = results["metrics"]
        card += "\n## Evaluation Results\n\n"
        card += "| Metric | Score |\n|--------|-------|\n"

        metric_labels = [
            ("bleu", "BLEU"),
            ("rouge1", "ROUGE-1"),
            ("rouge2", "ROUGE-2"),
            ("rougeL", "ROUGE-L"),
            ("bertscore_f1", "BERTScore F1"),
            ("distinct_1", "Distinct-1"),
            ("distinct_2", "Distinct-2"),
            ("perplexity", "Perplexity"),
            ("avg_length", "Avg Response Length"),
        ]

        for key, label in metric_labels:
            val = metrics.get(key)
            if val is not None:
                if key in ("perplexity", "avg_length"):
                    card += f"| {label} | {val:.1f} |\n"
                else:
                    card += f"| {label} | {val:.4f} |\n"

    card += """
## Limitations

This model was trained exclusively on casual conversation datasets (DailyDialog, PersonaChat).
It cannot answer factual questions, follow instructions, or discuss topics outside of everyday
small talk. With ~66-100M parameters and ~207k chitchat examples, it has no mechanism for
world knowledge. It is a research prototype demonstrating architectural concepts, not a
production chatbot.

## Citation

```bibtex
@misc{chatbert2026,
  title={ChatBERT: Deliberative Response Generation via Bidirectional Encoders},
  author={Nusrat, Omar},
  year={2026},
  url={https://github.com/onusrat/chatbert}
}
```
"""

    return card


def main():
    parser = argparse.ArgumentParser(description="Push ChatBERT model to HuggingFace Hub")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["encoder_decoder", "iterative_mlm"], help="Model type")
    parser.add_argument("--repo_name", type=str, required=True, help="HuggingFace repo (user/model)")
    parser.add_argument("--results_json", type=str, default=None, help="Eval results JSON")
    parser.add_argument("--dry_run", action="store_true", help="Preview model card without pushing")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    args = parser.parse_args()

    # Load results if provided
    results = None
    if args.results_json:
        with open(args.results_json) as f:
            results = json.load(f)

    # Generate model card
    card = generate_model_card(args.model_type, args.repo_name, results)

    if args.dry_run:
        print("=" * 60)
        print("MODEL CARD PREVIEW")
        print("=" * 60)
        print(card)
        print("=" * 60)
        print(f"Model path: {args.model_path}")
        print(f"Repo: {args.repo_name}")
        print(f"Type: {args.model_type}")
        if results:
            print(f"Results: {args.results_json}")
        print("\nRun without --dry_run to push to Hub.")
        return

    # Load model and push
    from transformers import AutoTokenizer

    if args.model_type == "encoder_decoder":
        from chatbert.models.encoder_decoder import ChatBERTEncoderDecoder
        model = ChatBERTEncoderDecoder.from_pretrained(args.model_path)
    else:
        from chatbert.models.iterative_mlm import ChatBERTIterativeMLM
        model = ChatBERTIterativeMLM.from_pretrained(args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Save model card
    model_path = Path(args.model_path)
    readme_path = model_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(card)

    print(f"Pushing to {args.repo_name}...")
    model.push_to_hub(args.repo_name, private=args.private)
    tokenizer.push_to_hub(args.repo_name, private=args.private)

    # Push the README separately
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.repo_name,
    )

    print(f"Pushed to https://huggingface.co/{args.repo_name}")


if __name__ == "__main__":
    main()
