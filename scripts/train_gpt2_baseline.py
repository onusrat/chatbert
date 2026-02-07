#!/usr/bin/env python3
"""Train a GPT-2 baseline on the same dialogue data as ChatBERT.

Matches ChatBERT-ED training setup: 5 epochs, batch 16, accum 2, lr 5e-5,
linear schedule, fp16. Uses format: <context> [SEP] <response> <eos>

Usage:
    python scripts/train_gpt2_baseline.py --config configs/gpt2_baseline.yaml

    # Quick test
    python scripts/train_gpt2_baseline.py --config configs/gpt2_baseline.yaml --max_steps 10
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chatbert.data.datasets import CombinedDialogueDataset
from chatbert.utils.config import load_config


class GPT2DialogueDataset(Dataset):
    """Wraps CombinedDialogueDataset for GPT-2 causal LM training.

    Formats each example as: <context> [SEP] <response> <eos>
    and tokenizes for causal LM training (labels = input_ids).
    """

    def __init__(self, dialogue_dataset, tokenizer, max_length=512):
        self.dialogue_dataset = dialogue_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialogue_dataset)

    def __getitem__(self, idx):
        item = self.dialogue_dataset[idx]
        context = item["context"]
        response = item["response"]

        text = f"{context} [SEP] {response}{self.tokenizer.eos_token}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels: same as input_ids for causal LM, mask padding with -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Also mask the context portion so we only train on the response
        # Find [SEP] token position (the text " [SEP] ")
        sep_text = " [SEP] "
        context_with_sep = context + sep_text
        context_tokens = self.tokenizer(
            context_with_sep, add_special_tokens=False, return_tensors="pt",
        )
        context_len = context_tokens["input_ids"].size(1)
        # Mask context tokens in labels
        labels[:context_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 baseline")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output dir")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps (for testing)")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Training GPT-2 baseline")
    print(f"Config: {args.config}")

    # Load tokenizer and model
    model_name = config.model.pretrained
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Load datasets using existing loaders
    max_length = config.model.get("max_length", 512)

    train_dialogue = CombinedDialogueDataset(
        dataset_names=list(config.training.datasets),
        tokenizer=tokenizer,
        split="train",
        max_turns=config.training.max_turns,
    )
    eval_dialogue = CombinedDialogueDataset(
        dataset_names=list(config.training.datasets),
        tokenizer=tokenizer,
        split="validation",
        max_turns=config.training.max_turns,
        max_examples_per_dataset=1000,
    )

    train_dataset = GPT2DialogueDataset(train_dialogue, tokenizer, max_length=max_length)
    eval_dataset = GPT2DialogueDataset(eval_dialogue, tokenizer, max_length=max_length)

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Training args
    output_dir = Path(args.output_dir) / config.model.name

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.scheduler,
        fp16=config.training.fp16,
        logging_steps=config.training.logging_steps,
        eval_strategy="steps",
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=4,
        max_steps=args.max_steps if args.max_steps else -1,
    )

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=config.training.early_stopping_patience,
        )
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )

    trainer.train()

    # Save
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()
