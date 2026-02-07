#!/usr/bin/env python3
"""Training script for ChatBERT models."""

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import wandb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chatbert.models.encoder_decoder import ChatBERTEncoderDecoder, ChatBERTEDConfig
from chatbert.models.iterative_mlm import ChatBERTIterativeMLM, ChatBERTIMRConfig
from chatbert.data.datasets import CombinedDialogueDataset
from chatbert.data.preprocessing import DialogueCollator, MLMCollator
from chatbert.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train ChatBERT model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name for logging",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    model_type = config.model.type

    print(f"Training {config.model.name} ({model_type})")
    print(f"Config: {args.config}")

    # Initialize tokenizer
    if model_type == "encoder_decoder":
        encoder_name = config.model.encoder.pretrained
    else:
        encoder_name = config.model.backbone.pretrained

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    # Ensure tokenizer has required special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize model
    if model_type == "encoder_decoder":
        model_config = ChatBERTEDConfig(
            encoder_name=config.model.encoder.pretrained,
            encoder_hidden_size=config.model.encoder.hidden_size,
            decoder_hidden_size=config.model.decoder.hidden_size,
            decoder_num_layers=config.model.decoder.num_layers,
            decoder_num_attention_heads=config.model.decoder.num_attention_heads,
            decoder_intermediate_size=config.model.decoder.intermediate_size,
            vocab_size=config.model.vocab_size,
            max_position_embeddings=config.model.decoder.max_position_embeddings,
        )
        model = ChatBERTEncoderDecoder(model_config)
        collator = DialogueCollator(
            tokenizer=tokenizer,
            max_context_length=config.model.max_context_length,
            max_response_length=config.model.max_response_length,
        )
    elif model_type == "iterative_mlm":
        model_config = ChatBERTIMRConfig(
            backbone_name=config.model.backbone.pretrained,
            hidden_size=config.model.backbone.hidden_size,
            vocab_size=config.model.vocab_size,
            max_context_length=config.model.max_context_length,
            max_response_length=config.model.max_response_length,
            num_iterations=config.model.refinement.num_iterations,
            mask_schedule=config.model.refinement.mask_schedule,
        )
        model = ChatBERTIterativeMLM(model_config)
        collator = MLMCollator(
            tokenizer=tokenizer,
            max_context_length=config.model.max_context_length,
            max_response_length=config.model.max_response_length,
            mask_ratio_min=config.training.mask_ratio_min,
            mask_ratio_max=config.training.mask_ratio_max,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load datasets
    train_split = "train"
    val_split = "validation"

    train_dataset = CombinedDialogueDataset(
        dataset_names=list(config.training.datasets),
        tokenizer=tokenizer,
        split=train_split,
        max_context_length=config.model.max_context_length,
        max_response_length=config.model.max_response_length,
        max_turns=config.training.max_turns,
    )

    # For validation, use smaller subset
    eval_dataset = CombinedDialogueDataset(
        dataset_names=list(config.training.datasets),
        tokenizer=tokenizer,
        split=val_split,
        max_context_length=config.model.max_context_length,
        max_response_length=config.model.max_response_length,
        max_turns=config.training.max_turns,
        max_examples_per_dataset=1000,  # Limit eval size
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")

    # Training arguments
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
        gradient_checkpointing=config.training.gradient_checkpointing,
        logging_steps=config.training.logging_steps,
        eval_strategy="steps",
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if args.wandb_project else "none",
        run_name=args.wandb_run_name or config.model.name,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Initialize W&B
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or config.model.name,
            config=config.to_dict() if hasattr(config, 'to_dict') else dict(config),
        )

    # Callbacks
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=config.training.early_stopping_patience
        )
    ]

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    # Train
    if args.resume_from:
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # Save final model
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save config
    config_path = final_dir / "chatbert_config.yaml"
    from chatbert.utils.config import save_config
    save_config(config, config_path)

    print(f"Training complete! Model saved to {final_dir}")

    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
