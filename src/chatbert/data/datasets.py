"""Dataset loading and processing for ChatBERT training."""

from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizer


def load_dailydialog(split: str = "train") -> List[Dict[str, Any]]:
    """Load DailyDialog dataset.

    Args:
        split: Dataset split ('train', 'validation', 'test').

    Returns:
        List of dialogue examples.
    """
    dataset = load_dataset("roskoN/dailydialog", split=split)

    examples = []
    for item in dataset:
        dialogue = item["utterances"]
        # Create context-response pairs from multi-turn dialogue
        for i in range(1, len(dialogue)):
            context = dialogue[:i]
            response = dialogue[i]
            examples.append({
                "context": context,
                "response": response,
                "source": "daily_dialog",
            })

    return examples


def load_personachat(split: str = "train") -> List[Dict[str, Any]]:
    """Load PersonaChat dataset.

    Args:
        split: Dataset split ('train', 'validation').

    Returns:
        List of dialogue examples.
    """
    dataset = load_dataset("AlekseyKorshuk/persona-chat", split=split)

    examples = []
    for item in dataset:
        # Each item has 'utterances' (list of dicts with 'candidates' and 'history')
        # and 'personality' (list of persona strings)
        persona = item.get("personality", [])
        for utt in item.get("utterances", []):
            history = utt.get("history", [])
            candidates = utt.get("candidates", [])
            if history and candidates:
                # Last candidate is the correct response
                response = candidates[-1]
                examples.append({
                    "context": history,
                    "response": response,
                    "persona": persona,
                    "source": "personachat",
                })

    return examples


def load_smoltalk(split: str = "train") -> List[Dict[str, Any]]:
    """Load SmolTalk dataset (HuggingFaceTB/smoltalk).

    Parses the messages format (list of role/content dicts) into
    context-response pairs.

    Args:
        split: Dataset split ('train', 'test').

    Returns:
        List of dialogue examples.
    """
    # SmolTalk only has train/test splits (no validation)
    if split == "validation":
        split = "test"
    dataset = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split=split)

    examples = []
    for item in dataset:
        messages = item.get("messages", [])
        if not messages or len(messages) < 2:
            continue

        # Build turns from message list
        turns = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            if not content:
                continue
            turns.append(content)

        # Create context-response pairs from consecutive turns
        for i in range(1, len(turns)):
            examples.append({
                "context": turns[:i],
                "response": turns[i],
                "source": "smoltalk",
            })

    return examples


def load_topical_chat(split: str = "train") -> List[Dict[str, Any]]:
    """Load Topical-Chat dataset.

    Uses agentlans/Conversational-Reasoning-Topical-Chat (Parquet, no loading script).
    Conversations are in ShareGPT format: list of dicts with 'from' and 'value' keys.

    Args:
        split: Dataset split ('train', 'validation', 'test').

    Returns:
        List of dialogue examples.
    """
    dataset = load_dataset(
        "agentlans/Conversational-Reasoning-Topical-Chat", split=split,
    )

    examples = []
    for item in dataset:
        conversations = item.get("conversations", [])
        # Extract turns (skip system messages)
        turns = []
        for msg in conversations:
            role = msg.get("from", "")
            value = msg.get("value", "").strip()
            if not value or role == "system":
                continue
            turns.append(value)

        for i in range(1, len(turns)):
            examples.append({
                "context": turns[:i],
                "response": turns[i],
                "source": "topical_chat",
            })

    return examples


def load_wizard_of_wikipedia(split: str = "train") -> List[Dict[str, Any]]:
    """Load Wizard of Wikipedia dataset.

    Uses chujiezheng/wizard_of_wikipedia (Parquet, no loading script).
    Each item has parallel lists: 'post' (user turns) and 'response' (wizard turns).

    Args:
        split: Dataset split ('train', 'validation', 'test').

    Returns:
        List of dialogue examples.
    """
    dataset = load_dataset("chujiezheng/wizard_of_wikipedia", split=split)

    examples = []
    for item in dataset:
        posts = item.get("post", [])
        responses = item.get("response", [])
        # Interleave posts and responses into a turn list
        turns = []
        for p, r in zip(posts, responses):
            if p and p.strip():
                turns.append(p.strip())
            if r and r.strip():
                turns.append(r.strip())

        for i in range(1, len(turns)):
            examples.append({
                "context": turns[:i],
                "response": turns[i],
                "source": "wizard_of_wikipedia",
            })

    return examples


def load_empathetic_dialogues(split: str = "train") -> List[Dict[str, Any]]:
    """Load EmpatheticDialogues dataset.

    Uses Estwld/empathetic_dialogues_llm (Parquet, no loading script).
    Each item has 'conversations' (list of role/content dicts), 'emotion', 'situation'.
    Split mapping: 'validation' -> 'valid', 'test' -> 'test'.

    Args:
        split: Dataset split ('train', 'validation', 'test').

    Returns:
        List of dialogue examples.
    """
    split_map = {"validation": "valid", "train": "train", "test": "test"}
    hf_split = split_map.get(split, split)
    dataset = load_dataset("Estwld/empathetic_dialogues_llm", split=hf_split)

    examples = []
    for item in dataset:
        conversations = item.get("conversations", [])
        emotion = item.get("emotion", "")

        # Extract turns
        turns = []
        for msg in conversations:
            content = msg.get("content", "").strip()
            if content:
                turns.append(content)

        for i in range(1, len(turns)):
            examples.append({
                "context": turns[:i],
                "response": turns[i],
                "emotion": emotion,
                "source": "empathetic_dialogues",
            })

    return examples


class DialogueDataset(Dataset):
    """PyTorch Dataset for dialogue data."""

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_context_length: int = 256,
        max_response_length: int = 128,
        max_turns: int = 5,
    ):
        """Initialize DialogueDataset.

        Args:
            examples: List of dialogue examples with 'context' and 'response' keys.
            tokenizer: HuggingFace tokenizer.
            max_context_length: Maximum context token length.
            max_response_length: Maximum response token length.
            max_turns: Maximum number of context turns to include.
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.max_response_length = max_response_length
        self.max_turns = max_turns

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Get context (last N turns)
        context = example["context"]
        if isinstance(context, list):
            context = context[-self.max_turns:]
            context_text = " [SEP] ".join(context)
        else:
            context_text = context

        response = example["response"]

        # Tokenize context (encoder input)
        context_encoding = self.tokenizer(
            context_text,
            max_length=self.max_context_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize response (decoder target)
        response_encoding = self.tokenizer(
            response,
            max_length=self.max_response_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Create labels for decoder (shift right happens in model)
        labels = response_encoding["input_ids"].clone()
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": context_encoding["input_ids"].squeeze(0),
            "attention_mask": context_encoding["attention_mask"].squeeze(0),
            "decoder_input_ids": response_encoding["input_ids"].squeeze(0),
            "decoder_attention_mask": response_encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


class CombinedDialogueDataset(Dataset):
    """Combined dataset from multiple dialogue sources."""

    def __init__(
        self,
        dataset_names: List[str],
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_context_length: int = 256,
        max_response_length: int = 128,
        max_turns: int = 5,
        max_examples_per_dataset: Optional[int] = None,
    ):
        """Initialize combined dataset.

        Args:
            dataset_names: List of dataset names to load.
            tokenizer: HuggingFace tokenizer.
            split: Data split to load.
            max_context_length: Maximum context length.
            max_response_length: Maximum response length.
            max_turns: Maximum context turns.
            max_examples_per_dataset: Optional limit per dataset.
        """
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.max_response_length = max_response_length
        self.max_turns = max_turns

        # Load all datasets
        all_examples = []
        dataset_loaders = {
            "daily_dialog": load_dailydialog,
            "personachat": load_personachat,
            "empathetic_dialogues": load_empathetic_dialogues,
            "smoltalk": load_smoltalk,
            "topical_chat": load_topical_chat,
            "wizard_of_wikipedia": load_wizard_of_wikipedia,
        }

        for name in dataset_names:
            if name not in dataset_loaders:
                raise ValueError(f"Unknown dataset: {name}")

            print(f"Loading {name}...")
            examples = dataset_loaders[name](split=split)

            if max_examples_per_dataset:
                examples = examples[:max_examples_per_dataset]

            all_examples.extend(examples)
            print(f"  Loaded {len(examples)} examples from {name}")

        self.examples = all_examples
        print(f"Total examples: {len(self.examples)}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]

        # Get context as text (collators handle tokenization)
        context = example["context"]
        if isinstance(context, list):
            context = context[-self.max_turns:]
            context_text = " [SEP] ".join(context)
        else:
            context_text = context

        response = example["response"]

        return {
            "context": context_text,
            "response": response,
        }
