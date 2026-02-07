"""GPT-2 baseline generator for ChatBERT comparison.

Wraps a fine-tuned GPT-2 model with the same .generate(context) interface
as ChatBERTGenerator, so it can be used with evaluate.py.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class GPT2Generator:
    """Generator wrapper for fine-tuned GPT-2 baseline.

    Uses the format: <context> [SEP] <response> <eos>
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cpu",
        max_length: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

        # Separator token — we use [SEP] as text since GPT-2 tokenizer doesn't have it
        self.sep = " [SEP] "

    def generate(self, context: Union[str, List[str]], **kwargs) -> str:
        """Generate a response given context.

        Args:
            context: Input context string or list of turns.

        Returns:
            Generated response string.
        """
        if isinstance(context, list):
            context = self.sep.join(context)

        # Format: context [SEP]
        prompt = context + self.sep
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        prompt_len = input_ids.size(1)

        max_length = kwargs.get("max_length", self.max_length)
        temperature = kwargs.get("temperature", self.temperature)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Strip the prompt prefix to get just the response
        generated_ids = outputs[0][prompt_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Clean up: remove anything after a second [SEP] if the model generates one
        if "[SEP]" in response:
            response = response.split("[SEP]")[0]

        return response.strip()

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cpu", **kwargs) -> "GPT2Generator":
        """Load from a fine-tuned checkpoint.

        Args:
            model_path: Path to saved model directory.
            device: Device string.

        Returns:
            GPT2Generator instance.
        """
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(model=model, tokenizer=tokenizer, device=device, **kwargs)
