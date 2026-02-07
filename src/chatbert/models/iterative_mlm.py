"""ChatBERT Iterative MLM Refinement Model (ChatBERT-IMR).

This module implements the iterative masked language model refinement approach,
which generates text by starting with fully masked tokens and iteratively
unmasking based on prediction confidence - similar to discrete diffusion.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)


class ChatBERTIMRConfig(PretrainedConfig):
    """Configuration for ChatBERT Iterative MLM Refinement model."""

    model_type = "chatbert_imr"

    def __init__(
        self,
        backbone_name: str = "distilbert-base-uncased",
        hidden_size: int = 768,
        vocab_size: int = 30522,
        max_context_length: int = 256,
        max_response_length: int = 64,
        num_iterations: int = 10,
        mask_schedule: str = "confidence",  # confidence, linear, cosine
        initial_temperature: float = 1.0,
        final_temperature: float = 0.1,
        pad_token_id: int = 0,
        mask_token_id: int = 103,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.backbone_name = backbone_name
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_context_length = max_context_length
        self.max_response_length = max_response_length
        self.num_iterations = num_iterations
        self.mask_schedule = mask_schedule
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.mask_token_id = mask_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id


class ChatBERTIterativeMLM(PreTrainedModel):
    """ChatBERT with Iterative MLM Refinement for conversational AI.

    This model generates responses by:
    1. Starting with a fully masked response template
    2. Iteratively predicting and unmasking tokens based on confidence
    3. Re-evaluating predictions with newly revealed context
    4. Repeating until all tokens are unmasked

    This approach enables "deliberative generation" where the model can
    consider the entire response structure before finalizing predictions.
    """

    config_class = ChatBERTIMRConfig
    base_model_prefix = "chatbert_imr"

    def __init__(self, config: ChatBERTIMRConfig, _from_checkpoint: bool = False):
        super().__init__(config)
        self.config = config

        # Load BERT with MLM head
        if _from_checkpoint:
            # When loading from checkpoint, use config-only init (weights loaded later)
            from transformers import AutoConfig
            backbone_config = AutoConfig.from_pretrained(config.backbone_name)
            self.bert = AutoModelForMaskedLM.from_config(backbone_config)
        else:
            # Fresh init: load pretrained backbone weights
            self.bert = AutoModelForMaskedLM.from_pretrained(config.backbone_name)

        # Length predictor: predict response length given context
        self.length_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.max_response_length),
        )

        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Override to avoid nested from_pretrained calls."""
        kwargs["_from_checkpoint"] = True
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        response_ids: Optional[torch.Tensor] = None,
        response_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            input_ids: Context input IDs [batch, context_len].
            attention_mask: Context attention mask.
            response_ids: Masked response IDs [batch, response_len].
            response_attention_mask: Response attention mask.
            labels: Target labels (unmasked response).
            return_dict: Whether to return dict.

        Returns:
            Dictionary with loss and logits.
        """
        batch_size = input_ids.size(0)

        # Concatenate context and response
        if response_ids is not None:
            # Add [SEP] between context and response
            sep_token = torch.full(
                (batch_size, 1),
                self.config.sep_token_id,
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

            combined_ids = torch.cat([input_ids, sep_token, response_ids], dim=1)

            if attention_mask is not None and response_attention_mask is not None:
                sep_mask = torch.ones(batch_size, 1, device=attention_mask.device)
                combined_mask = torch.cat(
                    [attention_mask, sep_mask, response_attention_mask], dim=1
                )
            else:
                combined_mask = None

            # Create labels for the combined sequence
            if labels is not None:
                # Only compute loss on response positions (after context + [SEP])
                context_len = input_ids.size(1) + 1  # +1 for [SEP]
                combined_labels = torch.full_like(combined_ids, -100)
                combined_labels[:, context_len:] = labels
            else:
                combined_labels = None
        else:
            combined_ids = input_ids
            combined_mask = attention_mask
            combined_labels = labels

        # Forward through BERT
        outputs = self.bert(
            input_ids=combined_ids,
            attention_mask=combined_mask,
            labels=combined_labels,
            return_dict=True,
        )

        result = {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

        # Predict response length from [CLS] token
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            cls_hidden = outputs.hidden_states[-1][:, 0]
            length_logits = self.length_predictor(cls_hidden)
            result["length_logits"] = length_logits

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 64,
        num_iterations: Optional[int] = None,
        temperature: float = 1.0,
        mask_schedule: Optional[str] = None,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Generate response using iterative MLM refinement.

        Args:
            input_ids: Context input IDs [batch, context_len].
            attention_mask: Context attention mask.
            max_length: Maximum response length.
            num_iterations: Number of refinement iterations.
            temperature: Sampling temperature.
            mask_schedule: Masking schedule ('confidence', 'linear', 'cosine').
            top_k: Top-k sampling.
            top_p: Nucleus sampling threshold.

        Returns:
            Generated response IDs [batch, response_len].
        """
        num_iterations = num_iterations or self.config.num_iterations
        mask_schedule = mask_schedule or self.config.mask_schedule

        batch_size = input_ids.size(0)
        device = input_ids.device

        # Initialize response with all [MASK] tokens
        response_ids = torch.full(
            (batch_size, max_length),
            self.config.mask_token_id,
            device=device,
            dtype=input_ids.dtype,
        )

        # Add [SEP] between context and response
        sep_token = torch.full(
            (batch_size, 1),
            self.config.sep_token_id,
            device=device,
            dtype=input_ids.dtype,
        )

        # Track which positions are still masked
        is_masked = torch.ones(batch_size, max_length, dtype=torch.bool, device=device)

        # Iterative refinement
        for iteration in range(num_iterations):
            # Compute current temperature (anneal over iterations)
            t = iteration / max(num_iterations - 1, 1)
            current_temp = self.config.initial_temperature + t * (
                self.config.final_temperature - self.config.initial_temperature
            )

            # Combine context and current response
            combined_ids = torch.cat([input_ids, sep_token, response_ids], dim=1)

            if attention_mask is not None:
                sep_mask = torch.ones(batch_size, 1, device=device)
                response_mask = torch.ones(batch_size, max_length, device=device)
                combined_mask = torch.cat([attention_mask, sep_mask, response_mask], dim=1)
            else:
                combined_mask = None

            # Forward pass
            outputs = self.bert(
                input_ids=combined_ids,
                attention_mask=combined_mask,
                return_dict=True,
            )

            # Get logits for response positions
            context_len = input_ids.size(1) + 1  # +1 for [SEP]
            response_logits = outputs.logits[:, context_len:context_len + max_length]

            # Apply temperature and get probabilities
            probs = F.softmax(response_logits / current_temp, dim=-1)

            # Get predictions and confidence scores
            if top_k > 0:
                predictions = self._sample_top_k(probs, top_k)
            elif top_p < 1.0:
                predictions = self._sample_top_p(probs, top_p)
            else:
                predictions = probs.argmax(dim=-1)

            confidence = probs.max(dim=-1).values

            # Determine how many tokens to unmask this iteration
            num_masked = is_masked.sum(dim=1).float()
            if mask_schedule == "linear":
                unmask_ratio = (iteration + 1) / num_iterations
            elif mask_schedule == "cosine":
                unmask_ratio = 1 - torch.cos(
                    torch.tensor((iteration + 1) / num_iterations * 3.14159 / 2)
                ).item()
            else:  # confidence-based
                unmask_ratio = (iteration + 1) / num_iterations

            target_unmasked = int(max_length * unmask_ratio)

            # Update response: unmask positions with highest confidence
            for b in range(batch_size):
                if is_masked[b].sum() == 0:
                    continue

                masked_indices = torch.where(is_masked[b])[0]
                masked_confidence = confidence[b, masked_indices]

                # Number to unmask this iteration
                current_unmasked = (~is_masked[b]).sum().item()
                num_to_unmask = min(
                    len(masked_indices),
                    max(1, target_unmasked - current_unmasked),
                )

                # Select highest confidence positions
                _, top_indices = masked_confidence.topk(num_to_unmask)
                unmask_positions = masked_indices[top_indices]

                # Unmask selected positions
                response_ids[b, unmask_positions] = predictions[b, unmask_positions]
                is_masked[b, unmask_positions] = False

        # Final pass: fill any remaining masks
        if is_masked.any():
            combined_ids = torch.cat([input_ids, sep_token, response_ids], dim=1)
            outputs = self.bert(input_ids=combined_ids, return_dict=True)
            response_logits = outputs.logits[:, context_len:context_len + max_length]
            final_predictions = response_logits.argmax(dim=-1)
            response_ids = torch.where(is_masked, final_predictions, response_ids)

        return response_ids

    def _sample_top_k(
        self, probs: torch.Tensor, k: int
    ) -> torch.Tensor:
        """Sample from top-k probabilities."""
        top_k_probs, top_k_indices = probs.topk(k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Sample from top-k
        sampled_indices = torch.multinomial(
            top_k_probs.view(-1, k), num_samples=1
        ).view(probs.shape[:-1])

        # Get actual token IDs
        batch_indices = torch.arange(probs.size(0), device=probs.device)
        seq_indices = torch.arange(probs.size(1), device=probs.device)

        samples = top_k_indices[
            batch_indices.unsqueeze(1),
            seq_indices.unsqueeze(0),
            sampled_indices,
        ]
        return samples

    def _sample_top_p(
        self, probs: torch.Tensor, p: float
    ) -> torch.Tensor:
        """Sample using nucleus (top-p) sampling."""
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)

        # Remove tokens with cumulative probability > p
        mask = cumsum - sorted_probs > p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # Sample
        samples = torch.multinomial(
            sorted_probs.view(-1, sorted_probs.size(-1)), num_samples=1
        )
        samples = samples.view(probs.shape[:-1])

        # Get actual indices
        batch_size, seq_len = probs.shape[:2]
        batch_indices = torch.arange(batch_size, device=probs.device)
        seq_indices = torch.arange(seq_len, device=probs.device)

        result = sorted_indices[
            batch_indices.unsqueeze(1),
            seq_indices.unsqueeze(0),
            samples,
        ]
        return result

    @classmethod
    def from_config_file(cls, config_path: str) -> "ChatBERTIterativeMLM":
        """Load model from configuration file."""
        from chatbert.utils.config import load_config

        config_dict = load_config(config_path)

        model_config = ChatBERTIMRConfig(
            backbone_name=config_dict.model.backbone.pretrained,
            hidden_size=config_dict.model.backbone.hidden_size,
            vocab_size=config_dict.model.vocab_size,
            max_context_length=config_dict.model.max_context_length,
            max_response_length=config_dict.model.max_response_length,
            num_iterations=config_dict.model.refinement.num_iterations,
            mask_schedule=config_dict.model.refinement.mask_schedule,
            initial_temperature=config_dict.model.refinement.initial_temperature,
            final_temperature=config_dict.model.refinement.final_temperature,
        )

        return cls(model_config)
