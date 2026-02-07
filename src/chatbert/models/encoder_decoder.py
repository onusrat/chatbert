"""ChatBERT Encoder-Decoder Model (ChatBERT-ED).

This module implements the encoder-decoder variant of ChatBERT,
which combines a BERT encoder with a lightweight GPT-style decoder.
"""

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    GenerationMixin,
)
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput


class ChatBERTEDConfig(PretrainedConfig):
    """Configuration for ChatBERT Encoder-Decoder model."""

    model_type = "chatbert_ed"

    def __init__(
        self,
        encoder_name: str = "distilbert-base-uncased",
        encoder_hidden_size: int = 768,
        decoder_hidden_size: int = 512,
        decoder_num_layers: int = 4,
        decoder_num_attention_heads: int = 8,
        decoder_intermediate_size: int = 2048,
        vocab_size: int = 30522,
        max_position_embeddings: int = 256,
        hidden_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        pad_token_id: int = 0,
        bos_token_id: int = 101,  # [CLS] for BERT
        eos_token_id: int = 102,  # [SEP] for BERT
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.encoder_name = encoder_name
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_layers = decoder_num_layers
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_intermediate_size = decoder_intermediate_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.num_hidden_layers = decoder_num_layers
        self.num_attention_heads = decoder_num_attention_heads
        self.is_encoder_decoder = True


class DecoderLayer(nn.Module):
    """Single decoder layer with self-attention and cross-attention."""

    def __init__(self, config: ChatBERTEDConfig):
        super().__init__()
        self.hidden_size = config.decoder_hidden_size
        self.num_heads = config.decoder_num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=config.attention_dropout_prob,
            batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(self.hidden_size)

        # Cross-attention to encoder
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=config.attention_dropout_prob,
            batch_first=True,
            kdim=config.encoder_hidden_size,
            vdim=config.encoder_hidden_size,
        )
        self.cross_attn_norm = nn.LayerNorm(self.hidden_size)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, config.decoder_intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.decoder_intermediate_size, self.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.ffn_norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Decoder hidden states [batch, seq, hidden].
            encoder_hidden_states: Encoder outputs [batch, enc_seq, enc_hidden].
            attention_mask: Decoder attention mask.
            encoder_attention_mask: Encoder attention mask.
            is_causal: Whether to apply causal masking.

        Returns:
            Updated hidden states.
        """
        # Self-attention with causal mask
        residual = hidden_states
        hidden_states = self.self_attn_norm(hidden_states)

        # Create causal mask
        seq_len = hidden_states.size(1)
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
                diagonal=1
            )
        else:
            causal_mask = None

        hidden_states, _ = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=causal_mask,
            key_padding_mask=attention_mask,
            need_weights=False,
        )
        hidden_states = residual + hidden_states

        # Cross-attention
        residual = hidden_states
        hidden_states = self.cross_attn_norm(hidden_states)
        hidden_states, _ = self.cross_attn(
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states,
            key_padding_mask=encoder_attention_mask,
            need_weights=False,
        )
        hidden_states = residual + hidden_states

        # Feed-forward
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class ChatBERTDecoder(nn.Module):
    """GPT-style decoder with cross-attention to encoder."""

    def __init__(self, config: ChatBERTEDConfig):
        super().__init__()
        self.config = config

        # Token embeddings (shared with encoder via tie_word_embeddings)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.decoder_hidden_size)

        # Position embeddings
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings, config.decoder_hidden_size
        )

        # Projection from encoder hidden size if different
        if config.encoder_hidden_size != config.decoder_hidden_size:
            self.encoder_proj = nn.Linear(
                config.encoder_hidden_size, config.decoder_hidden_size
            )
        else:
            self.encoder_proj = nn.Identity()

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.decoder_num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.decoder_hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """Forward pass.

        Args:
            input_ids: Decoder input IDs [batch, seq].
            encoder_hidden_states: Encoder outputs.
            attention_mask: Decoder attention mask (1 for valid, 0 for pad).
            encoder_attention_mask: Encoder attention mask.
            past_key_values: Cached key-values for generation.
            use_cache: Whether to return cache.

        Returns:
            Tuple of (hidden_states, cache).
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        token_embeds = self.embed_tokens(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device)
        position_embeds = self.embed_positions(positions)

        hidden_states = self.dropout(token_embeds + position_embeds)

        # Convert attention masks (1 -> 0 for valid, 0 -> 1 for masked, for PyTorch)
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask).bool()
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1.0 - encoder_attention_mask).bool()

        # Decoder layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )

        hidden_states = self.final_norm(hidden_states)

        return hidden_states, None


class ChatBERTEncoderDecoder(PreTrainedModel, GenerationMixin):
    """ChatBERT Encoder-Decoder model for conversational AI.

    Combines a pretrained BERT encoder with a lightweight GPT-style decoder
    for response generation.
    """

    config_class = ChatBERTEDConfig
    base_model_prefix = "chatbert"
    supports_gradient_checkpointing = True
    _tied_weights_keys = {"lm_head.weight": "decoder.embed_tokens.weight"}

    def __init__(self, config: ChatBERTEDConfig, _from_checkpoint: bool = False):
        super().__init__(config)
        self.config = config

        # Load encoder: use from_config when loading from checkpoint to avoid
        # nested from_pretrained conflict with meta device context
        if _from_checkpoint:
            from transformers import AutoConfig
            encoder_config = AutoConfig.from_pretrained(config.encoder_name)
            self.encoder = AutoModel.from_config(encoder_config)
        else:
            self.encoder = AutoModel.from_pretrained(config.encoder_name)

        # Initialize decoder
        self.decoder = ChatBERTDecoder(config)

        # Language model head (tied to decoder embeddings)
        self.lm_head = nn.Linear(config.decoder_hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.decoder.embed_tokens.weight

        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Override to avoid nested from_pretrained calls."""
        kwargs["_from_checkpoint"] = True
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        """Forward pass.

        Args:
            input_ids: Encoder input IDs [batch, enc_seq].
            attention_mask: Encoder attention mask.
            decoder_input_ids: Decoder input IDs [batch, dec_seq].
            decoder_attention_mask: Decoder attention mask.
            encoder_outputs: Precomputed encoder outputs.
            labels: Target labels for loss computation.
            past_key_values: Cached key-values.
            use_cache: Whether to return cache.
            output_attentions: Whether to output attention weights.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return dict.

        Returns:
            Seq2SeqLMOutput with loss, logits, and optional outputs.
        """
        # Encode if needed
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        encoder_hidden_states = encoder_outputs.last_hidden_state

        # Shift decoder inputs for teacher forcing
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs, cache = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # LM head
        logits = self.lm_head(decoder_outputs)

        # Compute loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + (encoder_hidden_states,)
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=cache,
            decoder_hidden_states=decoder_outputs if output_hidden_states else None,
            encoder_last_hidden_state=encoder_hidden_states,
        )

    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Shift input ids one position to the right for teacher forcing."""
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[:, 1:] = input_ids[:, :-1].clone()
        shifted[:, 0] = self.config.bos_token_id
        return shifted

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_outputs: Optional[BaseModelOutput] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare inputs for generation step."""
        return {
            "decoder_input_ids": decoder_input_ids,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

    @classmethod
    def from_config_file(cls, config_path: str) -> "ChatBERTEncoderDecoder":
        """Load model from configuration file.

        Args:
            config_path: Path to YAML config file.

        Returns:
            Initialized model.
        """
        from chatbert.utils.config import load_config

        config_dict = load_config(config_path)

        model_config = ChatBERTEDConfig(
            encoder_name=config_dict.model.encoder.pretrained,
            encoder_hidden_size=config_dict.model.encoder.hidden_size,
            decoder_hidden_size=config_dict.model.decoder.hidden_size,
            decoder_num_layers=config_dict.model.decoder.num_layers,
            decoder_num_attention_heads=config_dict.model.decoder.num_attention_heads,
            decoder_intermediate_size=config_dict.model.decoder.intermediate_size,
            vocab_size=config_dict.model.vocab_size,
            max_position_embeddings=config_dict.model.decoder.max_position_embeddings,
        )

        return cls(model_config)
