"""Generation utilities for ChatBERT models."""

from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from chatbert.models.encoder_decoder import ChatBERTEncoderDecoder
from chatbert.models.iterative_mlm import ChatBERTIterativeMLM


class ChatBERTGenerator:
    """Generator class for ChatBERT models.

    Provides a unified interface for generating responses from
    different ChatBERT model variants.
    """

    def __init__(
        self,
        model: Union[ChatBERTEncoderDecoder, ChatBERTIterativeMLM],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: Optional[str] = None,
        max_length: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_beams: int = 1,
        repetition_penalty: float = 1.2,
        do_sample: bool = True,
    ):
        """Initialize generator.

        Args:
            model: ChatBERT model instance.
            tokenizer: Tokenizer (loaded from model config if not provided).
            device: Device to run inference on.
            max_length: Maximum response length.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling.
            num_beams: Number of beams for beam search.
            repetition_penalty: Penalty for repeated tokens.
            do_sample: Whether to sample or use greedy decoding.
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        if tokenizer is None:
            if hasattr(model.config, "encoder_name"):
                tokenizer_name = model.config.encoder_name
            elif hasattr(model.config, "backbone_name"):
                tokenizer_name = model.config.backbone_name
            else:
                tokenizer_name = "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = tokenizer

        # Generation settings
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample

        # Determine model type
        self.is_encoder_decoder = isinstance(model, ChatBERTEncoderDecoder)

    def generate(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Generate a response given context.

        Args:
            context: Input context. Can be:
                - A string (single turn)
                - A list of strings (multi-turn history)
                - A list of dicts with 'role' and 'content' keys
            max_length: Override default max length.
            temperature: Override default temperature.
            **kwargs: Additional generation arguments.

        Returns:
            Generated response string.
        """
        # Format context
        context_text = self._format_context(context)

        # Tokenize
        inputs = self.tokenizer(
            context_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        max_length = max_length or self.max_length
        temperature = temperature or self.temperature

        with torch.no_grad():
            if self.is_encoder_decoder:
                outputs = self._generate_encoder_decoder(
                    inputs, max_length, temperature, **kwargs
                )
            else:
                outputs = self._generate_iterative_mlm(
                    inputs, max_length, temperature, **kwargs
                )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()

    def _format_context(
        self, context: Union[str, List[str], List[Dict[str, str]]]
    ) -> str:
        """Format context into a single string."""
        if isinstance(context, str):
            return context

        if isinstance(context, list):
            if len(context) == 0:
                return ""

            # List of strings
            if isinstance(context[0], str):
                return " [SEP] ".join(context)

            # List of message dicts
            if isinstance(context[0], dict):
                turns = []
                for msg in context:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "user":
                        turns.append(f"User: {content}")
                    else:
                        turns.append(f"Assistant: {content}")
                return " [SEP] ".join(turns)

        return str(context)

    def _generate_encoder_decoder(
        self,
        inputs: Dict[str, torch.Tensor],
        max_length: int,
        temperature: float,
        **kwargs,
    ) -> torch.Tensor:
        """Generate using encoder-decoder model with custom autoregressive loop."""
        import torch.nn.functional as F

        device = inputs["input_ids"].device
        bos_id = self.tokenizer.cls_token_id or 101
        eos_id = self.tokenizer.sep_token_id or 102
        pad_id = self.tokenizer.pad_token_id or 0

        # Encode
        encoder_outputs = self.model.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )

        # Start with [CLS] token
        decoder_ids = torch.full((1, 1), bos_id, dtype=torch.long, device=device)

        for step in range(max_length - 1):
            decoder_mask = torch.ones_like(decoder_ids, dtype=torch.long)

            outputs = self.model(
                encoder_outputs=encoder_outputs,
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=decoder_ids,
                decoder_attention_mask=decoder_mask,
                return_dict=True,
            )

            # Get logits for the last position
            next_logits = outputs.logits[:, -1, :] / max(temperature, 1e-7)

            # Apply repetition penalty
            if self.repetition_penalty != 1.0:
                for token_id in decoder_ids[0]:
                    next_logits[0, token_id] /= self.repetition_penalty

            if self.do_sample:
                # Top-k filtering
                if self.top_k > 0:
                    topk_vals, _ = next_logits.topk(self.top_k)
                    next_logits[next_logits < topk_vals[:, -1:]] = float('-inf')

                # Top-p filtering
                if self.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove_mask = cum_probs > self.top_p
                    remove_mask[:, 1:] = remove_mask[:, :-1].clone()
                    remove_mask[:, 0] = False
                    sorted_logits[remove_mask] = float('-inf')
                    next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            decoder_ids = torch.cat([decoder_ids, next_token], dim=1)

            if next_token.item() == eos_id:
                break

        return decoder_ids

    def _generate_iterative_mlm(
        self,
        inputs: Dict[str, torch.Tensor],
        max_length: int,
        temperature: float,
        **kwargs,
    ) -> torch.Tensor:
        """Generate using iterative MLM refinement."""
        num_iterations = kwargs.pop("num_iterations", None)

        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            temperature=temperature,
            num_iterations=num_iterations,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        return outputs

    def chat(
        self,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
    ):
        """Interactive chat interface.

        Args:
            history: Optional conversation history.
            stream: Whether to stream responses (not supported yet).

        Returns:
            Generator for interactive chat.
        """
        if history is None:
            history = []

        print("ChatBERT Interactive Chat")
        print("Type 'quit' to exit, 'clear' to reset history")
        print("-" * 50)

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            if user_input.lower() == "clear":
                history = []
                print("History cleared.")
                continue

            # Add user message to history
            history.append({"role": "user", "content": user_input})

            # Generate response
            response = self.generate(history)
            print(f"ChatBERT: {response}")

            # Add assistant response to history
            history.append({"role": "assistant", "content": response})

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        model_type: str = "encoder_decoder",
        **kwargs,
    ) -> "ChatBERTGenerator":
        """Load generator from pretrained model.

        Args:
            model_path: Path to model directory or HuggingFace model name.
            model_type: Type of model ('encoder_decoder' or 'iterative_mlm').
            **kwargs: Additional arguments for generator.

        Returns:
            ChatBERTGenerator instance.
        """
        from pathlib import Path

        model_path = Path(model_path)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        # Load model
        if model_type == "encoder_decoder":
            model = ChatBERTEncoderDecoder.from_pretrained(str(model_path))
        elif model_type == "iterative_mlm":
            model = ChatBERTIterativeMLM.from_pretrained(str(model_path))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return cls(model=model, tokenizer=tokenizer, **kwargs)
