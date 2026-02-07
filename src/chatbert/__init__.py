"""
ChatBERT: Conversational AI with Bidirectional Encoder Representations

This package implements multiple approaches for adapting BERT to conversational AI:
- ChatBERT-ED: Encoder-Decoder hybrid
- ChatBERT-IMR: Iterative MLM Refinement
- ChatBERT-RA: Retrieval-Augmented
- ChatBERT-FIB: Fill-in-the-Blank
"""

__version__ = "0.1.0"

from chatbert.models import ChatBERTEncoderDecoder, ChatBERTIterativeMLM
