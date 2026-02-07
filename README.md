<div align="center">

# `[CHAT]BERT`

**Deliberative response generation via bidirectional encoders**

**Omar Nusrat** &nbsp; [<img src="https://cdn.simpleicons.org/x/white" alt="X" width="14">](https://x.com/omarnusrat)

---

[Live Demo](https://onusrat.github.io/chatbert/)

</div>

## Overview

ChatBERT is a family of approaches that adapt BERT — a bidirectional encoder — for conversational AI. Where GPT generates left-to-right, BERT sees all positions simultaneously. We explore whether this bidirectional "deliberation" can produce coherent conversational responses through architectural adaptation rather than autoregressive generation.

```
ChatGPT = GPT + RLHF  →  ChatBERT = BERT + ???
```

Two variants are implemented and trained:

| Model | Approach | Parameters | Description |
|-------|----------|-----------|-------------|
| **ChatBERT-ED** | Encoder-Decoder | ~100M | DistilBERT encoder + 4-layer decoder with cross-attention. Generates autoregressively. |
| **ChatBERT-IMR** | Iterative MLM Refinement | ~66M | DistilBERT with MLM head. Response starts as all `[MASK]` tokens; most confident predictions are unmasked iteratively. |

## Project Structure

```
chatbert/
├── src/chatbert/
│   ├── models/
│   │   ├── encoder_decoder.py   # ChatBERT-ED model architecture
│   │   └── iterative_mlm.py     # ChatBERT-IMR model architecture
│   ├── data/
│   │   ├── datasets.py          # DailyDialog + PersonaChat loaders & collators
│   │   └── preprocessing.py     # Tokenization & dialogue formatting
│   ├── inference/
│   │   └── generator.py         # Generation interface (beam search, iterative unmasking)
│   └── utils/
│       ├── config.py            # YAML config loader
│       └── metrics.py           # Evaluation metrics
├── scripts/
│   ├── train.py                 # Main training script
│   ├── download_data.py         # Download DailyDialog + PersonaChat
│   └── demo.py                  # Interactive chat demo
├── configs/
│   ├── ed_small.yaml            # ChatBERT-ED small config
│   ├── ed_base.yaml             # ChatBERT-ED base config
│   └── imr_small.yaml           # ChatBERT-IMR small config
├── cloud/
│   ├── setup_prime.sh           # Prime Intellect pod setup
│   ├── launch.sh                # Launch training on cloud GPU
│   └── train_cloud.sh           # Cloud training wrapper
├── docs/
│   └── index.html               # Live demo website
└── site/
    ├── index.html               # Full project landing page
    └── blog.md                  # Blog post with model output examples
```

## Architecture

### ChatBERT-ED (Encoder-Decoder)

```
Dialogue Context → DistilBERT Encoder (6 layers, 768d) → Cross-Attention (4 layers) → Decoder (4 layers, 512d) → LM Head (30522 vocab)
```

A DistilBERT encoder processes the dialogue context bidirectionally. A lightweight 4-layer decoder with cross-attention generates responses autoregressively, attending to the full encoder representation at every layer. The decoder accounts for ~20% of total parameters.

### ChatBERT-IMR (Iterative MLM Refinement)

```
t=0  [M] [M] [M] [M] [M] [M] [M] [M]
t=1  i'm [M] [M] [M] [M] [M] [M] work
t=2  i'm [M] well [M] [M] got [M] work
t=3  i'm doing well [M] just got [M] work
t=4  i'm doing well ,  just got off work
```

Uses a standard DistilBERT with its MLM head. The response starts as all `[MASK]` tokens. Each iteration, the model predicts all positions simultaneously; the most confident predictions are unmasked and fixed. This repeats until all tokens are revealed — a non-autoregressive "deliberative" process.

## Installation

```bash
git clone https://github.com/onusrat/chatbert.git
cd chatbert
pip install -e ".[all]"
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Usage

### Download Data

```bash
python scripts/download_data.py
```

Downloads DailyDialog and PersonaChat (~207k dialogue examples combined).

### Train

```bash
# Train ChatBERT-ED
python scripts/train.py --config configs/ed_small.yaml

# Train ChatBERT-IMR
python scripts/train.py --config configs/imr_small.yaml
```

### Interactive Demo

```bash
python scripts/demo.py
```

### Inference

```python
from chatbert import ChatBERTEncoderDecoder
from chatbert.inference import ChatBERTGenerator

model = ChatBERTEncoderDecoder.from_pretrained("chatbert/chatbert-ed-small")
generator = ChatBERTGenerator(model)

response = generator.generate("Hello, how are you today?")
print(response)
```

## Training

Both models were trained on a single **NVIDIA A100 GPU** provisioned through **[Prime Intellect](https://www.primeintellect.ai/)**, using the HuggingFace Trainer. The combined dataset (DailyDialog + PersonaChat) contains ~207k dialogue examples with up to 5 turns of context.

| | ChatBERT-ED | ChatBERT-IMR |
|---|---|---|
| Backbone | DistilBERT (66M) | DistilBERT (66M) |
| Total Parameters | ~100M | ~66M |
| Decoder | 4 layers, 512d, 8 heads | N/A |
| Max Context / Response | 256 / 128 tokens | 256 / 64 tokens |
| Epochs | 5 | 10 (early stopped ~9.5) |
| Batch Size | 16 (accum 2, effective 32) | 32 |
| Learning Rate | 5e-5 (encoder: 2e-5) | 5e-5 |
| Scheduler | Linear with 10% warmup | Linear with 10% warmup |
| Weight Decay | 0.01 | 0.01 |
| Precision | FP16 | FP16 |
| Early Stopping | Patience 3 | Patience 15 |
| Mask Schedule | N/A | Confidence-based (0.15–0.95) |
| Training Time | 67 min | 66 min |

## Results

| Metric | ChatBERT-ED | ChatBERT-IMR |
|--------|------------|-------------|
| Final Train Loss | 3.715 | 3.288 |
| Final Eval Loss | — | 3.184 |
| Total Steps | 32,425 | ~62,000 |

### Example Output

**ChatBERT-ED**
```
User: Hello, how are you?
Bot:  i am doing okay. how are you?

User: What is your favorite food?
Bot:  i like mexican food, and you?

User: I just got a new puppy!
Bot:  haha, dogs are fun! whats yours?
```

**ChatBERT-IMR**
```
User: Hello, how are you?
Bot:  i'm doing well, just got off work

User: Do you want to go to the movies?
Bot:  i'd like to see the movie titanic.

User: Tell me about yourself.
Bot:  i'm a school teacher for a living.
```

## Limitations

Both models were trained exclusively on casual conversation datasets (DailyDialog, PersonaChat). ChatBERT has no ability to answer factual questions, follow instructions, or discuss topics outside of everyday small talk. This is a limitation of the training data, not the architecture — a more diverse training corpus would be needed to produce more informative responses, though model capacity remains a bottleneck at ~66–100M parameters.

## Acknowledgments

- Training compute provided by [Prime Intellect](https://www.primeintellect.ai/) (A100 GPU)
- Built on [DistilBERT](https://huggingface.co/distilbert-base-uncased) by HuggingFace
- Trained on [DailyDialog](http://yanran.li/dailydialog) and [PersonaChat](https://huggingface.co/datasets/personachat)

## License

[MIT](LICENSE)
