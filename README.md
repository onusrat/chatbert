<div align="center">

# `[CHAT]BERT`

**Deliberative response generation via bidirectional encoders**

Omar Nusrat

---

[Live Demo](https://onusrat.github.io/chatbert/) &nbsp;&middot;&nbsp; Paper (coming soon)

</div>

## Overview

ChatBERT is a family of approaches that adapt BERT — a bidirectional encoder — for conversational AI. Where GPT generates left-to-right, BERT sees all positions simultaneously. We explore whether this bidirectional "deliberation" can produce coherent conversational responses through architectural adaptation rather than autoregressive generation.

```
ChatGPT = GPT + RLHF  →  ChatBERT = BERT + ???
```

Two variants are implemented:

| Model | Approach | Parameters | Description |
|-------|----------|-----------|-------------|
| **ChatBERT-ED** | Encoder-Decoder | ~100M | DistilBERT encoder + 4-layer decoder with cross-attention. Generates autoregressively. |
| **ChatBERT-IMR** | Iterative MLM Refinement | ~66M | DistilBERT with MLM head. Response starts as all `[MASK]` tokens; most confident predictions are unmasked iteratively. |

## Architecture

### ChatBERT-ED

```
Dialogue Context → DistilBERT Encoder (6 layers, 768d) → Cross-Attention (4 layers) → Decoder (4 layers, 512d) → LM Head (30522 vocab)
```

### ChatBERT-IMR

```
t=0  [M] [M] [M] [M] [M] [M] [M] [M]
t=1  i'm [M] [M] [M] [M] [M] [M] work
t=2  i'm [M] well [M] [M] got [M] work
t=3  i'm doing well [M] just got [M] work
t=4  i'm doing well ,  just got off work
```

## Training

Both models trained on **DailyDialog + PersonaChat** (~207k dialogue examples) on a single A100 GPU.

| | ChatBERT-ED | ChatBERT-IMR |
|---|---|---|
| Backbone | DistilBERT (66M) | DistilBERT (66M) |
| Total Parameters | ~100M | ~66M |
| Epochs | 5 | 10 (stopped ~9.5) |
| Batch Size | 16 (accum 2) | 32 |
| Learning Rate | 5e-5 (encoder: 2e-5) | 5e-5 |
| Precision | FP16 | FP16 |
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

Both models were trained exclusively on casual conversation datasets. ChatBERT has no ability to answer factual questions, follow instructions, or discuss topics outside of everyday small talk. This is a limitation of the training data, not the architecture — a more diverse corpus would be needed to produce more informative responses, though model capacity remains a bottleneck at this scale.

## License

MIT
