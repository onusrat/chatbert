<div align="center">

# `[CHAT]BERT`

**Deliberative response generation via bidirectional encoders**

**Omar Nusrat** &nbsp; [<img src="https://cdn.simpleicons.org/x/white" alt="X" width="14">](https://x.com/omarnusrat)

---

[Site](https://onusrat.github.io/chatbert/)

</div>

## Overview

ChatBERT is a family of approaches that adapt BERT — a bidirectional encoder — for conversational AI. Where GPT generates left-to-right, BERT sees all positions simultaneously. This explores whether this bidirectional "deliberation" can produce coherent conversational responses through architectural adaptation rather than autoregressive generation.

This is more an exercise on seeing if such a thing is possible than creating something equal in capabilities to the latest versions of ChatGPT or Claude.

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
│   │   ├── datasets.py          # DailyDialog, PersonaChat, SmolTalk loaders
│   │   └── preprocessing.py     # Tokenization & dialogue formatting
│   ├── inference/
│   │   └── generator.py         # Generation interface (beam search, iterative unmasking)
│   ├── baselines/
│   │   └── gpt2_generator.py    # GPT-2 baseline wrapper
│   └── utils/
│       ├── config.py            # YAML config loader
│       └── metrics.py           # Evaluation metrics (BLEU, ROUGE, BERTScore, perplexity)
├── scripts/
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Full evaluation pipeline
│   ├── analyze_imr.py           # IMR iteration analysis & visualization
│   ├── compare_results.py       # Multi-model comparison tables & charts
│   ├── train_gpt2_baseline.py   # GPT-2 baseline training
│   ├── run_ablations.py         # Ablation study runner
│   ├── push_to_hub.py           # HuggingFace Hub publishing
│   ├── download_data.py         # Download datasets
│   └── demo.py                  # Interactive chat demo
├── configs/
│   ├── ed_small.yaml            # ChatBERT-ED small config
│   ├── ed_base.yaml             # ChatBERT-ED base config
│   ├── imr_small.yaml           # ChatBERT-IMR small config
│   ├── gpt2_baseline.yaml       # GPT-2 baseline config
│   ├── ed_small_smoltalk.yaml   # ED + SmolTalk data
│   ├── imr_small_smoltalk.yaml  # IMR + SmolTalk data
│   └── ablations/               # Ablation study configs
│       ├── ed_frozen_encoder.yaml
│       ├── ed_decoder_depth_2.yaml
│       ├── ed_decoder_depth_6.yaml
│       ├── ed_lr_1e4.yaml
│       ├── ed_lr_1e5.yaml
│       └── ed_dailydialog_only.yaml
├── cloud/
│   ├── train_all.sh             # Full training & eval pipeline
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

### Training

| Metric | ChatBERT-ED | ChatBERT-IMR |
|--------|------------|-------------|
| Final Train Loss | 3.715 | 3.288 |
| Final Eval Loss | — | 3.184 |
| Total Steps | 32,425 | ~62,000 |
| Training Time | 67 min | 66 min |

### Evaluation

Evaluated on test splits of DailyDialog + PersonaChat (500 samples).

| Metric | ChatBERT-ED | ChatBERT-IMR | GPT-2 (baseline) |
|--------|------------|-------------|------------------|
| BLEU | 0.0021 | 0.0017 | 0.0042 |
| ROUGE-1 | 0.1577 | 0.1193 | 0.0677 |
| ROUGE-2 | 0.0269 | 0.0134 | 0.0094 |
| ROUGE-L | 0.1419 | 0.1082 | 0.0608 |
| BERTScore F1 | 0.8535 | 0.8459 | 0.8382 |
| Distinct-1 | 0.1701 | 0.2498 | 0.1986 |
| Distinct-2 | 0.4809 | 0.6719 | 0.6521 |
| Perplexity | 30.7 | 4.0 | — |
| Avg Length | 9.8 | 8.8 | 8.5 |

Both ChatBERT variants outperform the GPT-2 baseline on ROUGE and BERTScore despite having fewer parameters (66–100M vs 124M). IMR achieves the highest lexical diversity (Distinct-1/2) and lowest perplexity, while ED produces the most n-gram overlap with references. GPT-2 shows competitive diversity but weaker reference overlap, suggesting the BERT-based encoder provides stronger contextual grounding.

To evaluate locally:

```bash
python scripts/evaluate.py --model_path checkpoints/chatbert-ed-small/final --model_type encoder_decoder
python scripts/evaluate.py --model_path checkpoints/chatbert-imr-small/final --model_type iterative_mlm --max_length 10 --num_iterations 25
```

### Baseline Comparison

A GPT-2 (124M) model fine-tuned on the same data provides a standard autoregressive baseline:

```bash
python scripts/train_gpt2_baseline.py --config configs/gpt2_baseline.yaml
python scripts/evaluate.py --model_path checkpoints/gpt2-baseline/final --model_type gpt2_baseline
```

### Ablation Studies

Six ablation configs explore key design decisions. All ablations use the ChatBERT-ED architecture with one variable changed at a time:

| Ablation | Change | ROUGE-L | BERTScore | Distinct-2 | Perplexity |
|----------|--------|---------|-----------|------------|------------|
| **ED (default)** | — | **0.1419** | 0.8535 | 0.4809 | 30.7 |
| Frozen encoder | No encoder fine-tuning | 0.1478 | 0.8540 | 0.4831 | 30.2 |
| 2-layer decoder | Shallower decoder | 0.1492 | 0.8539 | 0.4742 | 32.2 |
| 6-layer decoder | Deeper decoder | 0.1425 | 0.8532 | 0.5036 | **29.3** |
| LR = 1e-4 | Higher learning rate | 0.1452 | **0.8542** | **0.5315** | 24.9 |
| LR = 1e-5 | Lower learning rate | 0.1230 | 0.8481 | 0.3397 | 55.8 |
| DailyDialog only | No PersonaChat | 0.1195 | 0.8491 | 0.4191 | 62.8 |

Key findings:
- **Frozen encoder still works**: Surprisingly competitive — DistilBERT's pretrained representations are already strong for dialogue understanding.
- **2-layer decoder suffices**: Minimal quality loss with half the decoder parameters, suggesting the encoder does most of the heavy lifting.
- **Higher LR helps diversity**: LR=1e-4 produces the most diverse outputs and lowest perplexity among ablations.
- **Data mix matters most**: Removing PersonaChat roughly doubles perplexity and significantly reduces diversity.

Run all ablations:

```bash
python scripts/run_ablations.py --all
```

### SmolTalk Data Experiment

Adding [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) (everyday-conversations) to the training mix:

| Metric | ED | ED + SmolTalk | IMR | IMR + SmolTalk |
|--------|-----|--------------|------|---------------|
| ROUGE-L | 0.1419 | 0.1323 | 0.1082 | 0.1020 |
| BERTScore F1 | 0.8535 | 0.8540 | 0.8459 | 0.8456 |
| Distinct-2 | 0.4809 | 0.4992 | 0.6719 | 0.6768 |
| Perplexity | 30.7 | 29.6 | 4.0 | 4.5 |

SmolTalk slightly improves diversity and perplexity for ED but is roughly neutral for IMR, suggesting the original DailyDialog + PersonaChat mix already provides sufficient coverage for these model sizes.

```bash
python scripts/train.py --config configs/ed_small_smoltalk.yaml
python scripts/train.py --config configs/imr_small_smoltalk.yaml
```

### IMR Analysis

The `analyze_imr.py` script provides deep analysis of the iterative refinement process:

- Per-iteration unmasking traces
- Quality vs number of iterations curves
- Mask schedule comparison (confidence vs linear vs cosine)
- Confidence evolution statistics

```bash
python scripts/analyze_imr.py --model_path checkpoints/chatbert-imr-small/final --prompt "Hello, how are you?" --run_full_analysis
```

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

## Models

| Model | HuggingFace | Params |
|-------|-------------|--------|
| ChatBERT-ED Small | [onusrat/chatbert-ed-small](https://huggingface.co/onusrat/chatbert-ed-small) | ~100M |
| ChatBERT-IMR Small | [onusrat/chatbert-imr-small](https://huggingface.co/onusrat/chatbert-imr-small) | ~66M |

Publish models to HuggingFace Hub:

```bash
python scripts/push_to_hub.py --model_path checkpoints/chatbert-ed-small/final --model_type encoder_decoder --repo_name onusrat/chatbert-ed-small
python scripts/push_to_hub.py --model_path checkpoints/chatbert-imr-small/final --model_type iterative_mlm --repo_name onusrat/chatbert-imr-small
```

## Limitations

Both models were trained exclusively on casual conversation datasets (DailyDialog, PersonaChat). ChatBERT has no ability to answer factual questions, follow instructions, or discuss topics outside of everyday small talk. This is a limitation of the training data, not the architecture — a more diverse training corpus would be needed to produce more informative responses, though model capacity remains a bottleneck at ~66–100M parameters.

## Acknowledgments

- Training compute provided by [Prime Intellect](https://www.primeintellect.ai/) (A100 GPU)
- Built on [DistilBERT](https://huggingface.co/distilbert-base-uncased) by HuggingFace
- Trained on [DailyDialog](http://yanran.li/dailydialog) and [PersonaChat](https://huggingface.co/datasets/personachat)

## License

[MIT](LICENSE)
