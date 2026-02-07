"""Evaluation metrics for ChatBERT."""

from typing import Any, Dict, List, Optional

import numpy as np
import torch


def compute_perplexity_ed(model, tokenizer, contexts, references, device="cpu", max_context_length=256, max_response_length=128):
    """Compute perplexity for encoder-decoder model.

    Runs a forward pass on (context, reference) pairs and exponentiates
    the cross-entropy loss.

    Args:
        model: ChatBERTEncoderDecoder model.
        tokenizer: Tokenizer.
        contexts: List of context strings.
        references: List of reference response strings.
        device: Device string.
        max_context_length: Max context token length.
        max_response_length: Max response token length.

    Returns:
        Perplexity (float).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for ctx, ref in zip(contexts, references):
        ctx_enc = tokenizer(
            ctx, max_length=max_context_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        ref_enc = tokenizer(
            ref, max_length=max_response_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )

        input_ids = ctx_enc["input_ids"].to(device)
        attention_mask = ctx_enc["attention_mask"].to(device)
        decoder_input_ids = ref_enc["input_ids"].to(device)
        decoder_attention_mask = ref_enc["attention_mask"].to(device)

        # Build labels: shifted left by 1, pad → -100
        labels = torch.full_like(decoder_input_ids, -100)
        labels[:, :-1] = decoder_input_ids[:, 1:].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                return_dict=True,
            )

        # Count non-ignored tokens
        num_tokens = (labels != -100).sum().item()
        if num_tokens > 0:
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return float(np.exp(avg_loss))


def compute_perplexity_imr(model, tokenizer, contexts, references, device="cpu", max_context_length=256, max_response_length=64):
    """Compute pseudo-log-likelihood perplexity for iterative MLM model.

    Masks one response token at a time, sums log probabilities of the
    correct token at each masked position.

    Args:
        model: ChatBERTIterativeMLM model.
        tokenizer: Tokenizer.
        contexts: List of context strings.
        references: List of reference response strings.
        device: Device string.
        max_context_length: Max context token length.
        max_response_length: Max response token length.

    Returns:
        Pseudo-perplexity (float).
    """
    import torch.nn.functional as F

    model.eval()
    total_log_prob = 0.0
    total_tokens = 0

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id, pad_id}

    for ctx, ref in zip(contexts, references):
        ctx_enc = tokenizer(
            ctx, max_length=max_context_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        ref_enc = tokenizer(
            ref, max_length=max_response_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )

        context_ids = ctx_enc["input_ids"].to(device)
        context_mask = ctx_enc["attention_mask"].to(device)
        response_ids = ref_enc["input_ids"].squeeze(0).to(device)
        response_mask = ref_enc["attention_mask"].squeeze(0).to(device)

        # Find maskable positions (non-special, non-pad)
        maskable = []
        for pos in range(len(response_ids)):
            if response_ids[pos].item() not in special_ids and response_mask[pos].item() == 1:
                maskable.append(pos)

        if not maskable:
            continue

        # Mask one position at a time
        for pos in maskable:
            masked_response = response_ids.clone().unsqueeze(0)
            masked_response[0, pos] = mask_id

            # Build combined input: context [SEP] masked_response
            sep = torch.tensor([[tokenizer.sep_token_id]], device=device)
            combined_ids = torch.cat([context_ids, sep, masked_response], dim=1)
            sep_mask = torch.ones(1, 1, device=device)
            resp_mask = response_mask.unsqueeze(0).float()
            combined_mask = torch.cat([context_mask.float(), sep_mask, resp_mask], dim=1)

            with torch.no_grad():
                outputs = model.bert(
                    input_ids=combined_ids,
                    attention_mask=combined_mask,
                    return_dict=True,
                )

            # Get logits at the masked position (offset by context_len + 1 for [SEP])
            target_pos = context_ids.size(1) + 1 + pos
            logits = outputs.logits[0, target_pos]
            log_probs = F.log_softmax(logits, dim=-1)
            target_id = response_ids[pos].item()
            total_log_prob += log_probs[target_id].item()
            total_tokens += 1

    if total_tokens == 0:
        return float("inf")

    avg_neg_log_prob = -total_log_prob / total_tokens
    return float(np.exp(avg_neg_log_prob))


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute evaluation metrics for generated responses.

    Args:
        predictions: List of generated responses.
        references: List of reference responses.

    Returns:
        Dictionary of metric scores.
    """
    metrics = {}

    # BLEU score
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        refs_tokenized = [[r.split()] for r in references]
        preds_tokenized = [p.split() for p in predictions]
        smooth = SmoothingFunction().method1
        bleu_score = corpus_bleu(refs_tokenized, preds_tokenized, smoothing_function=smooth)
        metrics["bleu"] = float(bleu_score)
    except Exception as e:
        print(f"  Warning: BLEU computation failed: {e}")

    # ROUGE scores
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)
        for key in rouge_scores:
            metrics[key] = float(np.mean(rouge_scores[key]))
    except Exception as e:
        print(f"  Warning: ROUGE computation failed: {e}")

    # BERTScore
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(predictions, references, lang="en", verbose=False)
        metrics["bertscore_precision"] = float(P.mean())
        metrics["bertscore_recall"] = float(R.mean())
        metrics["bertscore_f1"] = float(F1.mean())
    except Exception as e:
        print(f"  Warning: BERTScore computation failed: {e}")

    # Distinct-n (diversity metrics)
    metrics.update(compute_distinct_n(predictions))

    return metrics


def compute_distinct_n(texts: List[str], ns: List[int] = [1, 2]) -> Dict[str, float]:
    """Compute Distinct-n diversity metrics.

    Args:
        texts: List of text strings.
        ns: N-gram sizes to compute.

    Returns:
        Dictionary with distinct-n scores.
    """
    results = {}

    for n in ns:
        all_ngrams = []
        total_ngrams = 0

        for text in texts:
            tokens = text.lower().split()
            ngrams = [
                tuple(tokens[i:i+n])
                for i in range(len(tokens) - n + 1)
            ]
            all_ngrams.extend(ngrams)
            total_ngrams += len(ngrams)

        if total_ngrams > 0:
            unique_ngrams = len(set(all_ngrams))
            results[f"distinct_{n}"] = unique_ngrams / total_ngrams
        else:
            results[f"distinct_{n}"] = 0.0

    return results


def compute_response_length_stats(texts: List[str]) -> Dict[str, float]:
    """Compute response length statistics.

    Args:
        texts: List of response strings.

    Returns:
        Dictionary with length statistics.
    """
    lengths = [len(t.split()) for t in texts]

    return {
        "avg_length": np.mean(lengths),
        "min_length": np.min(lengths),
        "max_length": np.max(lengths),
        "std_length": np.std(lengths),
    }
