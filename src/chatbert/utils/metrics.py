"""Evaluation metrics for ChatBERT."""

from typing import Any, Dict, List, Optional

import numpy as np


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
        from evaluate import load
        bleu = load("bleu")
        bleu_result = bleu.compute(
            predictions=predictions,
            references=[[r] for r in references],
        )
        metrics["bleu"] = bleu_result["bleu"]
    except Exception:
        pass

    # ROUGE scores
    try:
        from evaluate import load
        rouge = load("rouge")
        rouge_result = rouge.compute(
            predictions=predictions,
            references=references,
        )
        metrics["rouge1"] = rouge_result["rouge1"]
        metrics["rouge2"] = rouge_result["rouge2"]
        metrics["rougeL"] = rouge_result["rougeL"]
    except Exception:
        pass

    # BERTScore
    try:
        from evaluate import load
        bertscore = load("bertscore")
        bertscore_result = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
        )
        metrics["bertscore_precision"] = np.mean(bertscore_result["precision"])
        metrics["bertscore_recall"] = np.mean(bertscore_result["recall"])
        metrics["bertscore_f1"] = np.mean(bertscore_result["f1"])
    except Exception:
        pass

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
