from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

FUNCTION_WORDS = {
    "the", "and", "of", "to", "in", "a", "is", "that", "for", "it", "as", "with", "on", "was", "at", "by", "an", "be", "this", "from",
}
PUNCTUATION = {".", ",", "!", "?", ";", ":", "-", "(", ")", "[", "]", '"', "`"}


def ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _safe_entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = np.array([v / total for v in counter.values()], dtype=float)
    return float(-(probs * np.log2(np.clip(probs, 1e-12, 1))).sum())


def shannon_entropy(tokens: list[str]) -> float:
    return _safe_entropy(Counter(tokens))


def rolling_entropy(tokens: list[str], window: int = 100) -> list[float]:
    if not tokens:
        return []
    if len(tokens) <= window:
        return [shannon_entropy(tokens)]
    out = []
    for i in range(0, len(tokens) - window + 1, max(1, window // 5)):
        out.append(shannon_entropy(tokens[i : i + window]))
    return out


def stylometric_features(tokens: list[str], sentences: list[str]) -> dict[str, float]:
    words = [t for t in tokens if t.isalnum() or "'" in t]
    sent_lengths = [len(s.split()) for s in sentences if s.strip()]
    word_lengths = [len(w) for w in words]
    counts = Counter(tokens)
    total = max(1, len(tokens))
    function_freq = sum(counts[w] for w in FUNCTION_WORDS) / total
    punct_freq = sum(counts[p] for p in PUNCTUATION) / total
    ttr = len(set(words)) / max(1, len(words))
    return {
        "sentence_length_mean": float(np.mean(sent_lengths)) if sent_lengths else 0.0,
        "sentence_length_std": float(np.std(sent_lengths)) if sent_lengths else 0.0,
        "word_length_mean": float(np.mean(word_lengths)) if word_lengths else 0.0,
        "word_length_std": float(np.std(word_lengths)) if word_lengths else 0.0,
        "type_token_ratio": float(ttr),
        "function_word_frequency": float(function_freq),
        "punctuation_frequency": float(punct_freq),
    }


def build_segment_stylometry(segment_tokens: list[list[str]]) -> pd.DataFrame:
    rows = []
    for seg in segment_tokens:
        seg_text = " ".join(seg)
        sentences = [s.strip() for s in seg_text.split(".") if s.strip()]
        rows.append(stylometric_features(seg, sentences))
    return pd.DataFrame(rows)


def normalize_counter(counter: Counter[tuple[str, ...]] | Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {" ".join(k) if isinstance(k, tuple) else k: v / total for k, v in counter.items()}


def percentile_rare_ngrams(counter: Counter[tuple[str, ...]], percentile: float) -> list[tuple[tuple[str, ...], int]]:
    if not counter:
        return []
    values = np.array(list(counter.values()), dtype=float)
    threshold = np.percentile(values, percentile)
    return [(ng, c) for ng, c in counter.items() if c <= threshold]
