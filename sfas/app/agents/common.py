from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.preprocessing.text_pipeline import word_tokenize


def segment_similarity_distribution(segments: list[str]) -> np.ndarray:
    if len(segments) < 2:
        return np.array([0.0])
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    mat = vec.fit_transform(segments)
    sim = cosine_similarity(mat)
    vals = sim[np.triu_indices_from(sim, k=1)]
    return vals if len(vals) else np.array([0.0])


def stylometric_vector(text: str) -> np.ndarray:
    tokens = word_tokenize(text)
    words = [t for t in tokens if t.isalnum()]
    sentence_lengths = [len(s.split()) for s in text.split(".") if s.strip()]
    mean_sentence = np.mean(sentence_lengths) if sentence_lengths else 0.0
    std_sentence = np.std(sentence_lengths) if sentence_lengths else 0.0
    mean_word = np.mean([len(w) for w in words]) if words else 0.0
    ttr = len(set(words)) / max(1, len(words))
    return np.array([mean_sentence, std_sentence, mean_word, ttr], dtype=float)


def distribution_metrics(source_tokens: list[str], target_tokens: list[str]) -> dict[str, float]:
    vocab = sorted(set(source_tokens) | set(target_tokens))
    s = np.array([source_tokens.count(v) for v in vocab], dtype=float)
    t = np.array([target_tokens.count(v) for v in vocab], dtype=float)
    if s.sum() == 0 or t.sum() == 0:
        return {"wasserstein": 0.0, "kl_divergence": 0.0, "js_distance": 0.0}
    sn = s / s.sum()
    tn = t / t.sum()
    return {
        "wasserstein": float(wasserstein_distance(sn, tn)),
        "kl_divergence": float(entropy(np.clip(sn, 1e-12, 1), np.clip(tn, 1e-12, 1))),
        "js_distance": float(jensenshannon(sn, tn)),
    }
