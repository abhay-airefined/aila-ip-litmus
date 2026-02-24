from __future__ import annotations

import numpy as np

from app.agents.common import distribution_metrics
from app.config import settings
from app.models.schemas import AgentResponse
from app.models.storage import BookRecord
from app.preprocessing.text_pipeline import word_tokenize
from app.utils.model_gateway import generate_model_continuations
from app.utils.statistics import clip_lr, permutation_pvalue, safe_log_lr


def run(book: BookRecord, model_name: str, sample_count: int) -> AgentResponse:
    prompts = [" ".join(seg[:15]) for seg in book.segment_tokens[:sample_count]]
    outputs = generate_model_continuations(model_name, prompts, book.tokens, max_tokens=70)
    output_tokens = word_tokenize(" ".join(outputs))

    observed = distribution_metrics(book.tokens, output_tokens)
    observed_score = observed["js_distance"] + observed["wasserstein"] + observed["kl_divergence"]

    rng = np.random.default_rng(settings.random_seed)
    null_scores = []
    for _ in range(settings.permutation_iterations):
        idx_a = rng.integers(0, len(book.segment_tokens), size=max(2, len(book.segment_tokens) // 2))
        idx_b = rng.integers(0, len(book.segment_tokens), size=max(2, len(book.segment_tokens) // 2))
        a = [t for i in idx_a for t in book.segment_tokens[i]]
        b = [t for i in idx_b for t in book.segment_tokens[i]]
        m = distribution_metrics(a, b)
        null_scores.append(m["js_distance"] + m["wasserstein"] + m["kl_divergence"])
    null_scores = np.array(null_scores)
    p_value = permutation_pvalue(observed_score, null_scores, greater=False)

    null_mean = float(np.mean(null_scores))
    null_std = float(np.std(null_scores) + 1e-9)
    z = (null_mean - observed_score) / null_std
    lr = float(np.exp(np.clip(z, -8, 8)))
    lr = clip_lr(lr, settings.lr_min, settings.lr_max)

    return AgentResponse(
        agent_name="distribution",
        hypothesis_test={"H0": "Model was NOT trained on this book", "H1": "Model WAS trained on this book"},
        metrics={**observed, "effect_size": float(null_mean - observed_score)},
        p_value=p_value,
        likelihood_ratio=lr,
        log_likelihood_ratio=safe_log_lr(lr),
        evidence_direction="supports_H1" if lr > 1 else "supports_H0",
    )
