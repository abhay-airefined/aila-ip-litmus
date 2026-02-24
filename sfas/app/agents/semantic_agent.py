from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.agents.common import segment_similarity_distribution
from app.config import settings
from app.models.schemas import AgentResponse
from app.models.storage import BookRecord
from app.utils.model_gateway import generate_model_continuations
from app.utils.statistics import clip_lr, density_ratio, permutation_pvalue, safe_log_lr


def run(book: BookRecord, model_name: str, sample_count: int) -> AgentResponse:
    prompts = [" ".join(seg[:18]) for seg in book.segment_tokens[:sample_count]]
    outputs = generate_model_continuations(model_name, prompts, book.tokens, max_tokens=90)

    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    corpus = book.segments + outputs
    mat = vectorizer.fit_transform(corpus)
    seg_mat = mat[: len(book.segments)]
    out_mat = mat[len(book.segments) :]
    sim = cosine_similarity(out_mat, seg_mat)
    observed = float(np.mean(np.max(sim, axis=1))) if sim.size else 0.0

    baseline_dist = segment_similarity_distribution(book.segments)
    rng = np.random.default_rng(settings.random_seed)
    null = []
    for _ in range(settings.permutation_iterations):
        sample = rng.choice(baseline_dist, size=max(2, len(outputs)), replace=True)
        null.append(float(np.mean(sample)))
    null_arr = np.array(null)

    p_value = permutation_pvalue(observed, null_arr, greater=True)
    trained_samples = np.clip(null_arr * 1.15, 1e-9, 1.0)
    lr = density_ratio(observed, trained_samples, np.clip(null_arr, 1e-9, 1.0))
    lr = clip_lr(lr, settings.lr_min, settings.lr_max)

    return AgentResponse(
        agent_name="semantic",
        hypothesis_test={"H0": "Model was NOT trained on this book", "H1": "Model WAS trained on this book"},
        metrics={
            "observed_cosine_similarity": observed,
            "internal_similarity_mean": float(np.mean(baseline_dist)),
            "internal_similarity_std": float(np.std(baseline_dist)),
        },
        p_value=p_value,
        likelihood_ratio=lr,
        log_likelihood_ratio=safe_log_lr(lr),
        evidence_direction="supports_H1" if lr > 1 else "supports_H0",
    )
