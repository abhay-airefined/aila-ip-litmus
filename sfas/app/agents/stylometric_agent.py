from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from app.agents.common import stylometric_vector
from app.config import settings
from app.models.schemas import AgentResponse
from app.models.storage import BookRecord
from app.utils.model_gateway import generate_model_continuations
from app.utils.statistics import clip_lr, density_ratio, permutation_pvalue, safe_log_lr


def run(book: BookRecord, model_name: str, sample_count: int) -> AgentResponse:
    prompts = [" ".join(seg[:12]) for seg in book.segment_tokens[:sample_count]]
    outputs = generate_model_continuations(model_name, prompts, book.tokens, max_tokens=80)

    baseline_vectors = np.array([stylometric_vector(" ".join(seg)) for seg in book.segment_tokens], dtype=float)
    output_vectors = np.array([stylometric_vector(text) for text in outputs], dtype=float)

    centroid = baseline_vectors.mean(axis=0)
    observed_dist = float(np.mean(cdist(output_vectors, centroid.reshape(1, -1), metric="euclidean")))
    baseline_dist = np.linalg.norm(baseline_vectors - centroid, axis=1)

    rng = np.random.default_rng(settings.random_seed)
    null = []
    for _ in range(settings.permutation_iterations):
        sampled = baseline_vectors[rng.integers(0, len(baseline_vectors), size=len(output_vectors))]
        null.append(float(np.mean(cdist(sampled, centroid.reshape(1, -1), metric="euclidean"))))
    null_arr = np.array(null)

    p_value = permutation_pvalue(observed_dist, null_arr, greater=False)
    trained_samples = np.clip(null_arr * 0.8, 1e-9, None)
    lr = density_ratio(observed_dist, trained_samples, null_arr)
    lr = clip_lr(lr, settings.lr_min, settings.lr_max)

    return AgentResponse(
        agent_name="stylometric",
        hypothesis_test={"H0": "Model was NOT trained on this book", "H1": "Model WAS trained on this book"},
        metrics={
            "observed_distance": observed_dist,
            "baseline_distance_mean": float(np.mean(baseline_dist)),
            "permutation_null_mean": float(np.mean(null_arr)),
            "permutation_null_std": float(np.std(null_arr)),
        },
        p_value=p_value,
        likelihood_ratio=lr,
        log_likelihood_ratio=safe_log_lr(lr),
        evidence_direction="supports_H1" if lr > 1 else "supports_H0",
    )
