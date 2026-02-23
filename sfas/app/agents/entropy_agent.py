from __future__ import annotations

import numpy as np

from app.config import settings
from app.datasets.builders import shannon_entropy
from app.models.schemas import AgentResponse
from app.models.storage import BookRecord
from app.preprocessing.text_pipeline import word_tokenize
from app.utils.model_gateway import generate_model_continuations
from app.utils.statistics import clip_lr, permutation_pvalue, safe_log_lr


def run(book: BookRecord, model_name: str, sample_count: int) -> AgentResponse:
    prompts = [" ".join(seg[:10]) for seg in book.segment_tokens[:sample_count]]
    outputs = generate_model_continuations(model_name, prompts, book.tokens, max_tokens=60)
    output_entropy = np.array([shannon_entropy(word_tokenize(o)) for o in outputs], dtype=float)

    baseline = np.array(book.datasets["entropy"]["segment_entropy"], dtype=float)
    observed_diff = float(abs(np.mean(output_entropy) - np.mean(baseline)))

    rng = np.random.default_rng(settings.random_seed)
    null = []
    for _ in range(settings.bootstrap_iterations):
        a = rng.choice(baseline, size=len(output_entropy), replace=True)
        b = rng.choice(baseline, size=len(output_entropy), replace=True)
        null.append(float(abs(np.mean(a) - np.mean(b))))
    null_arr = np.array(null)

    p_value = permutation_pvalue(observed_diff, null_arr, greater=False)
    scale = float(np.mean(null_arr) + np.std(null_arr) + 1e-9)
    lr = float(np.exp(np.clip((scale - observed_diff) / scale, -8, 8)))
    lr = clip_lr(lr, settings.lr_min, settings.lr_max)

    return AgentResponse(
        agent_name="entropy",
        hypothesis_test={"H0": "Model was NOT trained on this book", "H1": "Model WAS trained on this book"},
        metrics={
            "observed_entropy_diff": observed_diff,
            "output_entropy_mean": float(np.mean(output_entropy)),
            "baseline_entropy_mean": float(np.mean(baseline)),
        },
        p_value=p_value,
        likelihood_ratio=lr,
        log_likelihood_ratio=safe_log_lr(lr),
        evidence_direction="supports_H1" if lr > 1 else "supports_H0",
    )
