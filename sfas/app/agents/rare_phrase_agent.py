from __future__ import annotations

import numpy as np
from scipy.stats import binomtest

from app.config import settings
from app.models.schemas import AgentResponse
from app.models.storage import BookRecord
from app.utils.model_gateway import generate_model_continuations
from app.utils.statistics import beta_binomial_lr, bootstrap_ci, clip_lr, safe_log_lr


def run(book: BookRecord, model_name: str, sample_count: int) -> AgentResponse:
    rare = book.datasets["ngrams"]["rare_twentygrams"][:sample_count]
    if not rare:
        rare = [{"ngram": " ".join(book.tokens[i : i + 20]), "count": 1} for i in range(0, min(len(book.tokens) - 20, sample_count))]
    prompts = [" ".join(item["ngram"].split()[:10]) for item in rare]
    expected = [" ".join(item["ngram"].split()[10:20]) for item in rare]
    outputs = generate_model_continuations(model_name, prompts, book.tokens, max_tokens=10)

    exact = np.array([1 if o.strip() == e.strip() else 0 for o, e in zip(outputs, expected)], dtype=int)
    partial = np.array([len(set(o.split()) & set(e.split())) / max(1, len(e.split())) for o, e in zip(outputs, expected)])

    exact_rate = float(exact.mean()) if len(exact) else 0.0
    partial_rate = float(partial.mean()) if len(partial) else 0.0
    p_value = float(binomtest(int(exact.sum()), max(1, len(exact)), p=0.01, alternative="greater").pvalue)

    boot = []
    rng = np.random.default_rng(settings.random_seed)
    for _ in range(settings.bootstrap_iterations):
        sampled = rng.choice(exact, size=len(exact), replace=True) if len(exact) else np.array([0])
        boot.append(float(np.mean(sampled)))
    ci = bootstrap_ci(np.array(boot))

    lr = beta_binomial_lr(int(exact.sum()), max(1, len(exact)), p0=0.01)
    if exact.sum() == 0:
        lr *= 0.5
    lr = clip_lr(lr, settings.lr_min, settings.lr_max)

    return AgentResponse(
        agent_name="rare_phrase",
        hypothesis_test={"H0": "Model was NOT trained on this book", "H1": "Model WAS trained on this book"},
        metrics={
            "exact_match_rate": exact_rate,
            "partial_match_rate": partial_rate,
            "binomial_test": {"successes": int(exact.sum()), "trials": int(len(exact)), "p_value": p_value},
            "bootstrap_ci": {"lower": ci[0], "upper": ci[1]},
        },
        p_value=p_value,
        likelihood_ratio=lr,
        log_likelihood_ratio=safe_log_lr(lr),
        evidence_direction="supports_H1" if lr > 1 else "supports_H0",
    )
