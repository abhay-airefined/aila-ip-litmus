from __future__ import annotations

import numpy as np

from app.config import settings
from app.models.schemas import AgentResponse
from app.utils.statistics import clip_lr, strength_from_log_lr


def fuse(agent_results: list[AgentResponse], prior_probability: float) -> dict:
    lrs = np.array([max(settings.lr_min, min(settings.lr_max, a.likelihood_ratio)) for a in agent_results], dtype=float)
    logs = np.log(lrs)
    p_values = np.array([a.p_value for a in agent_results], dtype=float)

    corr_penalty = 1.0
    if len(logs) > 1:
        corr = np.corrcoef(logs)
        upper = corr[np.triu_indices_from(corr, k=1)]
        mean_corr = float(np.nan_to_num(np.mean(np.abs(upper)), nan=0.0))
        corr_penalty = float(np.clip(1 - 0.35 * mean_corr, 0.4, 1.0))

    combined_log_lr = float(np.sum(logs) * corr_penalty)
    combined_lr = float(np.exp(np.clip(combined_log_lr, -30, 30)))

    posterior = (combined_lr * prior_probability) / (combined_lr * prior_probability + (1 - prior_probability))

    if np.all(p_values > 0.05):
        posterior = min(posterior, 0.65)

    rare_agent = next((a for a in agent_results if a.agent_name == "rare_phrase"), None)
    if rare_agent and rare_agent.metrics.get("binomial_test", {}).get("successes", 0) == 0:
        posterior = max(0.0, posterior - 0.08)
        combined_log_lr -= 0.15

    posterior = float(np.clip(posterior, 0.0, 1.0))
    strength = strength_from_log_lr(combined_log_lr)

    return {
        "posterior_probability": posterior,
        "log_likelihood_ratio": float(combined_log_lr),
        "strength_of_evidence": strength,
        "agent_breakdown": [a.model_dump() for a in agent_results],
    }
