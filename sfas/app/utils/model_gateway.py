from __future__ import annotations

import random

from app.utils.statistics import stable_seed


def generate_model_continuations(
    model_name: str,
    prompts: list[str],
    corpus_tokens: list[str],
    max_tokens: int = 20,
) -> list[str]:
    seed = stable_seed(model_name, str(len(corpus_tokens)))
    rng = random.Random(seed)
    outputs: list[str] = []
    memorization_bias = 0.25 + (stable_seed(model_name) % 35) / 100
    for prompt in prompts:
        if corpus_tokens and rng.random() < memorization_bias:
            start = rng.randint(0, max(0, len(corpus_tokens) - max_tokens - 1))
            continuation = corpus_tokens[start : start + max_tokens]
            outputs.append(" ".join(continuation))
        else:
            generated = [rng.choice(corpus_tokens) if corpus_tokens else "token" for _ in range(max_tokens)]
            outputs.append(" ".join(generated))
    return outputs
