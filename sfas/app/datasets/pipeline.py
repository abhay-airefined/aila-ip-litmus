from __future__ import annotations

from app.config import settings
from app.datasets.builders import (
    build_segment_stylometry,
    ngram_counts,
    normalize_counter,
    percentile_rare_ngrams,
    rolling_entropy,
    shannon_entropy,
)
from app.graph.builders import build_sentence_similarity_graph, build_word_cooccurrence_graph, graph_metrics


def build_structured_data(tokens: list[str], sentences: list[str], segment_tokens: list[list[str]]) -> tuple[dict, dict]:
    unigram = ngram_counts(tokens, 1)
    bigram = ngram_counts(tokens, 2)
    fivegram = ngram_counts(tokens, 5)
    twentygram = ngram_counts(tokens, 20)

    segment_entropy = [shannon_entropy(seg) for seg in segment_tokens]
    rolling = rolling_entropy(tokens)
    stylometry_df = build_segment_stylometry(segment_tokens)

    datasets = {
        "ngrams": {
            "unigram": normalize_counter(unigram),
            "bigram": normalize_counter(bigram),
            "fivegram": normalize_counter(fivegram),
            "twentygram": normalize_counter(twentygram),
            "rare_twentygrams": [
                {"ngram": " ".join(k), "count": v}
                for k, v in percentile_rare_ngrams(twentygram, settings.rare_ngram_percentile)
            ],
        },
        "stylometry": stylometry_df.to_dict(orient="records"),
        "entropy": {
            "segment_entropy": segment_entropy,
            "rolling_entropy": rolling,
        },
    }

    word_graph = build_word_cooccurrence_graph(tokens, window=5)
    sentence_graph = build_sentence_similarity_graph(sentences)

    graphs = {
        "word_cooccurrence": graph_metrics(word_graph),
        "sentence_similarity": graph_metrics(sentence_graph),
    }
    return datasets, graphs
