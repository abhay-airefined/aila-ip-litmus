from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_word_cooccurrence_graph(tokens: list[str], window: int = 5) -> nx.Graph:
    graph = nx.Graph()
    for i, token in enumerate(tokens):
        graph.add_node(token)
        for j in range(i + 1, min(i + 1 + window, len(tokens))):
            other = tokens[j]
            if token == other:
                continue
            if graph.has_edge(token, other):
                graph[token][other]["weight"] += 1
            else:
                graph.add_edge(token, other, weight=1)
    return graph


def build_sentence_similarity_graph(sentences: list[str], threshold: float = 0.2) -> nx.Graph:
    graph = nx.Graph()
    if not sentences:
        return graph
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    matrix = vec.fit_transform(sentences)
    sim = cosine_similarity(matrix)
    for i, sentence in enumerate(sentences):
        graph.add_node(i, text=sentence)
        for j in range(i + 1, len(sentences)):
            if sim[i, j] >= threshold:
                graph.add_edge(i, j, weight=float(sim[i, j]))
    return graph


def graph_metrics(graph: nx.Graph) -> dict[str, object]:
    if graph.number_of_nodes() == 0:
        return {
            "clustering_coefficient": 0.0,
            "modularity": 0.0,
            "degree_distribution": [],
            "betweenness_centrality_mean": 0.0,
        }
    degrees = [d for _, d in graph.degree()]
    clustering = nx.average_clustering(graph) if graph.number_of_nodes() > 1 else 0.0
    communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
    modularity = nx.algorithms.community.modularity(graph, communities) if communities else 0.0
    betw = nx.betweenness_centrality(graph)
    return {
        "clustering_coefficient": float(clustering),
        "modularity": float(modularity),
        "degree_distribution": degrees,
        "betweenness_centrality_mean": float(np.mean(list(betw.values())) if betw else 0.0),
    }
