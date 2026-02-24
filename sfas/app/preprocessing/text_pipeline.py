from __future__ import annotations

import re
import unicodedata

TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[.,!?;:\-()\[\]\"`]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentence_tokenize(text: str) -> list[str]:
    if not text:
        return []
    pieces = SENTENCE_RE.split(text)
    return [s.strip() for s in pieces if s.strip()]


def word_tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def split_segments(tokens: list[str], k: int, min_segment_tokens: int) -> list[list[str]]:
    if not tokens:
        return []
    k = max(1, min(k, len(tokens) // max(min_segment_tokens, 1) or 1))
    k = max(1, k)
    seg_len = max(min_segment_tokens, len(tokens) // k)
    segments: list[list[str]] = []
    for i in range(0, len(tokens), seg_len):
        chunk = tokens[i : i + seg_len]
        if chunk:
            segments.append(chunk)
    return segments
