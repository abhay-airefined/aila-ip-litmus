from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BookRecord:
    book_id: str
    metadata: dict[str, Any]
    raw_text: str
    normalized_text: str
    sentences: list[str]
    tokens: list[str]
    segments: list[str]
    segment_tokens: list[list[str]]
    datasets: dict[str, Any] = field(default_factory=dict)
    graphs: dict[str, Any] = field(default_factory=dict)


class InMemoryStore:
    def __init__(self) -> None:
        self._books: dict[str, BookRecord] = {}

    def upsert_book(self, record: BookRecord) -> None:
        self._books[record.book_id] = record

    def get_book(self, book_id: str) -> BookRecord:
        if book_id not in self._books:
            raise KeyError(f"book_id '{book_id}' not found")
        return self._books[book_id]

    def has_book(self, book_id: str) -> bool:
        return book_id in self._books


store = InMemoryStore()
