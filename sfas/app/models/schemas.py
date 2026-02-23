from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BookMetadata(BaseModel):
    sha256: str
    word_count: int
    sentence_count: int
    token_count: int
    page_count: int
    extraction_timestamp: datetime


class UploadBookResponse(BaseModel):
    book_id: str
    metadata: BookMetadata


class AgentRequest(BaseModel):
    book_id: str
    model_name: str
    sample_count: int = Field(default=20, ge=5, le=80)


class AgentResponse(BaseModel):
    agent_name: str
    hypothesis_test: dict[str, Any]
    metrics: dict[str, Any]
    p_value: float
    likelihood_ratio: float
    log_likelihood_ratio: float
    evidence_direction: str


class AggregateRequest(BaseModel):
    book_id: str
    model_name: str
    prior_probability: float = Field(default=0.5, ge=1e-6, le=1 - 1e-6)


class AggregateResponse(BaseModel):
    posterior_probability: float
    log_likelihood_ratio: float
    strength_of_evidence: str
    agent_breakdown: list[dict[str, Any]]
    executive_summary: str
