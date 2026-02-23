from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.agents import distribution_agent, entropy_agent, rare_phrase_agent, semantic_agent, stylometric_agent
from app.aggregation.bayesian_fusion import fuse
from app.config import settings
from app.datasets.pipeline import build_structured_data
from app.extraction.text_extractor import UnsupportedFileTypeError, extract_text
from app.models.schemas import AgentRequest, AgentResponse, AggregateRequest, AggregateResponse, BookMetadata, UploadBookResponse
from app.models.storage import BookRecord, store
from app.preprocessing.text_pipeline import normalize_text, sentence_tokenize, split_segments, word_tokenize

app = FastAPI(title=settings.app_name, version=settings.app_version)


@app.post("/upload-book", response_model=UploadBookResponse)
async def upload_book(file: UploadFile = File(...)) -> UploadBookResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        raw_text, page_count = extract_text(file.filename or "uploaded", content)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    normalized = normalize_text(raw_text)
    sentences = sentence_tokenize(normalized)
    tokens = word_tokenize(normalized)
    segment_tokens = split_segments(tokens, settings.default_segments, settings.min_segment_tokens)
    segments = [" ".join(seg) for seg in segment_tokens]

    if len(tokens) < settings.min_segment_tokens:
        raise HTTPException(status_code=400, detail="Book is too short for forensic analysis")

    sha = hashlib.sha256(content).hexdigest()
    book_id = str(uuid.uuid4())
    metadata = BookMetadata(
        sha256=sha,
        word_count=len([t for t in tokens if t.isalnum()]),
        sentence_count=len(sentences),
        token_count=len(tokens),
        page_count=page_count,
        extraction_timestamp=datetime.now(timezone.utc),
    )

    datasets, graphs = build_structured_data(tokens, sentences, segment_tokens)
    record = BookRecord(
        book_id=book_id,
        metadata=metadata.model_dump(),
        raw_text=raw_text,
        normalized_text=normalized,
        sentences=sentences,
        tokens=tokens,
        segments=segments,
        segment_tokens=segment_tokens,
        datasets=datasets,
        graphs=graphs,
    )
    store.upsert_book(record)
    return UploadBookResponse(book_id=book_id, metadata=metadata)


def _get_book_or_404(book_id: str) -> BookRecord:
    try:
        return store.get_book(book_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/agents/rare-phrase", response_model=AgentResponse)
def rare_phrase_endpoint(request: AgentRequest) -> AgentResponse:
    book = _get_book_or_404(request.book_id)
    return rare_phrase_agent.run(book, request.model_name, request.sample_count)


@app.post("/agents/stylometric", response_model=AgentResponse)
def stylometric_endpoint(request: AgentRequest) -> AgentResponse:
    book = _get_book_or_404(request.book_id)
    return stylometric_agent.run(book, request.model_name, request.sample_count)


@app.post("/agents/distribution", response_model=AgentResponse)
def distribution_endpoint(request: AgentRequest) -> AgentResponse:
    book = _get_book_or_404(request.book_id)
    return distribution_agent.run(book, request.model_name, request.sample_count)


@app.post("/agents/entropy", response_model=AgentResponse)
def entropy_endpoint(request: AgentRequest) -> AgentResponse:
    book = _get_book_or_404(request.book_id)
    return entropy_agent.run(book, request.model_name, request.sample_count)


@app.post("/agents/semantic", response_model=AgentResponse)
def semantic_endpoint(request: AgentRequest) -> AgentResponse:
    book = _get_book_or_404(request.book_id)
    return semantic_agent.run(book, request.model_name, request.sample_count)


@app.post("/aggregate", response_model=AggregateResponse)
def aggregate_endpoint(request: AggregateRequest) -> AggregateResponse:
    book = _get_book_or_404(request.book_id)
    agent_results = [
        rare_phrase_agent.run(book, request.model_name, settings.model_max_outputs),
        stylometric_agent.run(book, request.model_name, settings.model_max_outputs),
        distribution_agent.run(book, request.model_name, settings.model_max_outputs),
        entropy_agent.run(book, request.model_name, settings.model_max_outputs),
        semantic_agent.run(book, request.model_name, settings.model_max_outputs),
    ]
    fused = fuse(agent_results, request.prior_probability)

    if fused["posterior_probability"] >= 0.5:
        statement = "there is sufficient evidence"
    else:
        statement = "there is not sufficient evidence"

    summary = (
        "Based on the statistical analysis, "
        f"{statement} to conclude that the uploaded book was used in training the specified AI model."
    )
    return AggregateResponse(
        posterior_probability=fused["posterior_probability"],
        log_likelihood_ratio=fused["log_likelihood_ratio"],
        strength_of_evidence=fused["strength_of_evidence"],
        agent_breakdown=fused["agent_breakdown"],
        executive_summary=summary,
    )
