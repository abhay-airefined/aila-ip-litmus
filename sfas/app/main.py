from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.agents import distribution_agent, entropy_agent, rare_phrase_agent, semantic_agent, stylometric_agent
from app.aggregation.bayesian_fusion import fuse
from app.config import settings
from app.datasets.pipeline import build_structured_data
from app.extraction.text_extractor import UnsupportedFileTypeError, extract_text
from app.models.schemas import AgentRequest, AgentResponse, AggregateRequest, AggregateResponse, BookMetadata, UploadBookResponse
from app.models.storage import BookRecord, store
from app.preprocessing.text_pipeline import normalize_text, sentence_tokenize, split_segments, word_tokenize
from app.storage.azure_store import azure_persistence

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("sfas")

app = FastAPI(title=settings.app_name, version=settings.app_version)


@app.on_event("startup")
def startup_log() -> None:
    logger.info("sfas.startup llm_provider=%s max_upload_size_mb=%s", settings.llm_provider, settings.max_upload_size_mb)
    if settings.llm_provider.strip().lower() not in {"openai", "azure_openai", "azure_foundry"}:
        raise RuntimeError("SFAS_LLM_PROVIDER must be openai, azure_openai, or azure_foundry. Simulation mode is disabled.")
    if settings.persistence_required and not azure_persistence.configured():
        raise RuntimeError("Azure persistence is required. Set SFAS_AZURE_STORAGE_CONNECTION_STRING.")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    started = time.perf_counter()
    logger.info("request.started method=%s path=%s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.exception("request.failed method=%s path=%s elapsed_ms=%.2f", request.method, request.url.path, elapsed_ms)
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "request.completed method=%s path=%s status=%s elapsed_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


def _persist_book_artifacts(record: BookRecord) -> None:
    prefix = f"books/{record.book_id}"
    azure_persistence.upload_json(f"{prefix}/metadata.json", record.metadata)
    azure_persistence.upload_json(f"{prefix}/datasets.json", record.datasets)
    azure_persistence.upload_json(f"{prefix}/graphs.json", record.graphs)
    azure_persistence.upsert_book_entity(record.book_id, record.metadata, prefix)


def _persist_agent_result(book_id: str, response: AgentResponse) -> None:
    blob_path = f"books/{book_id}/agents/{response.agent_name}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}.json"
    azure_persistence.upload_json(blob_path, response.model_dump())
    azure_persistence.upsert_agent_entity(book_id, response.agent_name, response.model_dump(), blob_path)


def _persist_aggregate_result(book_id: str, response: AggregateResponse) -> None:
    blob_path = f"books/{book_id}/aggregate/{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}.json"
    azure_persistence.upload_json(blob_path, response.model_dump())
    azure_persistence.upsert_aggregate_entity(book_id, response.model_dump(), blob_path)


def _process_book_content(filename: str, content: bytes) -> tuple[BookMetadata, BookRecord]:
    raw_text, page_count = extract_text(filename, content)

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
    return metadata, record


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload-book", response_model=UploadBookResponse)
async def upload_book(file: UploadFile = File(...)) -> UploadBookResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(content) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max upload size is {settings.max_upload_size_mb} MB")

    filename = file.filename or "uploaded"
    logger.info("upload.received filename=%s size_bytes=%s", filename, len(content))

    try:
        metadata, record = await run_in_threadpool(_process_book_content, filename, content)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store.upsert_book(record)
    await run_in_threadpool(_persist_book_artifacts, record)
    logger.info("upload.processed book_id=%s tokens=%s", record.book_id, metadata.token_count)
    return UploadBookResponse(book_id=record.book_id, metadata=metadata)


def _get_book_or_404(book_id: str) -> BookRecord:
    try:
        return store.get_book(book_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/agents/rare-phrase", response_model=AgentResponse)
async def rare_phrase_endpoint(request: AgentRequest) -> AgentResponse:
    book = _get_book_or_404(request.book_id)
    response = await run_in_threadpool(rare_phrase_agent.run, book, request.model_name, request.sample_count)
    await run_in_threadpool(_persist_agent_result, request.book_id, response)
    return response


@app.post("/agents/stylometric", response_model=AgentResponse)
async def stylometric_endpoint(request: AgentRequest) -> AgentResponse:
    book = _get_book_or_404(request.book_id)
    response = await run_in_threadpool(stylometric_agent.run, book, request.model_name, request.sample_count)
    await run_in_threadpool(_persist_agent_result, request.book_id, response)
    return response


@app.post("/agents/distribution", response_model=AgentResponse)
async def distribution_endpoint(request: AgentRequest) -> AgentResponse:
    book = _get_book_or_404(request.book_id)
    response = await run_in_threadpool(distribution_agent.run, book, request.model_name, request.sample_count)
    await run_in_threadpool(_persist_agent_result, request.book_id, response)
    return response


@app.post("/agents/entropy", response_model=AgentResponse)
async def entropy_endpoint(request: AgentRequest) -> AgentResponse:
    book = _get_book_or_404(request.book_id)
    response = await run_in_threadpool(entropy_agent.run, book, request.model_name, request.sample_count)
    await run_in_threadpool(_persist_agent_result, request.book_id, response)
    return response


@app.post("/agents/semantic", response_model=AgentResponse)
async def semantic_endpoint(request: AgentRequest) -> AgentResponse:
    book = _get_book_or_404(request.book_id)
    response = await run_in_threadpool(semantic_agent.run, book, request.model_name, request.sample_count)
    await run_in_threadpool(_persist_agent_result, request.book_id, response)
    return response


@app.post("/aggregate", response_model=AggregateResponse)
async def aggregate_endpoint(request: AggregateRequest) -> AggregateResponse:
    book = _get_book_or_404(request.book_id)
    agent_results = [
        await run_in_threadpool(rare_phrase_agent.run, book, request.model_name, settings.model_max_outputs),
        await run_in_threadpool(stylometric_agent.run, book, request.model_name, settings.model_max_outputs),
        await run_in_threadpool(distribution_agent.run, book, request.model_name, settings.model_max_outputs),
        await run_in_threadpool(entropy_agent.run, book, request.model_name, settings.model_max_outputs),
        await run_in_threadpool(semantic_agent.run, book, request.model_name, settings.model_max_outputs),
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
    response = AggregateResponse(
        posterior_probability=fused["posterior_probability"],
        log_likelihood_ratio=fused["log_likelihood_ratio"],
        strength_of_evidence=fused["strength_of_evidence"],
        agent_breakdown=fused["agent_breakdown"],
        executive_summary=summary,
    )
    await run_in_threadpool(_persist_aggregate_result, request.book_id, response)
    return response
