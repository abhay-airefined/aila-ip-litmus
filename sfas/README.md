# Scientific Forensic Attribution System (SFAS)

Scientific Forensic Attribution System (SFAS) is a FastAPI backend for forensic training-data attribution. It estimates whether an uploaded book was used to train a specified AI model using **internal-control methodology only**.

## Features

- Upload PDF or DOCX books.
- Internal baseline generation by segmenting the same book (no external control book).
- Automatic dataset construction:
  - Unigrams, bigrams, 5-grams, 20-grams.
  - Stylometric metrics.
  - Entropy profiles.
  - Word co-occurrence + sentence similarity graph metrics.
- Five independent forensic agents with hypothesis testing:
  - Rare phrase regeneration.
  - Stylometric similarity.
  - Distribution distance.
  - Entropy drift.
  - Semantic similarity.
- Bayesian fusion endpoint with correlation penalty and legal/statistical guardrails.

## Installation

```bash
cd sfas
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI:

- `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`
- Health check: `http://localhost:8000/health`

## LLM Provider Configuration (real providers only)

Simulation mode is disabled. You must configure a real provider:

- `SFAS_LLM_PROVIDER=openai`
- `SFAS_LLM_PROVIDER=azure_openai`
- `SFAS_LLM_PROVIDER=azure_foundry`

### OpenAI

```bash
export SFAS_LLM_PROVIDER=openai
export SFAS_OPENAI_API_KEY="<your-openai-key>"
# optional
export SFAS_OPENAI_BASE_URL="https://api.openai.com/v1"
```

`model_name` in API payloads is the model ID (example: `gpt-4o-mini`).

### Azure OpenAI

```bash
export SFAS_LLM_PROVIDER=azure_openai
export SFAS_AZURE_OPENAI_API_KEY="<your-azure-openai-key>"
export SFAS_AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
export SFAS_AZURE_OPENAI_API_VERSION="2024-10-21"
```

`model_name` in payloads must be your Azure OpenAI deployment name.

### Azure Foundry

```bash
export SFAS_LLM_PROVIDER=azure_foundry
export SFAS_AZURE_FOUNDRY_API_KEY="<your-foundry-key>"
export SFAS_AZURE_FOUNDRY_BASE_URL="<your-openai-compatible-foundry-base-url>"
```

`model_name` in payloads must match the deployed model/deployment name for your Foundry endpoint.

## Azure Persistence (Blob + Tables)

SFAS now persists outputs to Azure Storage:

- **Blob container**: book metadata, datasets, graph metrics, per-agent responses, aggregate response (JSON).
- **Azure Tables**:
  - `SfasBooks` for book-level indexing
  - `SfasAgentResults` for per-agent run summaries
  - `SfasAggregateResults` for aggregate run summaries

Required env vars:

```bash
export SFAS_AZURE_STORAGE_CONNECTION_STRING="<connection-string>"
# optional overrides
export SFAS_AZURE_BLOB_CONTAINER="sfas-artifacts"
export SFAS_AZURE_TABLE_BOOKS="SfasBooks"
export SFAS_AZURE_TABLE_AGENT_RESULTS="SfasAgentResults"
export SFAS_AZURE_TABLE_AGGREGATE_RESULTS="SfasAggregateResults"
```

By default, persistence is required at startup (`SFAS_PERSISTENCE_REQUIRED=true`).

## API Usage

### 1) Upload book

```bash
curl -X POST "http://localhost:8000/upload-book" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/book.pdf"
```

Returns:

```json
{
  "book_id": "...",
  "metadata": {
    "sha256": "...",
    "word_count": 120000,
    "sentence_count": 6500,
    "token_count": 145000,
    "page_count": 390,
    "extraction_timestamp": "2026-01-01T00:00:00Z"
  }
}
```

### 2) Run an agent

```bash
curl -X POST "http://localhost:8000/agents/rare-phrase" \
  -H "Content-Type: application/json" \
  -d '{"book_id":"<BOOK_ID>","model_name":"target-model","sample_count":20}'
```

Available agent endpoints:

- `/agents/rare-phrase`
- `/agents/stylometric`
- `/agents/distribution`
- `/agents/entropy`
- `/agents/semantic`

### 3) Aggregate forensic conclusion

```bash
curl -X POST "http://localhost:8000/aggregate" \
  -H "Content-Type: application/json" \
  -d '{"book_id":"<BOOK_ID>","model_name":"target-model","prior_probability":0.5}'
```

Returns:

```json
{
  "posterior_probability": 0.71,
  "log_likelihood_ratio": 1.42,
  "strength_of_evidence": "Strong",
  "agent_breakdown": [...],
  "executive_summary": "Based on the statistical analysis, there is sufficient evidence to conclude that the uploaded book was used in training the specified AI model."
}
```


## Troubleshooting uploads

If `/upload-book` appears to hang in Postman:

1. Verify server reachability first:

```bash
curl -i http://localhost:8000/health
```

2. Check request logs in the backend terminal. SFAS now logs request start/completion and upload size.
3. Ensure file type is `.pdf` or `.docx`.
4. Large files above the configured limit are rejected with HTTP 413 (`SFAS_MAX_UPLOAD_SIZE_MB`, default 50).
5. If the PDF is very large or image-heavy, extraction can take time; keep one request at a time and wait for `request.completed` log line.

## Statistical Methodology

### Hypothesis framework

Each agent tests:

- **H0**: The AI model was **not** trained on this book.
- **H1**: The AI model **was** trained on this book.

### Internal-control baseline (no external book)

The uploaded book is split into equal-length token segments. Null behavior under H0 is estimated from:

- Segment-to-segment variability.
- Cross-segment distributional distances.
- Bootstrapped / permutation-derived null distributions.

### Agent calibration

- Rare phrase agent uses binomial test + beta-binomial calibrated LR.
- Stylometric and semantic agents use permutation + density-ratio LR.
- Distribution and entropy agents use permutation-calibrated distance scores and bounded LR mappings.
- LRs are clipped to safe finite bounds to avoid pathological infinities.

### Bayesian fusion logic

Aggregate endpoint:

1. Computes per-agent LR and p-value.
2. Applies correlation penalty over log-LR dependence.
3. Combines evidence in log space.
4. Computes posterior probability:

\[
posterior = \frac{LR \cdot prior}{LR \cdot prior + (1-prior)}
\]

### Guardrails

- If all p-values > 0.05, posterior is capped at 0.65.
- If rare phrase exact matches are zero, negative evidentiary weight is applied.
- No infinite LR values; variance-collapse protections are enabled.
- Final executive summary is aligned with statistical conclusion language.

## Notes

- This implementation uses real provider-backed model calls (OpenAI/Azure OpenAI/Azure Foundry) configured through environment variables.
- All generated artifacts and results are persisted to Azure Blob and Azure Tables when persistence is enabled (default required).
