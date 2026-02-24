from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Scientific Forensic Attribution System"
    app_version: str = "1.0.0"
    default_segments: int = 50
    min_segment_tokens: int = 150
    rare_ngram_percentile: float = 10.0
    lr_min: float = 1e-6
    lr_max: float = 1e6
    bootstrap_iterations: int = 400
    permutation_iterations: int = 300
    random_seed: int = 42
    model_max_outputs: int = 40
    max_upload_size_mb: int = 50
    log_level: str = "INFO"

    llm_provider: str = "openai"  # openai | azure_openai | azure_foundry
    llm_temperature: float = 0.0
    llm_max_tokens: int = 128

    openai_api_key: str | None = None
    openai_base_url: str | None = None

    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str = "2024-10-21"

    azure_foundry_api_key: str | None = None
    azure_foundry_base_url: str | None = None

    azure_storage_connection_string: str | None = None
    azure_blob_container: str = "sfas-artifacts"
    azure_table_books: str = "SfasBooks"
    azure_table_agent_results: str = "SfasAgentResults"
    azure_table_aggregate_results: str = "SfasAggregateResults"
    persistence_required: bool = True

    model_config = SettingsConfigDict(env_prefix="SFAS_", extra="ignore")


settings = Settings()
