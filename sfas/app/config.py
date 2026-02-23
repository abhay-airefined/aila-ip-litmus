from __future__ import annotations

from pydantic import Field
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

    model_config = SettingsConfigDict(env_prefix="SFAS_", extra="ignore")


settings = Settings()
