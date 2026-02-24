from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from azure.core.exceptions import ResourceExistsError
from azure.data.tables import TableServiceClient
from azure.storage.blob import BlobServiceClient

from app.config import settings

logger = logging.getLogger("sfas")


class AzurePersistence:
    def __init__(self) -> None:
        self._blob_service: BlobServiceClient | None = None
        self._table_service: TableServiceClient | None = None

    def configured(self) -> bool:
        return bool(settings.azure_storage_connection_string)

    def _ensure_clients(self) -> tuple[BlobServiceClient, TableServiceClient]:
        if not self.configured():
            raise RuntimeError("SFAS_AZURE_STORAGE_CONNECTION_STRING is required for persistence")

        if self._blob_service is None:
            self._blob_service = BlobServiceClient.from_connection_string(settings.azure_storage_connection_string)
            container_client = self._blob_service.get_container_client(settings.azure_blob_container)
            try:
                container_client.create_container()
            except ResourceExistsError:
                pass
        if self._table_service is None:
            self._table_service = TableServiceClient.from_connection_string(settings.azure_storage_connection_string)
            self._table_service.create_table_if_not_exists(settings.azure_table_books)
            self._table_service.create_table_if_not_exists(settings.azure_table_agent_results)
            self._table_service.create_table_if_not_exists(settings.azure_table_aggregate_results)
        return self._blob_service, self._table_service

    @staticmethod
    def _json_default(value: Any) -> str | float | int | bool | None:
        if isinstance(value, (datetime,)):
            return value.isoformat()
        return str(value)

    def upload_json(self, blob_path: str, payload: dict[str, Any]) -> str:
        blob_service, _ = self._ensure_clients()
        blob_client = blob_service.get_blob_client(container=settings.azure_blob_container, blob=blob_path)
        data = json.dumps(payload, ensure_ascii=False, default=self._json_default).encode("utf-8")
        blob_client.upload_blob(data, overwrite=True)
        return blob_client.url

    def upsert_book_entity(self, book_id: str, metadata: dict[str, Any], blob_prefix: str) -> None:
        _, table_service = self._ensure_clients()
        table = table_service.get_table_client(settings.azure_table_books)
        entity = {
            "PartitionKey": "BOOK",
            "RowKey": book_id,
            "blob_prefix": blob_prefix,
            "sha256": metadata.get("sha256", ""),
            "word_count": int(metadata.get("word_count", 0)),
            "sentence_count": int(metadata.get("sentence_count", 0)),
            "token_count": int(metadata.get("token_count", 0)),
            "page_count": int(metadata.get("page_count", 0)),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        table.upsert_entity(entity=entity, mode="MERGE")

    def upsert_agent_entity(self, book_id: str, agent_name: str, result: dict[str, Any], blob_path: str) -> None:
        _, table_service = self._ensure_clients()
        table = table_service.get_table_client(settings.azure_table_agent_results)
        entity = {
            "PartitionKey": book_id,
            "RowKey": f"{agent_name}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            "agent_name": agent_name,
            "p_value": float(result.get("p_value", 1.0)),
            "likelihood_ratio": float(result.get("likelihood_ratio", 1.0)),
            "evidence_direction": str(result.get("evidence_direction", "unknown")),
            "blob_path": blob_path,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        table.create_entity(entity=entity)

    def upsert_aggregate_entity(self, book_id: str, result: dict[str, Any], blob_path: str) -> None:
        _, table_service = self._ensure_clients()
        table = table_service.get_table_client(settings.azure_table_aggregate_results)
        entity = {
            "PartitionKey": book_id,
            "RowKey": datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f'),
            "posterior_probability": float(result.get("posterior_probability", 0.0)),
            "log_likelihood_ratio": float(result.get("log_likelihood_ratio", 0.0)),
            "strength_of_evidence": str(result.get("strength_of_evidence", "No Evidence")),
            "blob_path": blob_path,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        table.create_entity(entity=entity)


azure_persistence = AzurePersistence()
