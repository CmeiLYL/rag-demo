from __future__ import annotations

from typing import Any, Dict, List, Optional

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from app.config import settings


class MilvusClient:
    """Milvus client wrapper for vector storage."""

    def __init__(self) -> None:
        self.collection_name = settings.milvus_collection_name
        self.collection: Optional[Collection] = None
        self.search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        self.connected = False

    def _ensure_connected(self) -> None:
        """Connect to Milvus lazily when the client is first used."""
        if self.connected:
            return
        try:
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port,
            )
            self.connected = True
        except Exception as exc:
            self.connected = False
            raise RuntimeError("Milvus service is not available") from exc

    def create_collection(self) -> None:
        """Create the Milvus collection if it does not exist."""
        self._ensure_connected()

        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        ]
        schema = CollectionSchema(fields, description="Knowledge base for RAG demo")
        self.collection = Collection(self.collection_name, schema=schema)
        self.collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 128},
            },
        )
        self.collection.load()

    def insert_documents(self, texts: List[str], embeddings: List[List[float]]) -> List[int]:
        """Insert text chunks and their embeddings into Milvus."""
        self._ensure_connected()

        if self.collection is None:
            self.create_collection()

        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings length must match")

        entities = [texts, embeddings]
        result = self.collection.insert(entities)
        self.collection.flush()
        return list(result.primary_keys)

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Search the Milvus collection and return matching text chunks."""
        try:
            self._ensure_connected()
        except RuntimeError:
            return []

        if self.collection is None:
            self.create_collection()

        self.collection.load()
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=self.search_params,
            limit=top_k,
            expr=None,
            output_fields=["text"],
        )
        hits = []
        if results and len(results) > 0:
            for hit in results[0]:
                hits.append({"text": hit.entity.get("text"), "score": float(hit.score)})
        return hits
