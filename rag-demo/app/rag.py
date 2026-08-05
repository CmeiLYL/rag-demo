from __future__ import annotations

from typing import List

from app.embedding import EmbeddingService
from app.llm import LLMClient
from app.milvus_client import MilvusClient


class RAG:
    """Minimal RAG flow orchestrator."""

    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        self.milvus_client = MilvusClient()
        self.llm_client = LLMClient()

    def build_prompt(self, context_chunks: List[str], question: str) -> str:
        """Build a prompt using retrieved context and user question."""
        context_text = "\n\n".join(context_chunks)
        prompt = (
            "请基于以下上下文回答用户问题。如果上下文与问题不相关，请如实说明。\n\n"
            f"上下文:\n{context_text}\n\n"
            f"问题: {question}\n\n"
            "回答:\n"
        )
        return prompt

    def ask(self, question: str, top_k: int = 3) -> str:
        """Execute the RAG pipeline and return an answer."""
        query_embedding = self.embedding_service.encode([question])[0]
        results = self.milvus_client.search(query_embedding, top_k=top_k)
        context_chunks = [item["text"] for item in results]
        prompt = self.build_prompt(context_chunks, question)
        answer = self.llm_client.chat(prompt)
        return answer
