from __future__ import annotations

from typing import List


class TextSplitter:
    """Split incoming documents into overlapping chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Split text into fixed-size chunks with overlap."""
        cleaned = text.replace("\n", " ").strip()
        if not cleaned:
            return []

        chunks: List[str] = []
        start = 0
        while start < len(cleaned):
            end = min(start + self.chunk_size, len(cleaned))
            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(cleaned):
                break
            start += self.chunk_size - self.chunk_overlap
        return chunks
