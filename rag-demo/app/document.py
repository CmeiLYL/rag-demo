from __future__ import annotations

from pathlib import Path
from typing import List

from app.config import settings
from app.splitter import TextSplitter


class DocumentLoader:
    """Load documents from local files and split them into chunks."""

    def __init__(self, loader_path: str | Path) -> None:
        self.loader_path = Path(loader_path)
        self.splitter = TextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def load_text(self, file_path: Path) -> str:
        """Read text content from txt or markdown files."""
        suffix = file_path.suffix.lower()
        if suffix not in {".txt", ".md", ".markdown"}:
            raise ValueError("Unsupported file type: %s" % suffix)

        return file_path.read_text(encoding="utf-8")

    def discover_files(self) -> List[Path]:
        """List supported files under the document directory."""
        if not self.loader_path.exists():
            return []

        return [
            path
            for path in self.loader_path.rglob("*.*")
            if path.suffix.lower() in {".txt", ".md", ".markdown"}
        ]

    def load_documents(self) -> List[str]:
        """Load and split all documents found in the folder."""
        chunks: List[str] = []
        for path in self.discover_files():
            text = self.load_text(path)
            chunks.extend(self.splitter.split_text(text))
        return chunks

    def split_text(self, text: str) -> List[str]:
        """Split arbitrary text using the configured splitter."""
        return self.splitter.split_text(text)
