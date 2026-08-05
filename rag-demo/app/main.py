from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.document import DocumentLoader
from app.embedding import EmbeddingService
from app.milvus_client import MilvusClient
from app.rag import RAG

app = FastAPI(title="RAG Demo")
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")

milvus_client: Optional[MilvusClient] = None
embedding_service: Optional[EmbeddingService] = None
rag_service: Optional[RAG] = None


def get_milvus_client() -> MilvusClient:
    global milvus_client
    if milvus_client is None:
        milvus_client = MilvusClient()
    return milvus_client


def get_embedding_service() -> EmbeddingService:
    global embedding_service
    if embedding_service is None:
        embedding_service = EmbeddingService()
    return embedding_service


def get_rag_service() -> RAG:
    global rag_service
    if rag_service is None:
        rag_service = RAG()
    return rag_service

DOCUMENTS_DIR = Path(__file__).resolve().parents[1] / "data" / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a TXT or MD document and index its text chunks."""
    suffix = file.filename.lower().split(".")[-1]
    if suffix not in {"txt", "md", "markdown"}:
        raise HTTPException(status_code=400, detail="仅支持 txt/md/markdown 文件")

    content = await file.read()
    text = content.decode("utf-8")
    if not text.strip():
        raise HTTPException(status_code=400, detail="文件内容为空或无有效文本")

    file_path = DOCUMENTS_DIR / file.filename
    file_path.write_text(text, encoding="utf-8")

    loader = DocumentLoader(DOCUMENTS_DIR)
    chunks = loader.split_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="文件内容为空或无有效文本")

    embeddings = get_embedding_service().encode(chunks)
    get_milvus_client().insert_documents(chunks, embeddings)
    return JSONResponse(content={"message": "success"})


@app.post("/api/chat")
async def chat(request: dict[str, str]) -> JSONResponse:
    """Ask a question and return an answer based on vector retrieval."""
    question = request.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    try:
        answer = get_rag_service().ask(question)
    except RuntimeError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    return JSONResponse(content={"answer": answer})
