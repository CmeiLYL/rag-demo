from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(
            Path(__file__).resolve().parents[1] / ".env",
            Path(__file__).resolve().parents[2] / ".env",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    milvus_host: str = Field("localhost", env="MILVUS_HOST")
    milvus_port: int = Field(19530, env="MILVUS_PORT")
    milvus_collection_name: str = Field("knowledge_base", env="MILVUS_COLLECTION_NAME")
    bge_model_name: str = Field("BAAI/bge-m3", env="BGE_MODEL_NAME")
    ollama_url: str = Field("http://localhost:11434", env="OLLAMA_URL")
    ollama_model: str = Field("qwen2.5", env="OLLAMA_MODEL")
    ollama_connect_timeout_seconds: int = Field(10, env="OLLAMA_CONNECT_TIMEOUT_SECONDS")
    ollama_read_timeout_seconds: int = Field(180, env="OLLAMA_READ_TIMEOUT_SECONDS")
    chunk_size: int = Field(500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(100, env="CHUNK_OVERLAP")


settings = Settings()
