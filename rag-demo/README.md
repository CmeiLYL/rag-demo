# RAG Demo

一个最小可运行的本地知识库问答系统，基于 FastAPI + BGE-M3 + Milvus + Ollama。

## 目录结构

```
rag-demo/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── embedding.py
│   ├── milvus_client.py
│   ├── splitter.py
│   ├── document.py
│   ├── rag.py
│   └── llm.py
├── data/
│   └── documents/
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## 1. 环境准备

- Python 3.11+
- Docker + Docker Compose
- 本地 Milvus、etcd、MinIO
- 本地 Ollama 服务

## 2. 安装依赖

```bash
cd rag-demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. 启动 Milvus

```bash
docker compose -f docker-compose.yml up -d
```

Milvus 服务默认监听 `19530`，etcd `2379`，MinIO `9000`。

## 4. 启动 Ollama

请确保已经安装并启动 Ollama，本项目默认调用：

- URL: `http://localhost:11434`
- 模型: `qwen2.5`

```bash
ollama serve
```

## 5. 下载 BGE-M3

`sentence-transformers` 会在首次运行时自动下载模型。若希望提前准备模型缓存，可执行：

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

## 6. 启动 FastAPI

```bash
cd rag-demo
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 7. 测试上传接口

```bash
curl -X POST "http://127.0.0.1:8000/api/documents/upload" \
  -F "file=@path/to/your/document.txt"
```

返回示例：

```json
{ "message": "success" }
```

## 8. 测试问答接口

```bash
curl -X POST "http://127.0.0.1:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"question":"什么是Milvus?"}'
```

返回示例：

```json
{ "answer": "..." }
```

## 9. 说明

- 文档支持 `.txt` 和 `.md` 文件
- 文本切片默认 `chunk_size=500`，`chunk_overlap=100`
- Embedding 默认使用 `BAAI/bge-m3`，若模型无法下载则自动回退到 `sentence-transformers/all-MiniLM-L6-v2`
- 向量库使用 Milvus `knowledge_base` 集合，索引类型 `HNSW`，度量方式 `COSINE`
- LLM 调用 Ollama `qwen2.5`
