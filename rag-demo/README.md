# RAG Demo

一个最小可运行的本地知识库问答系统，基于 **FastAPI + BGE-M3 + Milvus + Ollama**，对外提供文档入库与 RAG 问答接口。完整链路（Vue 前端 → Django BFF → 本服务）见仓库根 `README.md`。

## 目录结构

```
rag-demo/
├── app/
│   ├── main.py              # FastAPI 入口：/api/documents/upload、/api/chat、静态页
│   ├── config.py            # pydantic-settings 配置（读取 rag_learn/.env）
│   ├── embedding.py         # BGE-M3 向量化（GPU 失败自动 fallback MiniLM）
│   ├── milvus_client.py     # Milvus 连接 / 建集合 / 插入 / 检索
│   ├── splitter.py          # 文本切片（chunk_size=500, overlap=100）
│   ├── document.py          # 文档加载
│   ├── rag.py               # RAG 编排：向量化 → 检索 → 拼 prompt → LLM
│   └── llm.py               # Ollama 客户端（native + OpenAI 兼容双通道）
├── data/documents/          # 知识库原始文档（txt/md）
├── volumes/                 # docker-compose 数据卷：etcd / minio / milvus（不入库）
├── models/                  # BGE-M3 本地模型缓存（2.2G，不入库）
├── docker-compose.yml       # Milvus standalone + etcd + MinIO
├── Dockerfile
├── requirements.txt
└── README.md
```

> venv 位于 `rag_learn/.venv`（仓库根，非本目录），启动用 `../.venv/bin/python3`。

## 1. 环境准备

- Python 3.11+（本机使用 uv 管理的 cpython-3.11.15）
- Docker + Docker Compose（Milvus、etcd、MinIO）
- Ollama 服务（本项目配置为远程 `192.168.123.71:11434`）

## 2. 安装依赖

```bash
cd rag-demo
python -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt
```

## 3. 启动 Milvus

```bash
docker compose -f docker-compose.yml up -d
```

Milvus `19530`（gRPC）/ `9091`（健康检查），etcd `2379`，MinIO `9000`。数据落在 `volumes/` 下。

```bash
curl http://localhost:9091/healthz   # OK
```

## 4. 配置远程 Ollama

运行配置在 `rag_learn/.env`（不入库），当前值：

```ini
OLLAMA_URL=http://192.168.123.71:11434
OLLAMA_MODEL=gemma4:12b
OLLAMA_CONNECT_TIMEOUT_SECONDS=10
OLLAMA_READ_TIMEOUT_SECONDS=300
PYTHONPATH=/home/bosiju/code_workspace/rag_learn/rag-demo
```

## 5. BGE-M3 模型

已离线缓存在 `rag-demo/models/BAAI_bge-m3`（2.2G）与 `~/.cache/huggingface/hub/`。服务器无法访问 HuggingFace，加载必须离线模式（见下）。

## 6. 启动 FastAPI（8080）

```bash
cd rag-demo
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  nohup ../.venv/bin/python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080 \
  < /dev/null > /tmp/rag-uvicorn.log 2>&1 &
```

> ⚠️ `CUDA_VISIBLE_DEVICES=''`：GPU 仅 4G，BGE-M3（2.2G）在 CUDA 上加载失败会 fallback 到 MiniLM（384 维），与 Milvus 库 1024 维不匹配导致检索报错。
> `HF_HUB_OFFLINE=1`：避免加载模型时联网挂起（HF 不可达）。

## 7. 上传文档入库

```bash
curl -X POST "http://127.0.0.1:8080/api/documents/upload" \
  -F "file=@path/to/your/document.txt"
# {"message": "success"}
```

## 8. 问答接口

```bash
curl -X POST "http://127.0.0.1:8080/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"question":"什么是Milvus?"}'
# {"answer": "..."}
```

## 9. 说明与参数

- 文档支持 `.txt` / `.md` / `.markdown`
- 切片默认 `chunk_size=500`，`chunk_overlap=100`（`.env` 可调）
- Embedding：`BAAI/bge-m3`，1024 维，归一化后入库；下载失败自动回退 `all-MiniLM-L6-v2`（384 维）
- 向量库：Milvus 集合 `knowledge_base`，HNSW 索引，COSINE 度量
- LLM：Ollama `gemma4:12b`（远程），响应 80~135s，前端超时需 ≥ 180s
- 单 worker 串行：慢请求会排队，属正常现象
