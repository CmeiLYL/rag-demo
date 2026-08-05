# rag_learn — RAG 学习项目

基于 **FastAPI + Milvus + BGE-M3 + Ollama** 的本地知识库问答（RAG）系统，配套 Django BFF 与 Vue 前端构成完整的人机对话链路。

## 整体架构

```
┌───────────────┐   POST /api/chat/   ┌────────────────┐   POST /api/chat   ┌───────────────────┐
│   Vue 前端     │ ─────────────────► │   Django BFF    │ ─────────────────► │  RAG FastAPI       │
│ (vite:5173)   │ ◄───────────────── │  (django_demo   │ ◄──────────────── │  (rag-demo:8080)   │
│ axios+Element │    {"answer":...}  │   :8000)        │    {"answer":...} │                    │
└───────────────┘                    └────────────────┘                    └────────┬──────────┘
                                                                                    │
                                    ┌───────────────────────────────────────────────┼──────────────────┐
                                    ▼                                               ▼                  ▼
                          BGE-M3 embedding                                  Milvus 向量库     远程 Ollama LLM
                          (sentence-transformers,                         (knowledge_base,    (192.168.123.71:11434,
                           1024 维, CPU 加载)                              HNSW + COSINE)      gemma4:12b)
```

**链路职责**
- **Vue 前端**（`~/vue_demo/vue_demo`）：对话页面，axios 直连 Django，`POST /api/chat/` 携带 `{"message": ...}`，读取响应 `answer` 字段
- **Django BFF**（`~/code_workspace/django_demo`）：`apps/botchat` 的 `chat` 视图，把前端消息转为 `{"question": ...}` 转发 RAG，透传 `answer`；同时提供注册/登录等用户接口
- **RAG 服务**（本仓库 `rag-demo`）：BGE-M3 文本向量化 → Milvus 向量检索 → 拼接上下文 → 调远程 Ollama 生成回答

## 技术栈

| 层次 | 技术 | 说明 |
|---|---|---|
| API 框架 | FastAPI + uvicorn | 单 worker，监听 8080 |
| 向量化 | sentence-transformers + `BAAI/bge-m3` | 输出 1024 维，CPU 加载 |
| 向量库 | Milvus（docker-compose：etcd + MinIO + Milvus standalone） | collection `knowledge_base`，HNSW 索引，COSINE 度量 |
| LLM | 远程 Ollama `gemma4:12b` | `http://192.168.123.71:11434`，响应较慢（80~135s/次） |
| 文档切片 | 自定义 splitter | 默认 chunk_size=500，overlap=100 |
| BFF | Django 5.2 + DRF | 端口 8000，见 django_demo 仓库 |
| 前端 | Vue3 + Vite + Element Plus + Pinia | 端口 5173，见 ~/vue_demo/vue_demo |

## 目录结构

```
rag_learn/
├── rag-demo/                  # RAG 服务（FastAPI）
│   ├── app/
│   │   ├── main.py            # 入口：/api/documents/upload、/api/chat
│   │   ├── config.py          # pydantic-settings 配置（读 .env）
│   │   ├── embedding.py       # BGE-M3 向量化（含 MiniLM fallback）
│   │   ├── milvus_client.py   # Milvus 增删查
│   │   ├── splitter.py        # 文本切片
│   │   ├── document.py        # 文档加载
│   │   ├── rag.py             # RAG 编排：检索 + 拼 prompt + LLM
│   │   └── llm.py             # Ollama 客户端（native + OpenAI 兼容双通道）
│   ├── data/documents/        # 知识库原始文档
│   ├── volumes/               # docker-compose 数据卷（etcd/minio/milvus，不入库）
│   ├── models/                # BGE-M3 本地模型（2.2G，不入库）
│   ├── docker-compose.yml     # Milvus 全家桶
│   ├── Dockerfile / requirements.txt
│   └── README.md
├── .env                       # 运行配置（远程 Ollama 等，不入库）
├── sitecustomize.py           # 自动注入 rag-demo 到 sys.path
├── pyproject.toml / uv.lock   # uv 依赖管理
└── .gitignore
```

## 快速开始

### 1. 启动 Milvus 全家桶

```bash
cd rag-demo
docker compose -f docker-compose.yml up -d
# Milvus 19530 / etcd 2379 / MinIO 9000，健康检查 http://localhost:9091/healthz
```

### 2. 启动 RAG 服务（关键：CPU 加载 embedding）

```bash
cd rag-demo
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  nohup ../.venv/bin/python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080 \
  < /dev/null > /tmp/rag-uvicorn.log 2>&1 &
```

> ⚠️ `CUDA_VISIBLE_DEVICES=''` 必须带：服务器 GPU 仅 4G，BGE-M3（2.2G）加载失败会静默 fallback 到 MiniLM（384 维），与 Milvus 库 1024 维 schema 检索报错。
> `HF_HUB_OFFLINE=1`：服务器无法访问 HuggingFace，离线加载本地缓存模型。

### 3. 上传知识库文档

```bash
curl -X POST "http://localhost:8080/api/documents/upload" -F "file=@data/documents/xxx.txt"
# {"message": "success"}
```

### 4. 问答

```bash
curl -X POST "http://localhost:8080/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是Milvus?"}'
# {"answer": "..."}
```

## 接口文档

| 方法 | 路径 | 请求 | 响应 | 说明 |
|---|---|---|---|---|
| POST | `/api/chat` | `{"question": str}` | `{"answer": str}` | RAG 问答（Django 也发 question） |
| POST | `/api/documents/upload` | `multipart: file`（txt/md） | `{"message": "success"}` | 上传并切块入库 |
| GET | `/` | - | 静态页面 | 内置演示页 |
| GET | `/api/health/` | - | `{"app":"botchat","status":"ok"}` | Django 侧健康检查 |

## 配置说明（.env）

| 变量 | 默认 | 说明 |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | 远程 LLM 地址，当前 `http://192.168.123.71:11434` |
| `OLLAMA_MODEL` | `qwen2.5` | 当前 `gemma4:12b` |
| `OLLAMA_CONNECT_TIMEOUT_SECONDS` | 10 | 连接超时 |
| `OLLAMA_READ_TIMEOUT_SECONDS` | 180 | 读超时，当前 300（LLM 慢） |
| `MILVUS_HOST/PORT` | localhost:19530 | Milvus 地址 |
| `MILVUS_COLLECTION_NAME` | knowledge_base | 向量集合名 |
| `BGE_MODEL_NAME` | BAAI/bge-m3 | embedding 模型 |
| `CHUNK_SIZE / CHUNK_OVERLAP` | 500 / 100 | 切片参数 |
| `PYTHONPATH` | rag-demo 绝对路径 | sitecustomize 注入 sys.path |

## 踩坑记录

1. **embedding 维度不匹配**：GPU 显存不足 → BGE-M3 加载失败 → fallback MiniLM（384维）→ Milvus 报 `vector dimension mismatch, expected 4096, actual 1536`。解决：CPU 模式加载（见上）。
2. **uvicorn "假死"**：远程 LLM 单次 80-135s，单 worker 串行导致请求排队、连接积压（`ss` 看 Recv-Q）。不是服务挂，是慢。
3. **前端契约**：axios timeout 需 ≥ 180s；响应字段是 `answer`（前端曾读 `reply` 显示 undefined）。
4. **HF 离线**：服务器访问不了 huggingface.co，模型已缓存在 `~/.cache/huggingface/hub/` 和 `rag-demo/models/`，启动必须 `HF_HUB_OFFLINE=1`。
5. **git 大文件**：`models/`、`volumes/`、`*-data/` 均已加入 .gitignore，勿提交。
