# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & Running

Install dependencies (use `.venv/bin/pip` — the project uses a local virtualenv):

```bash
pip install -r requirements.txt
```

Requires either `ANTHROPIC_FOUNDRY_API_KEY` (Azure AI Foundry / Claude) or `OPENAI_API_KEY` (fallback). See `.env.example` for all options.

**Start the API server:**
```bash
.venv/bin/python app.py          # runs uvicorn on http://0.0.0.0:8000
# Interactive docs at http://localhost:8000/docs
```

**Run the standalone MCP agent:**
```bash
python -m src.mcp.server                        # default query
python -m src.mcp.server "your question here"   # custom query
```

## Project Structure

```
src/
├── config.py              # All env vars and hardcoded constants (models, paths, weights)
├── llm.py                 # get_llm(tier?) + get_embeddings() factories; module-level instances
├── indexing/
│   ├── loaders.py         # PDF/DOCX/XLSX document loaders + directory scanner
│   ├── chunkers.py        # SemanticChunker with RecursiveCharacterTextSplitter fallback
│   └── embeddings.py      # Chroma (dense) + BM25 (sparse) index creation and persistence
├── retrieval/
│   ├── retrievers.py      # Hybrid ensemble (BM25 + MMR) + multi-query expansion
│   └── rerankers.py       # FlashrankRerank + content-hash deduplication
├── augmentation/
│   ├── graders.py         # CRAG per-chunk LLM relevance grading
│   └── context.py         # Lost-in-the-middle reordering + citation context builder
├── generation/
│   ├── schemas.py         # RAGAnswer Pydantic model + RAG_SYSTEM prompt constant
│   └── chains.py          # generate() with chat history + check_faithfulness()
├── datasources/
│   ├── base.py            # BaseDataSource ABC
│   ├── filesystem.py      # File-based source (wraps loaders.py)
│   ├── sql_server.py      # SQLAlchemy → Documents
│   ├── web_crawler.py     # WebBaseLoader multi-URL crawler
│   └── mcp_source.py      # async fetch_mcp_documents(query) — query-time MCP retrieval
├── tools/
│   ├── web_search.py      # Tavily or DuckDuckGo fallback → Documents
│   └── sql_query.py       # LangChain SQL tools (requires SQL_CONNECTION_STRING)
├── mcp/
│   └── server.py          # Standalone MCP client: agentic loop against Dubai Holding Foundry
└── graph/
    ├── state.py           # RAGState TypedDict
    ├── nodes.py           # LangGraph node factory functions
    ├── edges.py           # Conditional routing (route_after_augment, route_after_faithfulness)
    └── builder.py         # build_rag_graph() → CompiledGraph

api/
├── main.py                # FastAPI app + lifespan (auto-loads existing vector store on restart)
├── dependencies.py        # get_rag_graph(), get_indexer(), require_pipeline()
├── routers/
│   ├── health.py          # GET  /api/v1/health
│   ├── query.py           # POST /api/v1/query  (fetches MCP docs async before graph invoke)
│   └── index.py           # POST /api/v1/index  (scans source/ only, no request body)
└── schemas/
    ├── requests.py        # QueryRequest (query + chat_history)
    └── responses.py       # QueryResponse, IndexResponse, HealthResponse

source/                    # Raw input documents (PDFs, DOCX, XLSX) — only indexing source
data/rag_db/               # Persisted Chroma vector store (auto-generated)
app.py                     # Entry point: uvicorn launcher
```

## Architecture

**Self-correcting RAG pipeline** built on LangChain + LangGraph with dual retrieval (vector store + MCP).

### Pipeline Stages

1. **`IndexingPipeline`** (offline) — Scans `source/` recursively → cleans → chunks via `SemanticChunker` (90th percentile breakpoint) with `RecursiveCharacterTextSplitter` fallback (512 chars, 50 overlap) → embeds into Chroma (dense) + BM25 (sparse), persisted to `data/rag_db/`.

2. **`RetrievalPipeline`** (per query) — LLM expands query into 3 variants → hybrid retrieval (35% BM25 + 65% MMR, k=10 each) → FlashrankRerank (top 5) → deduplication → top 6 chunks. The `retrieve_node` **merges** pre-seeded MCP docs from state with vector results before returning.

3. **`AugmentationPipeline`** (per query) — LLM grades each chunk (`"relevant"` vs `"irrelevant"` — substring check guards against `"irrelevant"` matching) → reorders best-first/best-last → builds numbered citation context.

4. **`GenerationPipeline`** (per query) — Prepends `chat_history` as `HumanMessage`/`AIMessage` before the RAG context → generates `RAGAnswer` (structured output) → faithfulness check.

### LangGraph Orchestration (CRAG loop)

`RAGState` fields: `query`, `chat_history`, `documents`, `context`, `answer`, `faithful`, `web_search_retries`, `faith_retries`

Two **separate** retry budgets (both default `MAX_RETRIES=2`):
- `web_search_retries` — checked by `route_after_augment`: no context → `web_search` node
- `faith_retries` — checked by `route_after_faithfulness`: unfaithful → back to `generate`

### Query Flow (Dual Retrieval)

```
POST /api/v1/query
  1. Router (async): fetch_mcp_documents(query)  → mcp_docs
  2. graph.invoke(documents=mcp_docs, ...)
       retrieve_node:  vector_docs = retrieval.retrieve(query)
                       documents   = mcp_docs + vector_docs
       augment_node:   grade all docs, build context
       generate_node:  answer with chat_history + context
       faithfulness:   validate, retry if needed
```

MCP fetch fails gracefully (returns `[]`) when `FOUNDRY_TOKEN` is unset or connection fails.

### FastAPI Layer

`api/main.py` lifespan auto-loads `data/rag_db/` on startup and reconstructs BM25 docs from `vectorstore.get()`.

| Endpoint | Description |
|---|---|
| `GET /api/v1/health` | `pipeline_ready` + `vector_store_exists` |
| `POST /api/v1/index` | No body — scans `source/` recursively, rebuilds graph |
| `POST /api/v1/query` | `{"query": "...", "chat_history": [...]}` → `QueryResponse` |

`chat_history` format: `[{"role": "user"/"assistant", "content": "..."}]`

### LLM & Embedding Selection

**LLM** (`src/llm.py`) — `MODEL_TIER=sonnet|haiku|opus`:
| Priority | Condition | Provider |
|---|---|---|
| 1 | `ANTHROPIC_FOUNDRY_API_KEY` set | Azure AI Foundry → `ChatAnthropic` |
| 2 | fallback | OpenAI → `ChatOpenAI` (`gpt-4o`) |

**Embeddings** — Claude has no embedding model:
| Priority | Condition | Provider |
|---|---|---|
| 1 | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` set | `AzureOpenAIEmbeddings` |
| 2 | fallback | `OpenAIEmbeddings` (`text-embedding-3-large`) |

Azure Foundry base URL auto-derived: `https://{ANTHROPIC_FOUNDRY_RESOURCE}.services.ai.azure.com/anthropic`

### MCP Integration

Two separate MCP roles:

**Query-time retrieval** (`src/datasources/mcp_source.py`):
- `async fetch_mcp_documents(query)` called from `api/routers/query.py` before graph invocation
- Connects to Foundry, calls each tool with `{"query": query}`, returns results as Documents
- Docs seeded into initial `RAGState.documents` and merged with vector results in `retrieve_node`

**Standalone agent** (`src/mcp/server.py`):
- Full agentic loop: sends query → handles `tool_use` → calls `session.call_tool()` → feeds results back → repeats until `end_turn`
- `ANTHROPIC_API_KEY` falls back to `ANTHROPIC_FOUNDRY_API_KEY`; routes through Azure when `ANTHROPIC_FOUNDRY_BASE_URL` is set

### Key Constants (all in `src/config.py`)

| Constant | Value |
|---|---|
| Hybrid weights | `[0.35, 0.65]` (BM25 / vector) |
| Chunk size / overlap | `512` / `50` |
| Semantic breakpoint | 90th percentile |
| Reranker top-N | `5` |
| Final top-K | `6` |
| Max retries (each budget) | `2` |

### IDE Import Warnings

Pylance shows "Import could not be resolved" for `langchain_*`, `fastapi`, `mcp`. False positives — packages are in `.venv`. Set Pylance interpreter to `.venv/bin/python`.

### Package Import Notes

| Class | Correct import |
|---|---|
| `EnsembleRetriever` | `langchain_classic.retrievers.ensemble` |
| `ContextualCompressionRetriever` | `langchain_classic.retrievers.contextual_compression` |
| `FlashrankRerank` | `langchain_community.document_compressors.flashrank_rerank` |
| `RecursiveCharacterTextSplitter` | `langchain_text_splitters` |
| `BM25Retriever` | `langchain_community.retrievers.bm25` |
