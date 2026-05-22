# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Requires either `ANTHROPIC_FOUNDRY_API_KEY` (Azure AI Foundry / Claude) or `OPENAI_API_KEY` (fallback). See `.env.example` for all options.

**Start the API server:**
```bash
python app.py               # runs uvicorn on http://0.0.0.0:8000
# or equivalently:
uvicorn api.main:app --reload
# Interactive docs at http://localhost:8000/docs
```

**Run the MCP agent (connects to Dubai Holding Foundry endpoint):**
```bash
python -m src.mcp.server                        # default query
python -m src.mcp.server "your question here"   # custom query
```

## Project Structure

```
src/
├── config.py              # All hardcoded values (models, paths, weights)
├── llm.py                 # get_llm(tier?) + get_embeddings() factories; module-level instances
├── indexing/
│   ├── loaders.py         # PDF/DOCX/XLSX/web/text document loaders
│   ├── chunkers.py        # Semantic + recursive chunking strategies
│   └── embeddings.py      # Dual-index creation (Chroma + BM25), persistence
├── retrieval/
│   ├── retrievers.py      # Dense, sparse, hybrid ensemble + query expansion
│   └── rerankers.py       # FlashrankRerank + content-hash deduplication
├── augmentation/
│   ├── graders.py         # CRAG relevance grading per chunk
│   └── context.py         # Chunk positioning + citation context assembly
├── generation/
│   ├── schemas.py         # RAGAnswer Pydantic schema
│   └── chains.py          # GPT-4o generation + faithfulness validation chains
├── datasources/
│   ├── base.py            # BaseDataSource ABC
│   ├── filesystem.py      # Wraps loaders.py for file-based sources
│   ├── sql_server.py      # SQLAlchemy → Documents (SQL Server, Postgres, SQLite)
│   └── web_crawler.py     # WebBaseLoader multi-URL crawler
├── tools/
│   ├── web_search.py      # Tavily (if key set) or DuckDuckGo fallback → Documents
│   └── sql_query.py       # QuerySQLDataBaseTool / InfoSQLDatabaseTool / ListSQLDatabaseTool
├── mcp/
│   └── server.py          # MCP client: connects to Dubai Holding Foundry endpoint, lists tools, runs Claude agent
└── graph/
    ├── state.py           # RAGState TypedDict
    ├── nodes.py           # LangGraph node functions (web_search_node uses real search)
    ├── edges.py           # Conditional routing logic
    └── builder.py         # build_rag_graph() → CompiledGraph

api/
├── main.py                # FastAPI app + lifespan (auto-loads existing vector store on restart)
├── dependencies.py        # get_rag_graph(), get_indexer(), require_pipeline()
├── routers/
│   ├── health.py          # GET  /api/v1/health
│   ├── query.py           # POST /api/v1/query
│   └── index.py           # POST /api/v1/index
└── schemas/
    ├── requests.py        # QueryRequest, IndexRequest, DocumentSource, SQLSource, WebSource
    └── responses.py       # QueryResponse, IndexResponse, HealthResponse

source/                    # Raw input documents (PDFs, DOCX, XLSX)
data/rag_db/               # Persisted Chroma vector store (generated)
tests/                     # Unit tests per pipeline stage
notebooks/                 # Experimentation notebooks
app.py                     # Entry point: starts FastAPI server via uvicorn
```

## Architecture

**Self-correcting RAG pipeline** built on LangChain + LangGraph, organized into four pipeline stages followed by a LangGraph orchestration layer.

### Pipeline Stages

1. **`IndexingPipeline`** (offline, run once) — Loads documents (PDF/web/text) → cleans → chunks using `SemanticChunker` (90th percentile breakpoint) with fallback to `RecursiveCharacterTextSplitter` (512 chars, 50 overlap) → embeds into dual indices: Chroma (dense) + BM25 (sparse), persisted to `./rag_db`.

2. **`RetrievalPipeline`** (per query) — LLM expands query into 3 variants → hybrid retrieval (35% BM25 + 65% vector MMR, k=10 each) → FlashrankRerank (top 5) → content-hash deduplication → returns top 6 chunks.

3. **`AugmentationPipeline`** (per query) — CRAG-style LLM relevance grading drops irrelevant chunks → reorders chunks best-first/best-last to counter "lost-in-the-middle" degradation → builds numbered citation context string.

4. **`GenerationPipeline`** (per query) — LLM generates a `RAGAnswer` Pydantic struct (`answer`, `confidence`, `citations`, `caveats`) using structured output → post-hoc faithfulness check validates all claims are grounded in context.

### LangGraph Orchestration (CRAG loop)

`build_rag_graph()` wires these stages into a stateful graph (`RAGState` TypedDict: `query, documents, context, answer, faithful, retry_count`) with conditional routing:

- After augment: if no context and `retry_count < 2` → `web_search` node (Tavily or DuckDuckGo)
- After faithfulness check: if not faithful and `retry_count < 2` → back to `generate`; otherwise → END

### FastAPI Layer

`api/main.py` uses a **lifespan** context manager to initialize app state once on startup. Pipeline instances are stored on `app.state` and injected into route handlers via FastAPI dependencies (`api/dependencies.py`).

| Endpoint | Description |
|---|---|
| `GET /api/v1/health` | Reports `pipeline_ready` and `vector_store_exists` |
| `POST /api/v1/index` | Indexes documents and rebuilds the RAG graph. Body fields: `sources` (file list), `sql` (SQL source), `web` (URL list) — all optional; omitting all triggers full `source/` scan |
| `POST /api/v1/query` | Invokes the compiled LangGraph and returns a `QueryResponse` |

On restart, the lifespan auto-loads an existing `data/rag_db` Chroma store and reconstructs `Document` objects for BM25 from `vectorstore.get()`, so the service is immediately ready without re-indexing.

### LLM & Embedding Selection

`src/llm.py` exposes `get_llm(tier?)` and `get_embeddings()` factories. Priority order:

**LLM** — set `MODEL_TIER=sonnet|haiku|opus` in `.env`:
| Priority | Condition | Provider |
|---|---|---|
| 1 | `ANTHROPIC_FOUNDRY_API_KEY` set | Azure AI Foundry → `ChatAnthropic` |
| 2 | fallback | OpenAI → `ChatOpenAI` (`gpt-4o`) |

**Embeddings** — Claude has no embedding model, so:
| Priority | Condition | Provider |
|---|---|---|
| 1 | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` set | `AzureOpenAIEmbeddings` |
| 2 | fallback | `OpenAIEmbeddings` (`text-embedding-3-large`) |

Azure AI Foundry base URL is auto-derived: `https://{ANTHROPIC_FOUNDRY_RESOURCE}.services.ai.azure.com/anthropic`

### Key Hardcoded Values

- Hybrid retrieval weights: `[0.35, 0.65]` (BM25/vector) — tunable per domain
- Vector DB persist directory: `data/rag_db/`
- Web search: uses Tavily if `TAVILY_API_KEY` is set, otherwise DuckDuckGo (free, no key needed)
- SQL: any SQLAlchemy-compatible DB via `SQL_CONNECTION_STRING` env var

### Data Sources & Tools

`src/datasources/` provides pluggable connectors with a shared registry:
```python
from src.datasources import load_from_source
docs = load_from_source("sql_server", connection_string="...", table="products")
docs = load_from_source("filesystem", directory="./source")
docs = load_from_source("web", urls=["https://..."])
```
New sources are added by subclassing `BaseDataSource` and registering in `REGISTRY`.

`src/tools/` provides LangChain tools for the graph's web search fallback and SQL querying. `get_sql_tools()` returns empty list if `SQL_CONNECTION_STRING` is unset.

### MCP Agent

`src/mcp/server.py` is an **MCP client** (not a server) that connects to the Dubai Holding Foundry remote MCP endpoint via `streamablehttp_client`. It:
1. Authenticates with `FOUNDRY_TOKEN` Bearer header
2. Fetches available tools from the remote endpoint via `session.list_tools()`
3. Passes them to `AsyncAnthropic.messages.create()` so Claude can invoke them

`ANTHROPIC_API_KEY` auto-falls back to `ANTHROPIC_FOUNDRY_API_KEY` if not separately set. When `ANTHROPIC_FOUNDRY_BASE_URL` is configured, the `AsyncAnthropic` client routes through Azure AI Foundry instead of `api.anthropic.com`.

Required env vars: `FOUNDRY_TOKEN`, `ANTHROPIC_FOUNDRY_API_KEY` (or `ANTHROPIC_API_KEY`). `MCP_SERVER_URL` defaults to the Dubai Holding Foundry endpoint and can be overridden.

### IDE Import Warnings

Pylance will show "Import could not be resolved" for `langchain_*` and `fastapi`. This is a false positive — packages are installed in `.venv`. Point Pylance to the interpreter at `.venv/bin/python` to suppress. All imports are verified to work at runtime.

### Package Import Notes

In the installed version of LangChain, several classes have moved from their original locations:

| Class | Correct import |
|---|---|
| `EnsembleRetriever` | `langchain_classic.retrievers.ensemble` |
| `ContextualCompressionRetriever` | `langchain_classic.retrievers.contextual_compression` |
| `FlashrankRerank` | `langchain_community.document_compressors.flashrank_rerank` |
| `RecursiveCharacterTextSplitter` | `langchain_text_splitters` |
| `BM25Retriever` | `langchain_community.retrievers.bm25` |
