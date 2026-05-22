from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config import VECTOR_STORE_DIR
from src.indexing import IndexingPipeline
from src.indexing.embeddings import load_existing_store
from src.graph.builder import build_rag_graph
from api.routers import health, index, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialize the indexer and, if a persisted vector store already
    exists, rebuild the RAG graph from it so the service is immediately ready.
    Shutdown: nothing to clean up (Chroma persists to disk automatically).
    """
    app.state.rag_graph = None
    app.state.indexer = IndexingPipeline(persist_dir=str(VECTOR_STORE_DIR))

    result = load_existing_store()
    if result is not None:
        vectorstore, all_docs = result
        app.state.rag_graph = build_rag_graph(vectorstore, all_docs)
        print(f"[Startup] Loaded {len(all_docs)} chunks from existing vector store.")
    else:
        print("[Startup] No existing vector store found. POST /api/v1/index to build one.")

    yield


app = FastAPI(
    title="RAG Pipeline API",
    description="Self-correcting RAG pipeline powered by LangChain + LangGraph.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")
app.include_router(index.router, prefix="/api/v1")
