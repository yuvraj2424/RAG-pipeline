from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_core.documents import Document

from api.dependencies import get_indexer
from api.schemas.requests import IndexRequest
from api.schemas.responses import IndexResponse
from src.config import SOURCE_DIR
from src.datasources import load_from_source
from src.graph.builder import build_rag_graph
from src.indexing.embeddings import load_existing_store
from src.indexing.loaders import load_documents, load_from_directory

router = APIRouter(prefix="/index", tags=["Indexing"])


@router.post("", response_model=IndexResponse, summary="Index documents into the vector store")
async def index_documents(
    body: IndexRequest,
    request: Request,
    indexer=Depends(get_indexer),
):
    raw_docs: list[Document] = []
    sources_count = 0

    # ── Collect all raw docs first (one indexing pass at the end) ─────────────
    if body.sources:
        raw_docs.extend(load_documents([s.model_dump() for s in body.sources]))
        sources_count += len(body.sources)

    if body.sql:
        try:
            raw_docs.extend(load_from_source("sql_server", **body.sql.model_dump()))
            sources_count += 1
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"SQL indexing failed: {exc}") from exc

    if body.web:
        raw_docs.extend(load_from_source("web", **body.web.model_dump()))
        sources_count += 1

    # ── Default: scan source/ directory ───────────────────────────────────────
    if not raw_docs:
        raw_docs = load_from_directory(SOURCE_DIR)
        sources_count = 1

    # ── Single indexing pass for all collected docs ────────────────────────────
    _, chunks = indexer.run_from_docs(raw_docs)

    # ── Rebuild graph with the updated store ──────────────────────────────────
    result = load_existing_store()
    if result:
        vectorstore, store_docs = result
        request.app.state.rag_graph = build_rag_graph(vectorstore, store_docs)

    return IndexResponse(
        chunks_indexed=len(chunks),
        sources_processed=sources_count,
        message=f"Indexed {len(chunks)} chunks. Pipeline ready.",
    )
