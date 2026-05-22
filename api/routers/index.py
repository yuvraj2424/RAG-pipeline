from fastapi import APIRouter, Depends, Request

from api.dependencies import get_indexer
from api.schemas.responses import IndexResponse
from src.config import SOURCE_DIR
from src.graph.builder import build_rag_graph
from src.indexing.embeddings import load_existing_store

router = APIRouter(prefix="/index", tags=["Indexing"])


@router.post("", response_model=IndexResponse, summary="Index documents from the source/ directory")
async def index_documents(request: Request, indexer=Depends(get_indexer)):
    """Scan source/ recursively (PDF, DOCX, XLSX) and rebuild the RAG pipeline."""
    _, chunks = indexer.run_from_directory(SOURCE_DIR)

    result = load_existing_store()
    if result:
        vectorstore, store_docs = result
        request.app.state.rag_graph = build_rag_graph(vectorstore, store_docs)

    return IndexResponse(
        chunks_indexed=len(chunks),
        sources_processed=1,
        message=f"Indexed {len(chunks)} chunks from source/. Pipeline ready.",
    )
