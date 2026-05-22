from fastapi import Depends, HTTPException, Request


def get_rag_graph(request: Request):
    """Return compiled LangGraph instance from app state."""
    return request.app.state.rag_graph


def get_indexer(request: Request):
    """Return IndexingPipeline instance from app state."""
    return request.app.state.indexer


def require_pipeline(rag_graph=Depends(get_rag_graph)):
    """Dependency that raises 503 if the pipeline is not yet initialized."""
    if rag_graph is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not ready. POST /api/v1/index to build the index first.",
        )
    return rag_graph
