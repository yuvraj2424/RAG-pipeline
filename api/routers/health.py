from fastapi import APIRouter, Request

from api.schemas.responses import HealthResponse
from src.config import VECTOR_STORE_DIR

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse, summary="Service health check")
async def health(request: Request):
    vector_store_exists = VECTOR_STORE_DIR.exists() and any(VECTOR_STORE_DIR.iterdir())
    pipeline_ready = request.app.state.rag_graph is not None
    return HealthResponse(
        status="ok",
        pipeline_ready=pipeline_ready,
        vector_store_exists=vector_store_exists,
    )
