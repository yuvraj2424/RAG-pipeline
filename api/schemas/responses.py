from pydantic import BaseModel


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    citations: list[str]
    caveats: str | None


class IndexResponse(BaseModel):
    chunks_indexed: int
    sources_processed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    vector_store_exists: bool
