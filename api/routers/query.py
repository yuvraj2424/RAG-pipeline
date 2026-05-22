from fastapi import APIRouter, Depends

from api.dependencies import require_pipeline
from api.schemas.requests import QueryRequest
from api.schemas.responses import QueryResponse

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse, summary="Run the RAG pipeline on a query")
async def query_rag(body: QueryRequest, rag_graph=Depends(require_pipeline)):
    result = rag_graph.invoke({
        "query": body.query,
        "chat_history": body.chat_history,
        "documents": [],
        "context": "",
        "answer": None,
        "faithful": False,
        "web_search_retries": 0,
        "faith_retries": 0,
    })

    ans = result["answer"]
    return QueryResponse(
        answer=ans.answer,
        confidence=ans.confidence,
        citations=ans.citations,
        caveats=ans.caveats,
    )
