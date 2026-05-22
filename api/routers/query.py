from fastapi import APIRouter, Depends

from api.dependencies import require_pipeline
from api.schemas.requests import QueryRequest
from api.schemas.responses import QueryResponse
from src.datasources.mcp_source import fetch_mcp_documents

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse, summary="Run the RAG pipeline on a query")
async def query_rag(body: QueryRequest, rag_graph=Depends(require_pipeline)):
    # Fetch MCP docs async before invoking the (sync) graph.
    # retrieve_node will merge these with vector store results.
    mcp_docs = await fetch_mcp_documents(body.query)

    result = rag_graph.invoke({
        "query": body.query,
        "chat_history": body.chat_history,
        "documents": mcp_docs,          # seeded — merged with vector docs in retrieve_node
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
