import os

from langchain_core.documents import Document
from langchain_core.tools import BaseTool


def get_web_search_tool() -> BaseTool:
    """
    Return Tavily search if TAVILY_API_KEY is set, otherwise DuckDuckGo (free, no key).
    Tavily returns structured results with URLs and snippets.
    DuckDuckGo returns a plain text summary.
    """
    if os.getenv("TAVILY_API_KEY"):
        from langchain_community.tools.tavily_search import TavilySearchResults
        return TavilySearchResults(max_results=5)

    from langchain_community.tools import DuckDuckGoSearchRun
    return DuckDuckGoSearchRun()


def search_to_documents(query: str) -> list[Document]:
    """
    Run web search and normalise results to Documents regardless of which
    backend (Tavily vs DuckDuckGo) is active.
    """
    tool = get_web_search_tool()
    results = tool.invoke(query)

    # Tavily returns list[dict]; DuckDuckGo returns a plain string
    if isinstance(results, list):
        return [
            Document(
                page_content=r.get("content", ""),
                metadata={"source": r.get("url", "web_search"), "category": "web"},
            )
            for r in results if r.get("content")
        ]

    return [Document(page_content=str(results), metadata={"source": "web_search", "category": "web"})]
