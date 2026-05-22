import os

from langchain_core.tools import BaseTool

from src.tools.web_search import get_web_search_tool, search_to_documents
from src.tools.sql_query import get_sql_tools


def get_all_tools() -> list[BaseTool]:
    """
    Collect every available tool based on environment configuration.
    Used when building an agent-style graph that needs tool access.
    """
    tools: list[BaseTool] = [get_web_search_tool()]
    tools.extend(get_sql_tools())
    return tools


__all__ = [
    "get_all_tools",
    "get_web_search_tool",
    "search_to_documents",
    "get_sql_tools",
]
