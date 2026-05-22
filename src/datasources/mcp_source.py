"""
MCP query-time retrieval.

Connects to the Dubai Holding Foundry MCP endpoint, calls available tools
with the user's query, and returns results as LangChain Documents.

This is used at query time (not indexing time) to augment vector store results
with real-time data from the Foundry.
"""

from langchain_core.documents import Document
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src.config import FOUNDRY_TOKEN, MCP_SERVER_URL


async def fetch_mcp_documents(query: str) -> list[Document]:
    """
    Call MCP tools with the user query and return responses as Documents.

    Tools that don't accept a "query" argument are skipped gracefully.
    Returns an empty list if FOUNDRY_TOKEN is unset or connection fails.
    """
    if not FOUNDRY_TOKEN:
        return []

    docs: list[Document] = []

    try:
        async with streamablehttp_client(
            MCP_SERVER_URL,
            headers={"Authorization": f"Bearer {FOUNDRY_TOKEN}"},
            timeout=30.0,
            sse_read_timeout=60.0,
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools_result = await session.list_tools()
                tools = tools_result.tools
                print(f"[MCP] {len(tools)} tools available — fetching for query: {query!r}")

                for tool in tools:
                    try:
                        result = await session.call_tool(
                            tool.name,
                            arguments={"query": query},
                        )
                        if not result.content:
                            continue

                        text_parts = [
                            block.text if hasattr(block, "text") else str(block)
                            for block in result.content
                        ]
                        content = "\n".join(text_parts).strip()
                        if content:
                            docs.append(Document(
                                page_content=content,
                                metadata={
                                    "source": f"mcp:{tool.name}",
                                    "category": "mcp",
                                },
                            ))
                    except Exception as exc:
                        print(f"[MCP] Skipped tool '{tool.name}': {exc}")

    except Exception as exc:
        print(f"[MCP] Connection failed — skipping MCP results: {exc}")

    print(f"[MCP] Retrieved {len(docs)} documents from MCP tools")
    return docs
