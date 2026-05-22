"""
MCP client — connects to the Dubai Holding Foundry MCP endpoint via
streamable HTTP, fetches available tools, and runs a Claude agent loop.

Run:
    python -m src.mcp.server                        # default query
    python -m src.mcp.server "your question here"   # custom query
"""

import asyncio
import sys

from anthropic import AsyncAnthropic
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_FOUNDRY_BASE_URL,
    FOUNDRY_TOKEN,
    MCP_SERVER_URL,
    SONNET_MODEL,
)


async def run_agent(query: str) -> None:
    # Use Azure Foundry base URL if configured, otherwise default to api.anthropic.com
    client = AsyncAnthropic(
        api_key=ANTHROPIC_API_KEY,
        **({} if not ANTHROPIC_FOUNDRY_BASE_URL else {"base_url": ANTHROPIC_FOUNDRY_BASE_URL}),
    )

    async with streamablehttp_client(
        MCP_SERVER_URL,
        headers={"Authorization": f"Bearer {FOUNDRY_TOKEN}"},
        timeout=30.0,
        sse_read_timeout=300.0,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in tools_result.tools
            ]
            print(f"[MCP] Connected — {len(tools)} tools available")

            messages = [{"role": "user", "content": query}]

            # Agentic loop: keep running until Claude stops requesting tool calls
            while True:
                response = await client.messages.create(
                    model=SONNET_MODEL,
                    max_tokens=4000,
                    tools=tools,
                    messages=messages,
                )

                # Append assistant turn to message history
                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "end_turn":
                    # Extract and print the final text response
                    for block in response.content:
                        if hasattr(block, "text"):
                            print(block.text)
                    break

                if response.stop_reason == "tool_use":
                    # Execute each tool call and collect results
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            print(f"[MCP] Calling tool: {block.name}({block.input})")
                            result = await session.call_tool(block.name, arguments=block.input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": str(result.content),
                            })

                    # Feed tool results back to Claude
                    messages.append({"role": "user", "content": tool_results})
                else:
                    # Unexpected stop reason — print and exit
                    print(f"[MCP] Unexpected stop_reason={response.stop_reason}")
                    print(response)
                    break


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Your query here"
    asyncio.run(run_agent(query))
