from mcp.server.fastmcp import FastMCP
from mcp import types


def request_summary(mcp: FastMCP, prompt: str) -> str:
    """
    Use MCP sampling to request a completion from the connected client.
    The server sends a CreateMessageRequest — the client (Claude) responds.
    This is the key distinction from a direct API call:
    the server delegates generation back to the host application.
    """
    result = mcp.get_context().session.create_message(
        messages=[
            types.SamplingMessage(
                role="user",
                content=types.TextContent(type="text", text=prompt)
            )
        ],
        max_tokens=512,
        system_prompt=(
            "You are a precise document analyst. "
            "Summarize clearly and concisely, preserving key facts and numbers."
        )
    )
    if result.content.type == "text":
        return result.content.text
    return "Sampling returned non-text content."