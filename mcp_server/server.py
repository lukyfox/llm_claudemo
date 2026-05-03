import os
import sys

curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(curr_dir))

from mcp.server.fastmcp import FastMCP
from mcp_server.knowledge_base import KBase
from mcp_server.sampling import request_summary

kb = KBase()
mcp = FastMCP(
    name="document-intelligence",
    instructions=(
        "You have access to a persistent document knowledge base. "
        "Use add_document to index files, search_knowledge_base to retrieve "
        "relevant passages, list_documents to see what is indexed, and "
        "summarize_document to generate an AI summary via sampling."
    )
)


@mcp.tool()
def add_document(file_path: str) -> str:
    """
    Index a file into the knowledge base (chroma)
    Returns a confirmation with the number of chunks stored.
    """
    try:
        result = kb.add_document(file_path)
        return (f"Document '{result['filename']}' indexed successfully. "
                f"{result['chunks_added']} chunks stored.")
    except Exception as e:
        return f"Error indexing document: {e}"


@mcp.tool()
def search_knowledge_base(query: str, n_results: int = 5) -> str:
    """
    Search the knowledge base for passages relevant to a query.
    Returns up to n_results ranked chunks with source attribution.
    """
    results = kb.search(query, n_results=n_results)
    if not results:
        return "No results found. The knowledge base may be empty."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] Source: {r['source']} | Chunk: {r['chunk']} "
            f"| Relevance: {r['relevance']}\n{r['text']}"
        )
    return "\n\n---\n\n".join(lines)


@mcp.tool()
def list_documents() -> str:
    """
    List all documents currently indexed in the knowledge base
    """
    docs = kb.list_documents()
    if not docs:
        return "Knowledge base is empty."
    return "Indexed documents:\n" + "\n".join(f"  - {d}" for d in docs)


@mcp.tool()
def summarize_document(filename: str, focus: str = "") -> str:
    """
    Generate an AI summary of an indexed document.
    Optionally provide a focus topic to narrow the summary.
    Sampling means the MCP server requests a completion from the Claude client —
    the server does not call the Anthropic API directly, the work remains on client side (requested behaviour)
    """
    chunks = kb.search(
        query=focus or f"main topics and key information in {filename}",
        n_results=5
    )
    relevant = [c for c in chunks if c["source"] == filename]
    if not relevant:
        return f"Document '{filename}' not found in knowledge base."

    context = "\n\n".join(c["text"] for c in relevant)
    prompt = (
        f"Summarize the following document content"
        + (f" with focus on: {focus}" if focus else "")
        + f".\n\nContent:\n{context}"
    )
    return request_summary(mcp, prompt)

if __name__ == "__main__":
    mcp.run(transport="stdio")