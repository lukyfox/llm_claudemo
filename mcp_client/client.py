import asyncio
import threading
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic.types import ToolParam


class MCPDocumentClient:

    SERVER_SCRIPT = "mcp_server/server.py"

    def __init__(self):
        self._tools: list[ToolParam] = []
        self._session: ClientSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def connect(self):
        """
        Run client loop
        """
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        # Run event loop in new thread - this way avoids conflicts with Gradio's event loop
        future = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
        future.result(timeout=15)  # max 15s to start the server (timeout)

    async def _connect(self):
        server_params = StdioServerParameters(
            command="python",
            args=[self.SERVER_SCRIPT]
        )
        self._transport = stdio_client(server_params)
        read, write = await self._transport.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        self._tools = await self._build_tool_params()

    def call_tool(self, name: str, arguments: dict) -> str:
        """
        Calling of MCP tool
        """
        if not self._session or not self._loop:
            raise RuntimeError("MCP client not connected.")
        future = asyncio.run_coroutine_threadsafe(
            self._call_tool_async(name, arguments),
            self._loop
        )
        return future.result(timeout=30)

    async def _call_tool_async(self, name: str, arguments: dict) -> str:
        result = await self._session.call_tool(name, arguments)
        return "\n".join(
            block.text for block in result.content
            if hasattr(block, "text")
        )

    @property
    def tools(self) -> list[ToolParam]:
        return self._tools

    async def _build_tool_params(self) -> list[ToolParam]:
        response = await self._session.list_tools()
        return [
            ToolParam({
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema
            })
            for tool in response.tools
        ]

    def disconnect(self):
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)