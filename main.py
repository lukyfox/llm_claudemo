import os

from dotenv import load_dotenv
from models.gui import UserInterface
from models.anthropic_ext import AnthropicExt
from models.claude_chat import ClaudeChat
from mcp_client.client import MCPDocumentClient  # <-- nový import

if __name__ == '__main__':
    # read environment variables
    load_dotenv()

    # MCP client runs server as subprocess by the first call via stdio transport
    mcp_client = MCPDocumentClient()
    mcp_client.connect()
    # create Anthropic API instance - use inherited version with extra methods
    anthropic = AnthropicExt(api_key=os.getenv("ANTHROPIC_API_KEY"))
    # get model id (latest Haiku) from Anthropic API
    model_id = anthropic.get_fitting_model_id('haiku')
    # create LLM instance with initial model and mcp client
    chat = ClaudeChat(anthropic, model_id, mcp_client=mcp_client)
    # initialize Gradio GUI
    gui = UserInterface(chat)
    # run it all together
    gui.build_gui().launch()
