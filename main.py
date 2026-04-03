import os

from dotenv import load_dotenv
from models.gui import UserInterface
from models.anthropic_ext import AnthropicExt
from models.claude_chat import ClaudeChat

if __name__=='__main__':
    # read environment variables
    load_dotenv()
    # create Anthropic API instance - use inherited version with extra methods
    anthropic = AnthropicExt(api_key=os.getenv("ANTHROPIC_API_KEY"))
    # get model id (latest Haiku) from Anthropic API
    model_id = anthropic.get_fitting_model_id('haiku')
    # create LLM instance with initial model
    chat = ClaudeChat(anthropic, model_id)
    # initialize Gradio GUI
    gui = UserInterface(chat)
    # run it all together
    gui.build_gui().launch(theme='ocean')
