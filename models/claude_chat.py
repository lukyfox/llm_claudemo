import json
from typing import Any

from anthropic._exceptions import OverloadedError, RateLimitError
from anthropic.types import ToolParam, TextBlock, ToolUseBlock


from .file_processor import FileProcessor
from .anthropic_ext import AnthropicExt
from time import sleep


class ClaudeChat:
    """
    Definition of LLM model (chat) class
    """

    def __init__(
            self, client:AnthropicExt, model:str,
            user_message:str = '', system_message:str = '', parallel_tools:bool = True
    ):
        self.client = client
        self.model = model
        self.message = user_message
        self.messages = [{'role': 'user', 'content': user_message}] if user_message else []
        self.system_message = system_message or "You are a helpful assistant."
        self._parallel_tools_appendix = ""
        if parallel_tools:
            self._parallel_tools_appendix = ("\n<use_parallel_tool_calls>"
                               "For maximum efficiency, whenever you perform multiple independent operations, "
                               "invoke all relevant tools simultaneously rather than sequentially. Prioritize calling "
                               "tools in parallel whenever possible. For example, when reading 3 files, "
                               "run 3 tool calls in parallel to read all 3 files into context at the same time. "
                               "When running multiple read-only commands like `ls` or `list_dir`, "
                               "always run all of the commands in parallel."
                               "</use_parallel_tool_calls>")
            self.system_message += self._parallel_tools_appendix

        self.get_current_model_id_schema = ToolParam(
            {
                "name": "get_current_model_id",
                "description": "Get the model id or name of the currently running client or chat",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }
            })

        self.set_model_id_schema = ToolParam(
            {
                "name": "set_model_id",
                "description": "Get the Anthropic Claude model id to be set for current chat. Model id can be requested directly by model name (like Haiku, Opus) or by exact model ID (like claude-sonnet-4-6) or even on base of a description or purpose given by user",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "model name (like Haiku, Opus) or exact model ID (like claude-sonnet-4-6) or a description of what the requested model should used for",
                        },
                    },
                    "required": [],
                    "additionalProperties": False
                }
            }
        )

        self.tools = [
            # custom
            self.get_current_model_id_schema,
            self.set_model_id_schema,
            # built-in
            {"type": "web_search_20250305", "name": "web_search"}
        ]

    def add_message(self, role: str,  message: str):
        self.messages.append({'role': role, 'content': message})

    def set_system_message(self, message: str=None):
        if not message:
            message = "You are a helpful assistant."
        with open(file="data/rule.json", mode="r", encoding="utf-8") as f:
            rules = json.load(f)
            rules = '\n'.join(rules)
            message += '\n' + rules
        self.system_message = message

    def make_chat_round(self, repeated=0):
        try:
            chat = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=self.messages,
                system=self.system_message,
                tools=self.tools
            )
            # Accumulate tool exchanges locally — persisted to self.messages only after
            # the final text response, keeping self.messages clean for Gradio display
            tool_exchange = []
            while chat.stop_reason == "tool_use":
                all_tool_results = []
                for item in chat.content:
                    if isinstance(item, ToolUseBlock):
                        result = self.handle_tool_calls(item)
                        all_tool_results.extend(result['content'])
                tool_exchange.append({"role": "assistant", "content": chat.content})
                tool_exchange.append({"role": "user", "content": all_tool_results})
                chat = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    tools=self.tools,
                    messages=self.messages + tool_exchange
                )

            response_text = "".join([item.text for item in chat.content if item.type == 'text'])
            self.add_message(chat.role, response_text)
        except OverloadedError as e:
            if repeated > 2:
                raise Exception(f"Several attempts to process message failed due to OverloadedError: " + e.message)
            sleep(1)
            repeated += 1
            self.make_chat_round(repeated)
        except RateLimitError as e:
            self.add_message(
                "assistant",
            "Rate limit exceeded. Please try again later or reduce length of your prompt or restart conversation.")

    def set_model_id(self, model_id = None):
        """
        Select appropriate model according to chat history and system prompt (i.e. "claude-haiku-4-5-20251001"
        for a simple communication) or according to model given by parameter.
        :param model_id: model name (then search for latest model) or model id (the search for exact match) to find.
                         When the parameter model_id is empty then use chat history for best fitting model estimation
        """
        if not isinstance(model_id, str):
            message = (f"system prompt: {self.system_message}"
                       f"\nlatest chat_history: {self.messages if len(self.messages) < 11 else self.messages[-10:]}")
        else:
            message = model_id
        self.model = self.client.get_fitting_model_id(message)

    def get_current_model_id(self):
        """
        Return currently used model id
        :return: as string
        """
        return self.model

    def __str__(self):
        return f"{self.messages[-1]['role']}: {self.messages[-1]['content']}"

    def handle_tool_calls(self, tool_call) -> list[dict[str, Any]]:
        """

        :param tool_call:
        :return:
        """
        responses = {"role": 'user', "content": []}
        func_res = None
        # for tool_call in tool_use_block:
        if tool_call.type == 'tool_use':
            if tool_call.name == "get_current_model_id":
                func_res = self.get_current_model_id()
            elif tool_call.name == "set_model_id":
                self.set_model_id(tool_call.input.get("description"))
                func_res = self.model
            else:
                func_res = (f"Sorry, I cannot answer your question because I have no tool to do so "
                            f"(maybe implementation of {tool_call.name} would help?)")
        responses['content'].append({
            "type": "tool_result",
            "content": func_res,
            "tool_use_id": tool_call.id
        })
        return responses

    def add_files_to_conversation(self, file_paths: list[str], user_request: str):
        """
        Add one or more files together with the user request as a single user message.
        All file blocks are bundled into one message because the Anthropic API does not
        allow consecutive messages with the same role.
        :param file_paths: list of paths to files to include
        :param user_request: user instruction / question accompanying the files
        """
        # processor = FileProcessor()
        blocks = []

        for file_path in file_paths:
            result = FileProcessor.process(file_path)
            if result["type"] == "text_content":
                blocks.append({
                    "type": "text",
                    "text": (
                        f"<file name='{result['filename']}'>{result['text']}</file>"
                    )
                })
            elif result["type"] == "pdf":
                blocks.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": result["data"]
                    }
                })
            elif result["type"] == "image":
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": result["media_type"],
                        "data": result["data"]
                    }
                })

        # User request goes last so it reads naturally as "here are the files, now the question"
        blocks.append({"type": "text", "text": user_request})
        self.messages.append({"role": "user", "content": blocks})