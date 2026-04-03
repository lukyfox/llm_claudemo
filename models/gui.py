import gradio as gr
import json
from typing import Any

from logging import getLogger
logger = getLogger(__name__)

from models.claude_chat import ClaudeChat

class UserInterface:

    def __init__(self, client:ClaudeChat):
        """
        Constructor
        :param client: Anthropic model class instance
        """
        self.chat = client
        self.gui_texts = None
        self.system_prompts = None
        self.model_id = client.model
        self.keep_selected = False
        self.pending_file_paths = []
        # ---> GUI elements:
        self.user_specification_tar = None
        self.user_specification_tar_value = ''
        self.system_specification_tar = None
        self.chatbot = None
        self.msg = gr.Textbox()
        self.current_model_tbx = gr.Textbox(label="Current model", value=client.model)

    # ----------------------------------------------- #
    # ---> Model and Messaging related functions <--- #

    def set_chat_model(self, system_msg:str, user_selection:str) -> str:
        """
        Invoke setting of LLM model and post new
        :param system_msg: system message used for consideration of which model to use
        :param user_selection: user option - either direct selection of Haiku, Sonnet or Opus or let system to decide
        :return: model id as string
        """
        spec = user_selection
        if user_selection == 'by task':
            spec = system_msg or self.chat.system_message
        self.model_id = self.chat.client.get_fitting_model_id(spec)
        self.chat.model = self.model_id
        self.chat.set_system_message(system_msg)
        return self.model_id

    def _to_display_messages(self) -> list[dict]:
        """
        Convert API messages to Gradio-displayable format.
        Messages whose content is a list of blocks (file uploads) are replaced with a short placeholder
        so the raw file content is never shown in the chat window.
        The user's question text (always the last text block) is preserved in the placeholder.
        """
        display = []
        for msg in self.chat.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                display.append({"role": msg["role"], "content": content})
            elif isinstance(content, list):
                # The last block is always user's question, everything before it is a file
                user_text = next(
                    (block["text"] for block in reversed(content) if block.get("type") == "text"),
                    ""
                )
                file_count = len(content) - 1  # subtract the trailing user-text block
                label = f"📎 *[{file_count} file{'s' if file_count != 1 else '' } attached]*"
                placeholder = label + (f" — {user_text}" if user_text else "")
                display.append({"role": msg["role"], "content": placeholder})
        return display

    def send_user_msg(self, user_message:str) -> list[dict]:
        """
        Send user message - when there is a pending file then it must be appended to history
        :param user_message: user message to send
        :return: display messages history for Gradio chatbot
        """
        if not user_message:
            # empty message handling (preventing empty message error)
            return self._to_display_messages()

        file_processed = False
        if self.pending_file_paths:
            try:
                self.chat.add_files_to_conversation(
                    self.pending_file_paths,
                    user_message
                )
                file_processed = True  # user_message already included in file content block
            except ValueError as e:
                self.chat.add_message("user", f"**Error while processing file**: {e}")
            finally:
                self.pending_file_paths = []  # clear after send

        if not file_processed:
            # No file (or file failed) — append the user message on its own
            self.chat.add_message("user", user_message)
        return self._to_display_messages()

    def send_bot_msg(self) -> list[dict]:
        """
        Send chatbot message and run new round of conversation
        :return: display messages history for Gradio chatbot
        """
        if self.chat.messages:
            self.chat.make_chat_round()
        return self._to_display_messages()

    def get_model_id(self) -> str:
        """
        Get current model ID
        :return: model id as string (i.e. claude-haiku-4-5-20251001)
        """
        return self.chat.get_current_model_id()

    # ----------------------------------------- #
    # ---> File handling related functions <--- #

    def on_file_upload(self, filepaths: list[str]):
        """
        Save file paths (without sending files to chat) or clear if empty
        :param filepaths: list of full paths to uploaded files
        """
        self.pending_file_paths = filepaths if filepaths else []

    def on_file_delete(self):
        """
        File deleted by user or after processing
        """
        self.pending_file_paths = []

    def save_system_prompt(self, name=None, prompt=None, file:str="data/system_prompt.json"):
        """
        Save new system prompt to JSON file
        :param prompt:
        :param name:
        :param file:
        :return:
        """
        if not name:
            raise gr.Error(f"Prompt name cannot be empty! Enter unique name in the Dropdown list.")
        if not prompt:
            raise gr.Error(f"System prompt cannot be empty! "
                           f"Enter new specification or modify existing one with a new name.")
        # refresh system prompts (new may exist?)
        self.get_system_prompts()
        if name in self.get_system_prompt_choices():
            raise gr.Error(f"System prompt with name '{name}' already exists! Enter unique name in the Dropdown list.")
        value = {"name": name, "type": "user", "active": True, "message": prompt}
        self.system_prompts.append(value)
        with open(file, 'w') as f:
            f.write(json.dumps(self.system_prompts, indent=4))
        gr.Info(f"System prompt saved as '{name}'")
        return prompt

    def deactivate_system_prompt(self, name: str, file: str = "data/system_prompt.json") -> dict[str, Any]:
        """
        Set the active flag of the selected prompt to False and save to JSON.
        :param name: name of the prompt to deactivate
        :param file: path to JSON file
        :return: gr.update to refresh the dropdown
        """
        if not name:
            raise gr.Error("No prompt selected.")
        self.get_system_prompts()
        matched = False
        for item in self.system_prompts:
            if item["name"] == name:
                item["active"] = False
                matched = True
                break
        if not matched:
            raise gr.Error(f"Prompt '{name}' not found.")
        with open(file, 'w') as f:
            f.write(json.dumps(self.system_prompts, indent=4))
        gr.Info(f"System prompt '{name}' deactivated.")
        return gr.update(choices=self.get_system_prompt_choices(), value=None)

    # ------------------------------------- #
    # ---> GUI build related functions <--- #

    def get_gui_texts(self, file:str="data/gui_text.json"):
        """
        Get GUI texts from JSON file and transfer it to text dict
        :param file: texts
        """
        with open(file, 'r') as f:
            self.gui_texts = json.load(f)

    def get_system_prompts(self, file:str="data/system_prompt.json"):
        """
        Get stored system prompts from JSON file
        :param file: .json file containing system prompts
        """
        with open(file, 'r') as f:
            self.system_prompts = json.load(f)

    def get_system_prompt_choices(self) -> list[str]:
        """
        Get system prompt names to be used as Dropbox options
        :return: list of system prompt names
        """
        self.get_gui_texts()
        return [item["name"] for item in self.system_prompts if item["type"] == "user" and item["active"]]

    def refresh_system_prompt_dropdown(self, name: str) -> dict[str, Any]:
        """
        Refresh dropdown choices while keeping the given name selected.
        :param name: currently selected/saved prompt name
        :return: gr.update with refreshed choices and value preserved
        """
        return gr.update(choices=self.get_system_prompt_choices(), value=name)

    def get_system_prompt_value(self, selected:str="initial") -> str | dict[str, Any]:
        """
        Get value for the system prompt TextArea or update the component when message is not available
        to keep current value
        :param selected: key selected in dropdown
        :return: message (prompt) or update of the component
        """
        for item in self.system_prompts:
            if item["name"] == selected:
                return item["message"]
        # keep current textarea value when name doesn't match any stored prompt
        return gr.update()

    def build_gui(self) -> gr.Blocks:
        """
        Build GUI with all visual elements and workflows
        :return: Gradio Blocks instance
        """
        # read all texts for labels, placeholders etc.
        self.get_gui_texts()
        # read stored system prompts
        self.get_system_prompts()

        with gr.Blocks() as bl:
            with gr.Row():
                with gr.Column(scale=3):
                    self.system_specification_tar = gr.TextArea(
                        label="System prompt - leave it as it is || enter own definition || select from existing",
                        placeholder="Optional: Enter system prompt to specify communication style, role, "
                                    "output and other system parameters (i.e. Act as a molecular cuisine teacher "
                                    "willingness to provide step by step and cited explanation of...)."
                    )
                with gr.Column(scale=1):
                    # Selection from stored system prompts
                    system_prompt_drd = gr.Dropdown(
                        choices=self.get_system_prompt_choices(),
                        interactive=True,
                        label="Available system prompts",
                        allow_custom_value=True
                    )
                    system_prompt_drd.change(
                        fn=self.get_system_prompt_value,
                        inputs=system_prompt_drd,
                        outputs=self.system_specification_tar,
                    )
                    system_prompt_new_btn = gr.Button("Save as new").click(
                        fn=self.save_system_prompt,
                        inputs=[system_prompt_drd, self.system_specification_tar],
                        outputs=self.system_specification_tar,
                    ).then(
                        fn=self.refresh_system_prompt_dropdown,
                        inputs=system_prompt_drd,
                        outputs=system_prompt_drd,
                    )

                    system_prompt_deactivate_btn = gr.Button("Deactivate")
                    system_prompt_deactivate_btn.click(
                        fn=self.deactivate_system_prompt,
                        inputs=system_prompt_drd,
                        outputs=system_prompt_drd,
                    )

                with gr.Column(scale=1):
                    # Manual model selection and system prompt
                    model_selection_drd = gr.Dropdown(
                        ('by task', 'Haiku', 'Sonnet', 'Opus'),
                        interactive=True,
                        label="Select model"
                    )
                    self.current_model_tbx = gr.Textbox(label="Current model", value=self.model_id)
                    system_specification_apply_btn = gr.Button("Apply")
                    system_specification_apply_btn.click(
                        fn=self.set_chat_model,
                        inputs=[self.system_specification_tar, model_selection_drd],
                        outputs=self.current_model_tbx,
                        api_name="demo"
                    )

            # Chatbot window and user prompt with file uploader
            chatbot = gr.Chatbot(
                scale=1,
                # reasoning_tags=[("<thinking>", "</thinking>")]
            )

            with gr.Row():
                with gr.Column(scale=4):
                    self.msg = gr.Textbox(
                        placeholder="Write a message...",
                        show_label=False
                    )
                with gr.Column(scale=1, min_width=120):
                    send_btn = gr.Button("Send", variant="primary")

            with gr.Row():
                with gr.Column(scale=4):
                    file_selection = gr.File(
                        label="📎 Add files (py, ipynb, json, js, docx, xlsx, pdf, images)",
                        file_count="multiple",
                        file_types=[
                            ".py", ".ipynb", ".js", ".ts", ".jsx", ".tsx",
                            ".json", ".html", ".css", ".xml", ".yaml", ".yml",
                            ".md", ".txt", ".sql",
                            ".pdf",
                            ".docx", ".xlsx",
                            ".jpg", ".jpeg", ".png", ".gif", ".webp"
                        ],
                        elem_id="file-upload"
                    )
                with gr.Column(scale=1, min_width=120):
                    upload_additional_btn = gr.Button("➕ Add more files")
                    # Trigger the hidden file input inside the gr.File component via JS
                    upload_additional_btn.click(
                        fn=None,
                        js="() => {const inp = document.querySelector('#file-upload input[type=file]'); if (inp) inp.click();}"
                    )

            # Example user messages
            gr.Examples(
                [
                    ["What is your model id?"],
                    ["Switch your model into Sonnet"],
                    ["Analyze content of enclosed file"]
                ],
                self.msg
            )

            # File upload event
            file_selection.upload(
                fn=self.on_file_upload,
                inputs=file_selection
            )
            # File removed event
            file_selection.clear(
                fn=self.on_file_delete
            )

            # Sending a message has two triggers - keyboard and button - handle both in a loop
            # (no matter which one has triggered sending)
            for trigger in [self.msg.submit, send_btn.click]:
                trigger(
                    fn=self.send_user_msg,
                    inputs=self.msg,
                    outputs=chatbot,
                    queue=False
                ).then(
                    fn=lambda: ("", None),  # clear both msg textbox and file selection
                    outputs=[self.msg, file_selection]
                ).then(
                    fn=self.send_bot_msg,
                    outputs=chatbot
                ).then(
                    fn=self.get_model_id,  # updates textbox after each response
                    outputs=self.current_model_tbx
                )

        return bl