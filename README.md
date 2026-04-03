# llm_claudemo

A demo application built on top of the **Anthropic Claude API**, showcasing practical AI engineering patterns in a structured, class-based Python project. Created as a hands-on companion to the [Anthropic Academy — Building with the Claude API](https://anthropic.skilljar.com/claude-with-the-anthropic-api) course.

---

## Features

- **Automatic model selection** — selects the most suitable Claude model (Haiku / Sonnet / Opus) based on the complexity of the task, either automatically via LLM inference or manually by the user
- **Automatic latest model IDs** — queries the Anthropic API at runtime to resolve the newest available model version, automatic selection from Haiku, Sonnet, Opus based on requirements/complexity of system prompt 
- **Multi-turn conversation** — full chat history management with proper tool exchange buffering (tool and file calls are kept separate from the display history)
- **Persistent system prompts** — save, load, and deactivate custom system prompts stored in JSON; select from a dropdown in the UI
- **Tool use / Function calling** — custom tools (`get_current_model_id`, `set_model_id`) with parallel tool call support; built-in `web_search` integration
- **Multimodal file upload** — attach and analyse files directly in the chat: Python, JS, JSON, Markdown, SQL, Jupyter notebooks (`.ipynb` with outputs), PDF, Word (`.docx`), Excel (`.xlsx`), and images
- **Gradio UI** — browser-based interface with model selection, system prompt management, file uploader, and example prompts

---

## Architecture

```
llm_claudemo/
├── main.py                  # Entry point
├── requirements.txt
├── data/
│   ├── system_prompt.json   # Persistent system prompts
│   └── rule.json            # Strict conversation rules injected into every system message
└── models/
    ├── config.py            # Centralised path configuration (pathlib-based)
    ├── anthropic_ext.py     # AnthropicExt — extends Anthropic client with model selection logic
    ├── claude_chat.py       # ClaudeChat — conversation state, tool loop, file bundling
    ├── file_processor.py    # FileProcessor — multimodal file parsing (text, PDF, DOCX, XLSX, images)
    └── gui.py               # UserInterface — Gradio layout and event wiring
```

**Key design decisions:**

- `AnthropicExt` inherits from `Anthropic` directly rather than wrapping it — extends the client without hiding its interface
- Tool exchange is buffered locally during a `while stop_reason == "tool_use"` loop and only merged into `self.messages` after the final text response — this keeps the Gradio chat history clean
- `FileProcessor` uses `@classmethod` methods only — stateless, easily extensible
- All data file paths are resolved via `Config` using `pathlib`, making the app portable regardless of working directory

---

## Requirements

- Python 3.11+
- Anthropic API key

---

## Installation

```
git clone https://github.com/lukyfox/llm_claudemo.git
cd llm_claudemo
pip install -r requirements.txt
```

Create a `.env` file in the project root and add:

```
ANTHROPIC_API_KEY=your-api-key-here
```

---

## Run

```
python main.py
```

The Gradio interface will open in your browser at `http://localhost:7860`(port number may change in you case).

---

## Usage

### Model selection
Choose between **Haiku**, **Sonnet**, **Opus**, or **by task** — the last option lets Claude itself decide which model is most appropriate based on the system prompt you have set.

### System prompts
Select a pre-saved prompt from the dropdown, or write your own and save it with a unique name. Prompts can be deactivated (hidden from the list) without deleting them.

### File upload
Attach one or more files using the file picker, then send a message — the files and your question are bundled into a single API call. Supported formats: `.py`, `.js`, `.ts`, `.json`, `.html`, `.css`, `.xml`, `.yaml`, `.md`, `.txt`, `.sql`, `.ipynb`, `.pdf`, `.docx`, `.xlsx`, `.jpg`, `.png`, `.gif`, `.webp`

### Example prompts
Click any example below the chat to pre-fill the input:
- *"What is your model id?"*
- *"Switch your model into Sonnet"*
- *"Analyze content of enclosed file"*

---

## Concepts demonstrated

| Concept | Where |
|---|---|
| Anthropic API — Messages, Tools, Models | `anthropic_ext.py`, `claude_chat.py` |
| Tool use / Function calling | `claude_chat.py` — `handle_tool_calls`, tool schemas |
| Parallel tool calls | `claude_chat.py` — system prompt appendix + tool loop |
| Model selection via LLM inference | `anthropic_ext.py` — `get_fitting_model_id` |
| Multimodal inputs (PDF, images, DOCX, XLSX) | `file_processor.py`, `claude_chat.py` |
| Chat history management | `claude_chat.py` — `messages`, `tool_exchange` buffer |
| System prompt engineering | `data/rule.json`, `data/system_prompt.json` |
| Gradio UI with chained events | `gui.py` — `.then()` chains, file upload, dropdown |

---

## Related

- [llm_engineering](https://github.com/lukyfox/llm_engineering) — weekly exercises from Ed Donner's LLM Engineering course (PR #1 merged into upstream, many will follow...)
- [Anthropic Academy — Building with the Claude API](https://anthropic.skilljar.com/claude-with-the-anthropic-api)

---

## License

MIT
