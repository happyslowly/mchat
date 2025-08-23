# mchat

A clean, fast terminal chat client for local LLMs with streaming responses and conversation history.

## Features

- **Streaming responses** with thinking process display
- **Conversation history** with persistence
- **Rich terminal UI** with markdown rendering
- **Vi/Emacs editing modes** via prompt-toolkit
- **Model switching** at runtime
- **System prompt management**

## Installation

```bash
pip install -e .
```

## Configuration

Create `~/.config/mchat/config.toml`:

```toml
base_url = "http://localhost:8000/v1"
model = "your-model-name"
api_key = "optional-api-key"
```

## Usage

```bash
mchat
```

### Commands

- `/help` - Show available commands
- `/models` - List available models (* = current)
- `/model <name>` - Switch to specified model
- `/system [prompt]` - View or set system prompt
- `/edit_mode [vi|emacs]` - Switch editing mode
- `/show_history` - Print conversation history
- `/clear_history` - Clear conversation history
- `/quit` - Exit (or Ctrl+C/Ctrl+D)

## Requirements

- Python 3.12+
- Compatible with OpenAI API format endpoints