# BUTLER (v2.2) Update: Model and API Key rotation + multi-key API support

BUTLER is an advanced AI assistant that operates on your local machine. It combines the privacy and control of local-first tools with the immense reasoning speed of cutting-edge APIs, all wrapped up in a sleek, voice-activated desktop widget.

## Key Features
- **Sleek Desktop GUI**: A floating, draggable widget out-of-the-box (`--gui`).
- **Voice Capabilities**: Built-in Speech-to-Text (`faster-whisper`), continuous Wake-Word detection, and premium Text-to-Speech (`edge-tts`). 
- **Tri-Provider Architecture**: Run completely offline using **Ollama**, or switch to ultra-fast cloud inference using the **Gemini API** (`--gem`) or **Anthropic Claude API** (`--claude`). 
- **High Availability & Failover**: Automatic multi-key API rotation with mandatory labels, and intelligent model fallback rotation if endpoints hit rate-limits or experience outages.
- **Safe-by-default**: File access is tightly allowlisted; write actions require explicit confirmation.
- **Persistent Memory**: Stores chat, tool audit logs, and note metadata in thread-safe SQLite.
- **Web-Connected**: Can instantly pull live facts, weather, and breaking news utilizing DuckDuckGo (`web.news` & `web.search`).

## Quickstart (Windows PowerShell)

### 1) Prerequisites
- Python 3.10+
- *(Optional)* Ollama running locally (for offline use).
  - `ollama pull mistral:7b-instruct`

### 2) Installation

```powershell
# Clone the repository and navigate into it
cd C:\Users\HP\OneDrive\Desktop\BUTLER

# Create your virtual environment and install the package
python -m venv butenv
.\butenv\Scripts\python.exe -m pip install -e ".[dev]"
```

*Note: For the Voice Widget to work, `pyaudio`, `faster-whisper`, `edge-tts`, `customtkinter`, and `pygame` will be installed.*

### 3) Run

**Launch the Voice Desktop Widget (Recommended)**
```powershell
.\butenv\Scripts\butler.exe --claude --gui
```
*If you haven't set up Claude or Gemini yet, the CLI will prompt you for your key on the first run.*

**Launch the Interactive Terminal REPL**
```powershell
.\butenv\Scripts\butler.exe --claude
```

**One-shot CLI Execution**
```powershell
.\butenv\Scripts\butler.exe --chat "Summarize the latest news on AI"
```

## How BUTLER Works (Under The Hood)

BUTLER operates on a sophisticated loop that analyzes your input to determine if you are making conversation or asking it to perform an actionable task.

1. **Chat Mode**
- Bypasses the tool loop for extremely fast, conversational output.
- Excellent for casual prompts or asking generic knowledge questions.

2. **Action Mode (Tool Using)**
- Safely enforces a strict JSON "action" object parsing sequence.
- Action Mode loops until the task completes (e.g., File search -> Read file -> Summarize file).
- If the model's structure violently fails (bad JSON), BUTLER intercepts and uses self-healing logic to attempt to format the response natively.

## Commands (Inside the CLI Terminal)

Help:
- `/help`

Allowlist folders (Required to grant the AI file access):
- `/roots add C:\path\to\folder`
- `/roots list`

Indexing (Builds a lightning-fast vector-style searchable index for `.md`/`.txt`):
- `/index sync`

Notes Management:
- `/notes create "title" "content"`
- `/notes append "title" "more content"`
- `/notes read "title"`
- `/notes search "query"`

System:
- `/config` (Prints the current system setup)
- `/exit`

## Built-In Tools

BUTLER's internal arsenal of capabilities exposed to Action Mode:
- `system.now` (Time and location)
- `web.search` / `web.news` / `weather.current` (Live external data)
- `files.list` (Read-only, strictly within allowlisted roots)
- `files.read_text` (Read-only text scraping)
- `files.search` (Searches the user-synced index created by `/index sync`)
- `notes.create` / `notes.append` (Writes, requires user confirmation)
- `notes.read` / `notes.search` (Read-only)

## Configuration and Environment

BUTLER stores state files inside a "BUTLER home" directory (Defaulting to `%LOCALAPPDATA%\BUTLER\` or `.butler\`).
Inside BUTLER home you will find:
- `config.json` (Stores your API keys with labels, fallback models, and system preferences)
- `butler.sqlite3`
- `notes\` directory

**Configuration Flags:**
- Manage API Keys & Settings: `butler --change`
- Enable Gemini Provider: `butler --gem` or `butler --gemini`
- Launch GUI: `butler --gui`
- Override Provider via Env: `BUTLER_PROVIDER=gemini` or `BUTLER_PROVIDER=ollama`
- Default Local Model: `BUTLER_MODEL=mistral:7b-instruct`

## Debugging

If you want to look at the raw un-sanitized matrices and parse loops:
```powershell
$env:BUTLER_LOG_LEVEL="DEBUG"
.\butenv\Scripts\butler.exe --chat "Say hi"
```

Common Issues:
- **"No allowed roots configured"**: Run `/roots add C:\Users\YourName\Documents` to give BUTLER a folder to index.
- **Microphone not working**: Ensure `PyAudio` installed cleanly, and that your OS isn't blocking microphone permissions for Python.
- **Database Threading Crash**: BUTLER requires `WAL` mode and `check_same_thread=False` across its background Audio threads.

## What's Next?
1) Training custom `.onnx` models for dynamic wake-words in `openwakeword`.
2) Deep Research background agents.
3) Executable API tools for OS-level control (Volume, Display brightness, app launching).
