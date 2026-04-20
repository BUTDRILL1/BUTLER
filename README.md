# BUTLER (v1)

BUTLER is a local-first AI assistant that runs on your machine, uses Ollama for the LLM, and can safely use tools
(files, notes, indexing) with guardrails.

Key properties:
- Local-first: talks to Ollama at `http://127.0.0.1:11434`
- Safe-by-default: file access is allowlisted; write actions require confirmation
- Persistent: stores chat + tool audit logs + note metadata in SQLite
- Robust: handles imperfect model output with repair + fallbacks
- Single-model runtime: chat, action, routing, repair, and summarization all use one Ollama model

## Quickstart (Windows PowerShell)

### 1) Prereqs
- Python 3.10+
- Ollama installed and running
- Pull the default model (`mistral:7b-instruct`):
  - `ollama pull mistral:7b-instruct`
  - Verify: `ollama list`

### 2) Create a venv and install

```powershell
cd C:\Users\HP\OneDrive\Desktop\BUTLER
python -m venv butenv
.\butenv\Scripts\python.exe -m pip install -e ".[dev]"
```

### 3) Run

```powershell
.\butenv\Scripts\butler.exe
```

One-shot:
```powershell
.\butenv\Scripts\butler.exe --chat "Say hi"
```

## How BUTLER Works (Beginner-Friendly)

BUTLER has two LLM modes:

1. Chat Mode (fast)
- Normal conversation output (plain text).
- Used for small talk like "Say hi".

2. Action Mode (tool-using)
- The model must output a strict JSON "action" object.
- Used when your message looks tool-related (files/notes/search/index).
- Action Mode can call tools, then see tool results, then produce a final answer.

If Action Mode fails (bad JSON, wrong schema, refusal text), BUTLER falls back to Chat Mode so you still get a human-style reply.

One important detail:
- BUTLER uses the same Ollama model for both modes.
- You only choose the model once, through `BUTLER_MODEL` or the saved config file.

## Commands (Inside the Interactive CLI)

Help:
- `/help`

Allowlist folders (required before file tools work):
- `/roots add C:\path\to\folder`
- `/roots list`

Indexing (builds searchable index for `.md`/`.txt` under allowlisted roots):
- `/index sync`

Notes:
- `/notes create "title" "content"`
- `/notes append "title" "more content"`
- `/notes read "title"`
- `/notes search "query"`

Config view:
- `/config`

Exit:
- `/exit`

## Tools (What the Model Can Call)

These are internal capabilities exposed to Action Mode:
- `system.now` (read-only)
- `files.list` (read-only, within allowlisted roots)
- `files.read_text` (read-only, within allowlisted roots, size capped)
- `files.search` (read-only, searches the index created by `/index sync`, returns <= 10 short snippets)
- `notes.create` / `notes.append` (writes, asks for confirmation)
- `notes.read` / `notes.search` (read-only)
- `index.sync` (side effect: rebuilds index)

## Data and Storage

BUTLER stores state under a "BUTLER home" directory.

Default resolution (first writable wins):
1) `%LOCALAPPDATA%\BUTLER\`
2) `%APPDATA%\BUTLER\`
3) `%USERPROFILE%\.butler\`
4) `.butler\` (inside the repo)

You can override explicitly:
- `BUTLER_HOME`: a writable folder (recommended if you hit permissions issues)

Inside BUTLER home:
- `config.json`
- `butler.sqlite3`
- `notes\` (markdown files)

## Configuration and Env Vars

General:
- `BUTLER_HOME`: override data directory
- `BUTLER_OLLAMA_URL`: default `http://127.0.0.1:11434`
- `BUTLER_MODEL`: default `mistral:7b-instruct`
- `BUTLER_LOG_LEVEL`: set to `DEBUG` for deep troubleshooting

Model reliability (timeouts/retries):
- `BUTLER_MODEL_TIMEOUT_SECONDS`: default `180`
- `BUTLER_MODEL_RETRY_COUNT`: default `1`
- `BUTLER_MODEL_TOTAL_TIMEOUT_SECONDS`: default `220` (caps total time across retries)

## Debugging (Beginner to Expert)

### Turn on debug logs

```powershell
$env:BUTLER_LOG_LEVEL="DEBUG"
.\butenv\Scripts\butler.exe --chat "Say hi"
```

What you will see in DEBUG:
- The raw model output and "cleaned" output (after fence stripping/unescaping).
- Parse events like:
  - `parse_stage=direct_json|json_string|escaped_json|repair_a|repair_b|fallback`
  - `parse_confidence=direct|repaired|fallback`

### Common issues

1) Ollama is running, but BUTLER times out
- Some prompts take longer in Action Mode because it may do repairs and tool loops.
- Increase timeouts via env vars above.

2) PowerShell `curl` confusion
- In PowerShell, `curl` is an alias to `Invoke-WebRequest`.
- Use `curl.exe` if you want real curl behavior.

3) "No allowed roots configured"
- Add at least one root:
  - `/roots add C:\Users\HP\Documents`

## Development

Run tests:
```powershell
.\butenv\Scripts\pytest.exe -q
```

Repo layout:
- `src\butler\cli.py`: CLI UX and confirmations
- `src\butler\agent\loop.py`: routing + tool loop + fallbacks
- `src\butler\agent\parsing.py`: robust parsing and schema validation
- `src\butler\tools\`: tool registry and implementations
- `src\butler\db.py`: SQLite schema and persistence

## What To Build Next

Good next upgrades:
1) Incremental indexing (avoid full rebuild on every `/index sync`)
2) Add a Web UI (keep the same core runtime and tool layer)
3) Improve tool-selection heuristics for more reliable action routing
