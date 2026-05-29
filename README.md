# Learning Agent System

AI-powered checkpoint learning platform with adaptive assessment, Feynman-style remediation, and Streamlit UI.

## Quick Start

### 1) Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running

Pull a model:

```bash
ollama pull llama3.1
```

### 2) Install dependencies

```bash
# Windows (PowerShell)
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 3) Configure environment

```bash
# Copy example env
copy .env.example .env
```

Update `.env` as needed (model, LangSmith keys, etc.).

For Streamlit secrets, use:

```bash
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
```

For cloud-friendly model routing, set:

```env
LLM_PROVIDER=auto
HF_TOKEN=your_hf_token
HF_MODEL=HuggingFaceH4/zephyr-7b-beta
```

### 4) Run the app

```bash
streamlit run app.py
```

To auto-start Ollama first (Windows):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_streamlit_with_ollama.ps1
```

App URL (default): `http://localhost:8501`

## Modes

- Web UI: `streamlit run app.py`
- CLI flow: `python -m src.multi_checkpoint`

## What the App Does

- Checkpoint-based progression through a learning path
- Optional user file upload (`.pdf`, `.docx`, `.md`, `.txt`)
- Dynamic material generation + context processing
- Assessment per checkpoint with enforced question mix:
  - 5 MCQ
  - 3 short-answer
  - 2 long-answer
- 70% threshold to pass a checkpoint
- Feynman teaching fallback when score is below threshold

## LangSmith Tracing

Tracing is optional. The UI has a **LangSmith Status** panel showing current runtime status.

Set these in `.env`:

```env
LANGSMITH_TRACING_ENABLED=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=Learning-Agent-System
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
# EU accounts:
# LANGCHAIN_ENDPOINT=https://eu.api.smith.langchain.com
```

After updating env values, restart Streamlit.

## Troubleshooting

### Ollama not reachable

```bash
ollama serve
ollama list
```

### Streamlit deployment note

If deploying on Streamlit Community Cloud, local Ollama cannot run inside that environment.
Use one of these:

- `LLM_PROVIDER=auto` + `HF_TOKEN` (recommended): auto-falls back to Hugging Face
- A reachable remote Ollama URL in `OLLAMA_BASE_URL`

### Streamlit starts but no AI responses

- Confirm Ollama is running on `http://localhost:11434`
- Verify `OLLAMA_MODEL` exists locally (`ollama list`)

### Hugging Face rate-limit warning

Optional: set `HF_TOKEN` in `.env`.

## Project Files

- `app.py` - Streamlit application
- `src/` - core workflow, LLM, context, and evaluation modules
- `.env.example` - sample environment configuration
- `custom_topics.json` - custom learning paths
- `Documentation.md` - detailed technical docs

## License

MIT - see [LICENSE](LICENSE)
