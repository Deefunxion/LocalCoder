# Academicon Code Assistant - OpenRouter Edition

Quick start scripts for Windows users.

## Available Commands

### 1. **Index the Codebase** (Do this FIRST)
```batch
index_academicon_lite.bat
```
- Indexes only Python files (faster)
- Uses GPU if available
- Creates `academicon_chroma_db/`
- Takes 10-15 minutes with GPU

### 2. **Test in Command Line**
```batch
test_assistant.bat
```
- Interactive query mode
- Test before using web UI
- Requires indexed database

### 3. **Start Web UI** (Recommended)
```batch
start_web_ui.bat
```
- Beautiful Gradio interface
- Real-time conversation
- Query metrics and source attribution
- Open: http://127.0.0.1:7860

## Python Commands (for advanced users)

```bash
# Index codebase
python src/index_academicon_lite.py

# Command-line testing
python src/main_openrouter.py

# Web UI
python src/web_ui_openrouter.py
```

## Configuration

Edit `.env` file with your settings:

```bash
# REQUIRED - Get from https://openrouter.ai/
OPENROUTER_API_KEY=sk-or-v1-...

# Optional - change these if needed
ORCHESTRATOR_MODEL=meta-llama/llama-2-7b-chat
SYNTHESIZER_MODEL=openai/gpt-4-turbo-preview
SYNTHESIZER_TEMPERATURE=0.2
```

## Troubleshooting

**"Vector database not found"**
- Run `index_academicon_lite.bat` first

**"OPENROUTER_API_KEY not set"**
- Copy `.env.example` to `.env`
- Add your API key from https://openrouter.ai/

**"CUDA out of memory"**
- Lower batch size in `config/settings.py`
- Or use CPU (slower)

**GPU not detected**
- Check NVIDIA drivers
- Run `python -c "import torch; print(torch.cuda.is_available())"`

## Logging

All output is logged to `logs/` directory with timestamps.

Check for errors:
```bash
# View latest log
type logs\academicon_*.log
```

## Next Steps

1. **Copy `.env.example` → `.env`** and add your API key
2. **Run `index_academicon_lite.bat`** to create database
3. **Run `start_web_ui.bat`** to start web interface
4. **Ask questions** about your codebase!

## System Requirements

- Windows 10+ or WSL2 with Python 3.11+
- 8GB+ RAM
- GPU recommended (NVIDIA with CUDA 11.8+)
- Internet for OpenRouter API calls

## Architecture

- **Indexer**: Retrieves relevant code chunks
- **Orchestrator**: Plans search strategy
- **Graph Analyst**: (optional) Analyzes code relationships
- **Synthesizer**: Generates final answer

See `config/settings.py` for detailed configuration options.
