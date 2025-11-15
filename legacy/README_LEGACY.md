# LEGACY CODE - Ollama Local Models

This folder contains the legacy Ollama implementation of the Academicon Code Assistant.

## Status
**DEPRECATED** - Use the OpenRouter version (`src/` folder) for production use.

## Performance Comparison
- **OpenRouter (Active)**: 40-80s per query, 99% success rate ✅
- **Local Ollama (Legacy)**: 250s+ per query, 30% success rate ❌

## Files
- `agents.py` - 4-agent system using Ollama models
- `main.py` - CLI pipeline with Ollama
- `web_ui.py` - Gradio web UI with Ollama

## Requirements
- Ollama running locally (`http://localhost:11434`)
- `qwen2.5-coder:14b` model pulled
- 40-64GB RAM
- 50-100GB disk space

## Why Legacy?
The local Ollama implementation was replaced because:
1. **Slow**: 250+ seconds per query
2. **Unreliable**: ~30% success rate with frequent timeouts
3. **Resource intensive**: Requires 40-64GB RAM
4. **Complex setup**: Requires local model installation

## Migration Path
To switch from Ollama to OpenRouter:
1. Ensure you have OpenRouter API key
2. Set it in `.env` file
3. Use `main_openrouter.py` instead of `main.py`
4. Use `web_ui_openrouter.py` instead of `web_ui.py`

No changes needed to vector database - it's compatible!

## Archive
This code is preserved for reference and compatibility testing.
Do not use for new development.
