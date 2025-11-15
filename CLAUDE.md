# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Multi-Agent RAG (Retrieval-Augmented Generation) system designed to answer questions about the Academicon codebase. It uses 4 specialized AI agents orchestrated in a pipeline to retrieve, analyze, and synthesize information from a vectorized code database.

**ACTIVE VERSION**: OpenRouter (cloud-based models) - Fast and reliable ✅
**LEGACY VERSION**: Ollama (local models) - Deprecated ❌

## Multi-Agent Architecture

The system uses a coordinated pipeline of 4 agents:

### OpenRouter Version (ACTIVE) - agents_openrouter.py ⭐
Uses cloud APIs (fast and reliable):
1. **IndexerAgent**: Retrieves code chunks via vector search (ChromaDB + nomic-embed-text-v1.5, local GPU)
2. **GraphAnalystAgent**: Code analysis (openai/gpt-oss-120b) - Optional
3. **OrchestratorAgent**: Planning (z-ai/glm-4.5-air:free or deepseek)
4. **SynthesizerAgent**: Answer generation (z-ai/glm-4.5-air:free or gemini-2.0-flash)

**Performance**: 40-80s per query, 99% success rate ✅
**Cost**: FREE models available (glm-4.5, deepseek, gemini)
**Conversation Memory**: Maintains last 3 exchanges for follow-up questions

### Legacy Ollama Version - agents.py (DEPRECATED)
Uses local Ollama models (slow):
1. **IndexerAgent**: Same vector search
2. **GraphAnalystAgent**: Analyzes code (qwen2.5-coder:14b)
3. **OrchestratorAgent**: Plans (qwen2.5-coder:14b)
4. **SynthesizerAgent**: Generates answers (qwen2.5-coder:14b)

**Performance**: 250s+ per query, 30% success rate ❌
**Location**: `legacy/` folder for backwards compatibility

## File Structure

```
src/                                    # Active OpenRouter code
├── agents_openrouter.py                # 4-agent system
├── main_openrouter.py                  # CLI pipeline
├── web_ui_openrouter.py                # Gradio web UI
├── index_academicon_lite.py            # Indexing script
└── __init__.py

legacy/                                 # Deprecated Ollama code
├── agents.py
├── main.py
├── web_ui.py
├── README_LEGACY.md
└── __init__.py

config/                                 # Config modules (coming)
tests/                                  # Unit tests (coming)
docs/                                   # Documentation
```

## Key Commands

### Setup & Installation
```bash
# Create and activate virtual environment
python -m venv academicon-agent-env
academicon-agent-env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Indexing the Codebase
```bash
# Fast indexing (Python files only, 5-10 minutes)
python src/index_academicon_lite.py
```

### Running the Assistant

**OpenRouter Version (RECOMMENDED)** 🚀:
```bash
# Web UI - double-click or run:
start_web_ui_openrouter.bat

# CLI interface
python src/main_openrouter.py
```

### Setup OpenRouter (One-time)
```bash
# 1. Get API key from https://openrouter.ai/keys
# 2. Edit .env file:
OPENROUTER_API_KEY=sk-or-v1-YOUR-KEY-HERE

# 3. Configure models (all FREE):
ORCHESTRATOR_MODEL=z-ai/glm-4.5-air:free
SYNTHESIZER_MODEL=z-ai/glm-4.5-air:free
GRAPH_ANALYST_MODEL=openai/gpt-oss-120b
FALLBACK_MODEL=z-ai/glm-4.5-air:free

# 4. Enable free models at: https://openrouter.ai/settings/privacy
# Toggle ON: "Enable free endpoints that may train on inputs"
```

## Configuration & File Locations

**Environment Variables (.env)**:
- `OPENROUTER_API_KEY`: API key for cloud models *(required)*
- `ORCHESTRATOR_MODEL`: Model for planning (default: z-ai/glm-4.5-air:free)
- `SYNTHESIZER_MODEL`: Model for answers (default: z-ai/glm-4.5-air:free)
- `GRAPH_ANALYST_MODEL`: Model for code analysis (default: openai/gpt-oss-120b)
- `FALLBACK_MODEL`: Backup model if others fail (default: z-ai/glm-4.5-air:free)

**Cache Directories**: All AI models cache to D: drive to avoid C: drive space issues
- HF_HOME: `D:/AI-Models/huggingface-moved`
- TRANSFORMERS_CACHE: `D:/AI-Models/transformers`
- SENTENCE_TRANSFORMERS_HOME: `D:/AI-Models/embeddings`

**Vector Database**: `./academicon_chroma_db/` - ChromaDB with ~357 chunks indexed

**Target Codebase**: `//wsl$/Ubuntu/home/deeznutz/projects/Academicon-Rebuild`

## Code Structure

**OpenRouter (Active)**:
- `src/agents_openrouter.py`: 4 agents using OpenAILike client for OpenRouter API
- `src/main_openrouter.py`: Pipeline with conversation memory
- `src/web_ui_openrouter.py`: Gradio UI with metrics + export
- `src/index_academicon_lite.py`: Indexing with GPU support

**Legacy (Deprecated)**:
- `legacy/agents.py`: Ollama-based agents
- `legacy/main.py`: Ollama pipeline
- `legacy/web_ui.py`: Basic Gradio UI
- `legacy/README_LEGACY.md`: Deprecation notice

## Development Notes

**Agent Configuration**:
- LLM timeout: 60-90s (fast models)
- Temperature: 0.1 (Orchestrator, deterministic), 0.2 (Synthesizer)
- Retrieval: top_k=5 chunks per search query
- Embedding: 768-dimensional (nomic-embed-text-v1.5)
- GPU batch size: 128 (GPU) / 32 (CPU)

**Adding New Features**:
1. Create feature in `src/` directory
2. Add tests in `tests/` directory
3. Update docs in `docs/` directory
4. Update this file

**Re-indexing**: Delete `./academicon_chroma_db/` and run `python src/index_academicon_lite.py` when codebase changes

**Migration from Ollama**:
1. No changes needed - same vector DB works!
2. Just set OPENROUTER_API_KEY in .env
3. Run src/main_openrouter.py instead of main.py
4. No re-indexing required

## Performance Characteristics

**OpenRouter Version (Active)** ⚡:
- Query response time: 40-80 seconds total
  - Orchestration: 2-5s
  - Retrieval: 0.5-1s (local GPU)
  - Graph analysis: Disabled (optional)
  - Synthesis: 30-50s
- Success rate: 99%+
- Cost: FREE with glm-4.5-air:free
- Conversation memory: Last 3 exchanges maintained
- **16x faster than local Ollama!**

**Indexing** (GPU-accelerated):
- Time: 5-10 minutes (357 chunks)
- Hardware: RTX 5070 Ti (16GB VRAM)
- Batch size: 128 (GPU)

## System Requirements

**OpenRouter Version (Active)**:
- Python 3.11+
- OpenRouter API key (free tier available)
- Internet connection for API calls
- 16GB RAM (embeddings run locally on GPU)
- NVIDIA GPU recommended (but not required)

**Legacy Ollama Version**:
- Python 3.11+
- Ollama v0.12.6+ running locally
- qwen2.5-coder:14b model (~9GB)
- 40-64GB RAM
- 50-100GB disk space

## Roadmap (Planned)

- ✅ OpenRouter integration (complete)
- ⏳ config.py module (centralized config)
- ⏳ utils.py module (shared functions)
- ⏳ Structured logging (replace print statements)
- ⏳ Unit tests for all agents
- ⏳ CI/CD pipeline

## Testing

**Coming soon**: Unit tests in `tests/` directory

For now, test manually:
```bash
python src/main_openrouter.py
# Type your question and verify answer quality
```
