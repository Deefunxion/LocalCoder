# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Multi-Agent RAG (Retrieval-Augmented Generation) system designed to answer questions about the Academicon codebase. It uses 4 specialized AI agents orchestrated in a pipeline to retrieve, analyze, and synthesize information from a vectorized code database.

**NEW**: Now supports both local Ollama models AND cloud-based models via OpenRouter API for faster, more reliable responses.

## Multi-Agent Architecture

The system uses a coordinated pipeline of 4 agents:

### Local Version (agents.py)
Uses local Ollama models (slow but free):
1. **IndexerAgent**: Retrieves code chunks via vector search (ChromaDB + nomic-embed-text-v1.5)
2. **GraphAnalystAgent**: Analyzes code relationships (qwen2.5-coder:14b)
3. **OrchestratorAgent**: Plans search strategies (qwen2.5-coder:14b)
4. **SynthesizerAgent**: Generates final answers (qwen2.5-coder:14b)

**Performance**: 250s+ per query, frequent timeouts ❌

### OpenRouter Version (agents_openrouter.py) ⭐ RECOMMENDED
Uses cloud APIs (fast and reliable):
1. **IndexerAgent**: Same vector search (local, GPU-accelerated)
2. **GraphAnalystAgent**: Code analysis (openai/gpt-oss-120b) - Optional
3. **OrchestratorAgent**: Planning (z-ai/glm-4.5-air:free)
4. **SynthesizerAgent**: Answer generation (z-ai/glm-4.5-air:free)

**Performance**: 40-80s per query, reliable ✅
**Cost**: FREE models available (glm-4.5, deepseek, gemini)
**Conversation Memory**: Maintains last 3 exchanges for follow-up questions

## Key Commands

### Setup & Installation
```bash
# Create and activate virtual environment
python -m venv academicon-agent-env
academicon-agent-env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Pull required Ollama model
ollama pull qwen2.5-coder:14b
```

### Indexing the Codebase
```bash
# Initial indexing (5-10 minutes, Python files only)
python index_academicon_lite.py

# Full indexing (30-60 minutes, all code files) - if needed
python index_academicon.py
```

### Running the Assistant

**OpenRouter Version (RECOMMENDED)** 🚀:
```bash
# Web UI - double-click or run:
start_web_ui_openrouter.bat

# CLI interface
python main_openrouter.py
```

**Local Ollama Version** (slow but free):
```bash
# Web UI
start_web_ui.bat

# CLI interface
python main.py
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

### Testing Individual Agents
```bash
# Test specific agent
run_test.bat [1|2|3|4]

# Test all agents
run_test.bat all

# Individual test files
python testing/test_agent1.py  # Embedding model
python testing/test_agent2.py  # Graph analyst
python testing/test_agent3.py  # Orchestrator
python testing/test_agent4.py  # Synthesizer
```

## Configuration & File Locations

**Environment Variables (.env)**:
- `OPENROUTER_API_KEY`: API key for cloud models
- `ORCHESTRATOR_MODEL`: Model for planning (default: z-ai/glm-4.5-air:free)
- `SYNTHESIZER_MODEL`: Model for answers (default: z-ai/glm-4.5-air:free)
- `GRAPH_ANALYST_MODEL`: Model for code analysis (default: openai/gpt-oss-120b)
- `FALLBACK_MODEL`: Backup model if others fail (default: z-ai/glm-4.5-air:free)

**Cache Directories**: All AI models cache to D: drive to avoid C: drive space issues
- HF_HOME: `D:/AI-Models/huggingface-moved`
- TRANSFORMERS_CACHE: `D:/AI-Models/transformers`
- SENTENCE_TRANSFORMERS_HOME: `D:/AI-Models/embeddings`

**Vector Database**: `./academicon_chroma_db/` - ChromaDB with 357 chunks, 229 documents indexed

**Target Codebase**: `//wsl$/Ubuntu/home/deeznutz/projects/Academicon-Rebuild`

## Code Structure

**Core Files - OpenRouter (Recommended)**:
- `agents_openrouter.py`: All 4 agents using OpenAILike client for OpenRouter API
- `main_openrouter.py`: Pipeline with conversation memory + OpenRouter integration
- `web_ui_openrouter.py`: Gradio UI with conversation history, auto-port detection
- `start_web_ui_openrouter.bat`: Quick launcher for OpenRouter version
- `.env`: Configuration for API keys and model selection
- `OPENROUTER_SETUP.md`: Complete setup guide

**Core Files - Local Ollama (Legacy)**:
- `agents.py`: All 4 agent class definitions with Ollama LLM integration
- `main.py`: Main pipeline orchestration + CLI interface
- `web_ui.py`: Gradio web interface (localhost:7860)
- `start_web_ui.bat`: Quick launcher for local version

**Indexing**:
- `index_academicon_lite.py`: Fast indexing (Python only, 357 chunks)
- `index_academicon.py`: Full indexing (all code files, smaller chunks)

**Indexing Configuration** (index_academicon_lite.py:16-21):
- Chunk size: 2048 tokens with 256 token overlap
- File types: `.py` only for lite version
- Excludes: node_modules, .git, dist, build, __pycache__, venv, migrations

**Agent Configuration**:
- LLM timeout: 300s (5 minutes) for all qwen2.5-coder agents
- Temperature: 0.1 (Graph, Orchestrator), 0.7 (Synthesizer)
- Retrieval: top_k=5 chunks per search query
- Embedding: 768-dimensional vectors (nomic-embed-text-v1.5)
- GPU batch size: 128 (GPU) / 32 (CPU)

**Available Local Models** (check with `ollama list`):
- qwen2.5-coder:14b (9GB) - Main coding LLM
- qwen2.5:14b-instruct-q4_K_M (9GB) - Alternative LLM
- phi3.5:3.8b-mini-instruct (1.4-2.4GB) - Lightweight option
- mxbai-embed-large (669MB) - Alternative embedding model
- snowflake-arctic-embed:xs (45MB) - Tiny embedding model
- krikri-optimized (5.9GB) - Greek language support

## Development Notes

**Adding New Agents**: Follow the pattern in agents.py with Ollama LLM initialization and structured JSON response parsing (see lines 86-102 for JSON extraction from markdown blocks)

**Modifying Retrieval**: Adjust top_k parameter in main.py:106 or modify deduplication logic (main.py:109-116)

**Changing Models**: Update model names in agent __init__ methods and ensure Ollama has the model pulled

**Re-indexing**: Delete `./academicon_chroma_db/` directory and run indexing script again when codebase changes

## Performance Characteristics

**OpenRouter Version (Recommended)** ⚡:
- Query response time: 40-80 seconds total
  - Orchestration: 10-20s (GLM-4.5)
  - Retrieval: 0.5-1s (local GPU)
  - Graph analysis: Disabled for speed
  - Synthesis: 30-40s (GLM-4.5)
- Success rate: 99%+
- Cost: FREE with glm-4.5-air:free model
- Conversation memory: Last 3 exchanges maintained
- **16x faster than local Ollama!**

**Local Ollama Version** (Legacy):
- Query response time: 250+ seconds with frequent timeouts
  - Orchestration: 70s
  - Retrieval: 1-2s
  - Graph analysis: 80s (disabled)
  - Synthesis: 180s+ (often timeout)
- Success rate: ~30%
- Cost: Free but unreliable
- Requires: 40-64GB RAM

**Indexing** (same for both):
- Time: 5-10 minutes (357 chunks)
- GPU accelerated: RTX 5070 Ti (16GB VRAM)
- First query slower due to model loading

## System Requirements

**OpenRouter Version**:
- Python 3.11+
- OpenRouter API key (free tier available)
- Internet connection for API calls
- 16GB RAM (embeddings run locally on GPU)
- NVIDIA GPU recommended for embedding acceleration

**Local Ollama Version**:
- Python 3.11+
- Ollama v0.12.6+ running locally (http://localhost:11434)
- qwen2.5-coder:14b model pulled in Ollama
- 40-64GB RAM available
- 50-100GB disk space (models + cache + database)

## GPU Acceleration

**Current Status**: GPU code is implemented but **RTX 5070 Ti (sm_120) is not yet supported** by PyTorch 2.6.0. The system automatically falls back to CPU.

**Embedding Model GPU Support**: All indexing scripts automatically detect and use GPU when supported:
- `index_academicon_lite.py`: Auto-detects GPU, increases batch size from 32→128
- `main.py`: Loads embedding model on GPU for faster queries
- `testing/test_agent1.py`: Tests GPU detection

**Performance Impact**:
- CPU indexing: 30-60 minutes (current)
- GPU indexing: 5-10 minutes (3-6x faster - when supported)
- Batch size: 32 (CPU) vs 128 (GPU)

**RTX 5070 Ti Issue**:
- GPU architecture: Blackwell sm_120 (very new)
- PyTorch 2.6.0 supports up to sm_90
- **Solution**: Wait for PyTorch 2.7+ or use nightly build when Windows Long Path issue is resolved
- Code is already GPU-ready and will work automatically when PyTorch adds support

**Check GPU Status**:
```bash
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```
