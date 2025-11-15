# LocalCoder 🤖

Multi-Agent RAG System for Academicon Codebase Analysis

![Status](https://img.shields.io/badge/status-active-success)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

LocalCoder is an intelligent code assistant that answers questions about the Academicon codebase using a multi-agent RAG (Retrieval-Augmented Generation) architecture. It combines vector search, LLM reasoning, and conversation memory to provide accurate, context-aware answers.

**Key Features:**
- 🚀 **Fast**: 40-80s queries with OpenRouter (16x faster than local models)
- 🧠 **Smart**: Maintains conversation history for follow-up questions
- 💰 **Free**: Uses free OpenRouter models (glm-4.5, deepseek, gemini)
- 🎯 **Accurate**: Multi-agent pipeline with specialized roles
- ⚡ **GPU Accelerated**: Local embeddings run on NVIDIA GPU

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup OpenRouter (Recommended)
```bash
# Get API key from https://openrouter.ai/keys
# Edit .env file:
OPENROUTER_API_KEY=sk-or-v1-YOUR-KEY-HERE

# Configure free models:
ORCHESTRATOR_MODEL=z-ai/glm-4.5-air:free
SYNTHESIZER_MODEL=z-ai/glm-4.5-air:free
```

### 3. Index the Codebase (One-time)
```bash
python src/index_academicon_lite.py
# Takes 5-10 minutes, creates vector database
```

### 4. Run the Assistant
```bash
# Web UI (Recommended)
start_web_ui_openrouter.bat

# Or manually:
python src/web_ui_openrouter.py
```

Open: http://127.0.0.1:7860

## Architecture

```
User Query
    ↓
1. Orchestrator (GLM-4.5) → Plans search strategy
    ↓
2. Indexer (Local GPU) → Retrieves relevant code chunks
    ↓
3. Graph Analyst (Optional) → Analyzes code relationships
    ↓
4. Synthesizer (GLM-4.5) → Generates final answer with conversation context
```

**Performance:**
- OpenRouter: 40-80s per query, 99% success rate ✅
- Local Ollama: 250s+ per query, 30% success rate ❌

## Features

### Conversation Memory 🧠
Maintains last 3 exchanges for context-aware follow-up questions:
```
Q1: What is the CIP service?
A1: CIP handles citation processing...

Q2: How does it integrate with the database?
A2: [Uses context from Q1] CIP connects to citation_db...

Q3: Show me the code
A3: [Knows you mean CIP] Here's the implementation...
```

### Multi-Model Support 🎯
Different models for different tasks:
- **Orchestrator**: `z-ai/glm-4.5-air:free` (fast planning)
- **Synthesizer**: `z-ai/glm-4.5-air:free` (quality answers)
- **Graph Analyst**: `openai/gpt-oss-120b` (code analysis)

All models are **FREE** via OpenRouter!

### GPU Acceleration ⚡
- Embeddings: nomic-embed-text-v1.5 on CUDA
- Hardware: NVIDIA RTX 5070 Ti (16GB VRAM)
- Speedup: 3-6x faster indexing

## Documentation

- **[docs/OPENROUTER_SETUP.md](docs/OPENROUTER_SETUP.md)**: Complete OpenRouter setup guide
- **[docs/CLAUDE.md](docs/CLAUDE.md)**: Technical documentation for AI assistants
- **[docs/HOW_TO_USE.md](docs/HOW_TO_USE.md)**: Usage guide
- **[AGENTS.md](AGENTS.md)**: Agent design guidelines

## Project Structure

```
LocalCoder/
├── src/
│   ├── agents_openrouter.py       # OpenRouter agents (recommended)
│   ├── main_openrouter.py         # Pipeline with conversation memory
│   ├── web_ui_openrouter.py       # Gradio web interface
│   ├── index_academicon_lite.py   # Codebase indexing
│   └── __init__.py
├── legacy/                         # Deprecated Ollama implementation
│   ├── agents.py
│   ├── main.py
│   ├── web_ui.py
│   └── README_LEGACY.md
├── docs/                           # Documentation
│   ├── README.md
│   ├── CLAUDE.md
│   └── OPENROUTER_SETUP.md
├── config/                         # Configuration (to be created)
├── tests/                          # Unit tests (to be created)
├── .env                            # Configuration (API keys, models)
├── .env.example                    # Configuration template
├── requirements.txt                # Python dependencies
├── academicon_chroma_db/           # Vector database
└── start_web_ui_openrouter.bat     # Quick launcher
```

## Performance Comparison

| Metric | OpenRouter | Local Ollama |
|--------|-----------|--------------|
| Query Time | 40-80s | 250s+ |
| Success Rate | 99% | 30% |
| Cost | FREE | FREE |
| RAM | 16GB | 64GB |
| Internet | Required | Not required |
| **Verdict** | ✅ **Recommended** | ❌ Legacy |

## Requirements

- Python 3.11+
- OpenRouter API key (free tier available)
- 16GB+ RAM
- NVIDIA GPU (optional but recommended)
- Internet connection for API calls

## Troubleshooting

### "WARN: Planning failed (404)"
Enable free models at: https://openrouter.ai/settings/privacy
Toggle ON: "Enable free endpoints that may train on inputs"

### Slow Queries
Check model selection in `.env`:
```bash
# Use fastest free models:
ORCHESTRATOR_MODEL=z-ai/glm-4.5-air:free
SYNTHESIZER_MODEL=z-ai/glm-4.5-air:free
```

### No GPU Detected
Embeddings will run on CPU (slower but functional):
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Contributing

This is a personal project for Academicon codebase analysis. Feel free to fork and adapt for your own codebases!

## License

MIT License

## Credits

Built with:
- [LlamaIndex](https://www.llamaindex.ai/) - RAG framework
- [OpenRouter](https://openrouter.ai/) - LLM API aggregator
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Gradio](https://gradio.app/) - Web UI framework

---

**Status:** Active development | **Last Updated:** November 13, 2025
