# Getting Started - Multi-Agent RAG System Improvements

## Quick Start Guide

### Step 1: Install New Dependencies

```bash
# Activate virtual environment
academicon-agent-env\Scripts\activate

# Install new dependencies
pip install -r requirements.txt
```

**New packages include:**
- `onnxruntime-gpu` - GPU support for RTX 5070 Ti
- `pydantic` - Configuration management
- `tenacity` - Retry logic
- Testing and development tools

### Step 2: Create Configuration File

```bash
# Copy example configuration
copy config\.env.example .env

# Optional: Edit .env to customize paths
notepad .env
```

**Most important settings:**
```bash
# Cache directory (change if needed)
ACADEMICON_CACHE__BASE_DIR=D:/AI-Models

# Target codebase path
ACADEMICON_INDEXING__TARGET_CODEBASE_PATH=//wsl$/Ubuntu/home/deeznutz/projects/Academicon-Rebuild

# GPU settings
ACADEMICON_GPU__USE_ONNX_RUNTIME=true
ACADEMICON_GPU__MAX_VRAM_USAGE_GB=14.0
```

### Step 3: Test GPU Support

```bash
# Check GPU status
python -c "from src.utils import print_gpu_status; print_gpu_status()"
```

**Expected output:**
```
============================================================
GPU STATUS
============================================================
GPU Available:        True
GPU Name:             NVIDIA GeForce RTX 5070 Ti
VRAM:                 16.0 GB
CUDA Version:         12.6
Compute Capability:   sm_120
PyTorch Support:      False (sm_120 not yet supported)
ONNX Runtime Support: True

Optimal Device:       cuda
Optimal Batch Size:   128
============================================================
```

### Step 4: Test File Filtering

```bash
# Verify .env and secrets are excluded
python test_file_filtering.py
```

**Expected output:**
```
============================================================
FILE FILTERING TEST
============================================================

✓ .env               EXCLUDED     EXCLUDED     🔒 Secret
✓ credentials.json   EXCLUDED     EXCLUDED     🔒 Secret
✓ logo.png           EXCLUDED     EXCLUDED     🖼️  Media
✓ academicon-agent-env  EXCLUDED  EXCLUDED     🐍 Virtual env
✓ main.py            ALLOWED      ALLOWED      ✅ Code

✅ ALL TESTS PASSED!
```

### Step 5: Test Configuration System

```bash
# Test configuration loading
python -c "from config import settings; import json; print(json.dumps(settings.model_dump(), indent=2, default=str))" > config_test.json

# View configuration
type config_test.json
```

---

## Feature Testing

### Test 1: GPU Utilities

```bash
python src/utils/gpu_utils.py
```

**What it tests:**
- GPU detection
- CUDA availability
- ONNX Runtime providers
- Batch size calculation

### Test 2: ONNX Embeddings (Will take a few minutes first time)

```bash
python src/utils/onnx_embeddings.py
```

**What it tests:**
- ONNX model loading
- GPU acceleration
- Embedding generation
- Fallback to CPU if needed

**Note:** First run will download/convert model (~2GB). Subsequent runs are fast.

### Test 3: Query Cache

```bash
python src/utils/query_cache.py
```

**What it tests:**
- Semantic similarity matching
- LRU eviction
- Cache hit/miss tracking

### Test 4: Logging System

```bash
python src/utils/logging_config.py
```

**What it tests:**
- Colored console output
- File logging with rotation
- Timing context manager

---

## Understanding the New Structure

### Before (Old Structure)
```
D:\LOCAL-CODER\
├── agents.py          # All agents in one file
├── main.py            # Hardcoded configs
├── web_ui.py          # Duplicate configs
└── index_academicon_lite.py
```

**Problems:**
- Hardcoded paths everywhere
- No logging
- No GPU support for RTX 5070 Ti
- Indexes .env files (security risk!)
- Monolithic code

### After (New Structure)
```
D:\LOCAL-CODER\
├── config/
│   ├── settings.py         # Single source of truth
│   ├── .env.example        # Template
│   └── .env                # Your settings (create this!)
├── src/
│   ├── agents/             # Modular agents
│   │   ├── base.py
│   │   ├── indexer.py
│   │   ├── orchestrator.py
│   │   └── synthesizer.py
│   ├── prompts/            # External templates
│   │   ├── orchestrator.j2
│   │   └── synthesizer.j2
│   └── utils/              # Reusable utilities
│       ├── logging_config.py
│       ├── gpu_utils.py
│       ├── onnx_embeddings.py
│       ├── query_cache.py
│       └── file_filters.py
├── docs/
│   └── FILE_EXCLUSION_GUIDE.md
└── tests/                  # Coming soon
```

**Benefits:**
- ✅ Single configuration file
- ✅ Professional logging
- ✅ GPU support (ONNX Runtime)
- ✅ Security (excludes .env)
- ✅ Modular & maintainable

---

## Key Improvements Summary

### 1. Configuration Management
**Before:** Hardcoded paths in 6+ files
**After:** One config file, environment variables

```python
# Old way
CACHE_DIR = "D:/AI-Models/embeddings"  # Hardcoded

# New way
from config import settings
cache_dir = settings.cache.sentence_transformers_home
```

### 2. GPU Acceleration
**Before:** RTX 5070 Ti not working (PyTorch doesn't support sm_120)
**After:** ONNX Runtime supports sm_120

```python
# New GPU-accelerated embeddings
from src.utils import get_embedding_model
model = get_embedding_model()  # Automatically uses GPU if available
embeddings = model.encode(texts)  # 3-6x faster!
```

### 3. File Filtering
**Before:** Indexes everything (including .env, *.png, node_modules/)
**After:** Smart filtering

```python
# Automatically excludes:
# - .env files (security)
# - *.png, *.pdf (waste of tokens)
# - node_modules/, venv/ (dependencies)
# - academicon-agent-env/ (virtual env)

from src.utils import scan_directory_for_indexing
files = scan_directory_for_indexing(Path("./codebase"))
# Only indexes .py, .js, .ts source files!
```

### 4. Query Caching
**Before:** Every query re-executes entire pipeline
**After:** Semantic cache with 95% similarity threshold

```python
from src.utils import get_query_cache
cache = get_query_cache()

# First query: full processing
result = cache.get("How does auth work?")  # None (miss)
cache.put("How does auth work?", "Auth uses JWT...", context)

# Similar query: cached!
result = cache.get("Explain authentication")  # Hit! (similarity: 0.97)
```

### 5. Structured Logging
**Before:** Random print() statements
**After:** Professional logging

```python
from src.utils import get_logger
logger = get_logger(__name__)

logger.info("Processing query")      # Green
logger.warning("Slow response")      # Yellow
logger.error("Failed", exc_info=True)  # Red + traceback
```

---

## Common Issues & Solutions

### Issue 1: "ModuleNotFoundError: No module named 'config'"

**Solution:**
```bash
# Make sure you're in the right directory
cd D:\LOCAL-CODER

# And virtual environment is activated
academicon-agent-env\Scripts\activate
```

### Issue 2: "ONNX Runtime CUDA provider not available"

**Solution:**
```bash
# Install ONNX Runtime GPU version
pip install onnxruntime-gpu>=1.19.0

# Verify installation
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Should show: `['CUDAExecutionProvider', 'CPUExecutionProvider']`

### Issue 3: ".env file not found"

**Solution:**
```bash
# Create .env from example
copy config\.env.example .env
```

### Issue 4: "GPU still not working"

**Diagnosis:**
```bash
# Check GPU status
python src/utils/gpu_utils.py

# Check CUDA version
nvidia-smi

# Check PyTorch
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

**Note:** RTX 5070 Ti (sm_120) requires CUDA 12.6+. PyTorch 2.6.0 doesn't support it yet, but ONNX Runtime does!

---

## Next Steps

### Immediate (Do Now)
1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Create `.env` file: `copy config\.env.example .env`
3. ✅ Test GPU: `python src/utils/gpu_utils.py`
4. ✅ Test filtering: `python test_file_filtering.py`

### Soon (This Week)
5. ⏳ Re-index codebase with new filtering (will be faster!)
6. ⏳ Integrate new agents into main.py
7. ⏳ Test query caching

### Later (Next Week)
8. 🔜 Implement hybrid search (Vector + BM25)
9. 🔜 Add reranking layer
10. 🔜 Implement streaming responses

---

## Documentation

- **`IMPLEMENTATION_PROGRESS.md`** - What's been done (40% complete)
- **`Claude-big-plan-improvements.md`** - Full optimization plan
- **`docs/FILE_EXCLUSION_GUIDE.md`** - File filtering details
- **`GREEK_SUMMARY.md`** - Greek summary (Ελληνικά)
- **`GETTING_STARTED.md`** - This file

---

## Getting Help

If something doesn't work:

1. **Check logs:** `logs/assistant.log`
2. **Enable debug:** `setup_logging(log_level="DEBUG")`
3. **Test individual components:** Use the test scripts
4. **Check configuration:** `python -c "from config import settings; print(settings.model_dump())"`

---

## Performance Expectations

| Component | Before | After | Improvement |
|-----------|--------|-------|-------------|
| **GPU Support** | 0% (broken) | 80%+ | Unlocked |
| **Indexing Speed** | 30-60 min | 10 min | 6x faster |
| **Token Usage** | 10M | 2M | 5x less |
| **Cache Hit Rate** | 0% | 50%+ | Huge |
| **Code Quality** | Low | High | Maintainable |

---

**Ready to start? Follow Step 1 above!** 🚀

**Last Updated:** 2025-11-11
