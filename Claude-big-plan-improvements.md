Multi-Agent RAG System: Comprehensive Optimization Plan

 Phase 1: Foundation & Infrastructure (Days 1-3)

 1.1 Configuration Management

 - Create config/ directory with settings.py using Pydantic
 - Centralize all hardcoded paths (D: drive cache, DB paths, Ollama URLs)
 - Add environment variable support (.env file)
 - Move model names, temperatures, timeouts to config

 1.2 Project Restructuring

 - Reorganize codebase:
 src/
   ├── agents/         # Split agents.py into separate files
   │   ├── base.py
   │   ├── indexer.py
   │   ├── orchestrator.py
   │   └── synthesizer.py
   ├── prompts/        # Externalized prompt templates
   ├── retrieval/      # Indexing & search logic
   ├── utils/          # Shared utilities
   └── config/         # Configuration files
 - Update all imports across codebase

 1.3 Logging & Monitoring

 - Replace all print() with structured logging
 - Add performance metrics tracking (latency per agent)
 - Create logs/ directory with rotation

 ---
 Phase 2: GPU Acceleration (Days 4-6) 🚀

 2.1 ONNX Runtime GPU Support

 - Install ONNX Runtime with CUDA 12.6+ for RTX 5070 Ti (sm_120 support)
 - Convert nomic-embed-text-v1.5 to ONNX format
 - Implement GPU embedding fallback chain: ONNX GPU → PyTorch GPU → CPU
 - Test batch sizes: 128, 256, 512 to find optimal throughput

 2.2 Dynamic Batch Sizing

 - Implement VRAM-aware batch size calculator
 - Add automatic scaling based on available memory
 - Create GPU memory monitoring utilities

 2.3 Model Loading Optimization

 - Pre-warm embedding model on startup (web_ui.py)
 - Implement singleton pattern for embedding model
 - Add CUDA_VISIBLE_DEVICES coordination with Ollama
 - Optimize VRAM allocation (pin embedding model, leave space for LLM)

 ---
 Phase 3: Retrieval Quality Enhancement (Days 7-10)

 3.1 Hybrid Search Implementation

 - Add BM25 indexing alongside vector search
 - Implement weighted fusion (0.7 vector + 0.3 BM25)
 - Create hybrid search class in retrieval/hybrid_search.py

 3.2 Reranking Layer

 - Add cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
 - Change retrieval flow: retrieve top-10 → rerank → select top-3
 - Benchmark precision improvement with evaluation dataset

 3.3 Hierarchical Chunking

 - Implement multi-level chunking strategy:
   - Level 1: Function-level (256-512 tokens)
   - Level 2: Class-level (1024 tokens)
   - Level 3: File summaries (2048 tokens)
 - Update indexing scripts to generate all three levels
 - Add chunk-level metadata (type, scope, complexity)

 3.4 Rich Metadata Extraction

 - Parse Python AST to extract:
   - Function/class names
   - Import statements
   - Docstrings
   - Decorator information
 - Store metadata in ChromaDB for filtering
 - Enable metadata-aware retrieval

 3.5 Upgrade Embedding Model

 - Test nomic-embed-text-v2.0 vs v1.5
 - Benchmark retrieval quality on evaluation set
 - Migrate if improvement ≥10%
 - Re-index database with new embeddings

 ---
 Phase 4: Performance & Caching (Days 11-13)

 4.1 Query Response Cache

 - Implement semantic similarity cache (threshold: 0.95)
 - Use LRU eviction policy (cache size: 100 queries)
 - Store embeddings + full responses
 - Add cache hit/miss metrics

 4.2 Streaming Responses

 - Modify all agents to use Ollama streaming API
 - Implement Server-Sent Events (SSE) in web_ui.py
 - Show real-time progress (Orchestrating → Retrieving → Analyzing → Synthesizing)
 - Update CLI to show streaming output

 4.3 Conversation Context

 - Add sliding window context (last 5 exchanges)
 - Store in session state (web UI) and global state (CLI)
 - Pass context to Orchestrator for query refinement
 - Implement context pruning to stay within token limits

 4.4 Query Classification & Routing

 - Build query classifier (simple/complex/follow-up)
 - Route simple queries directly to retrieval + synthesis
 - Skip orchestration for follow-up questions (use context)
 - Only use full pipeline for complex architectural queries

 ---
 Phase 5: Code Quality & Testing (Days 14-16)

 5.1 Agent Refactoring

 - Split monolithic agents.py into modular classes
 - Create abstract base agent with common interface
 - Externalize all prompts to prompts/ templates (Jinja2)
 - Implement Pydantic models for agent outputs

 5.2 Testing Framework

 - Set up pytest with fixtures
 - Create unit tests for each agent (mock Ollama responses)
 - Add integration tests for full pipeline
 - Build evaluation dataset (50 query-answer pairs)
 - Add performance regression benchmarks

 5.3 Error Handling & Resilience

 - Implement retry logic with exponential backoff (Tenacity)
 - Add circuit breaker for Ollama failures
 - Create fallback strategies (keyword search if vector fails)
 - Add graceful degradation options

 ---
 Phase 6: Advanced Optimizations (Days 17-20)

 6.1 Vector Search Tuning

 - Configure ChromaDB HNSW parameters (ef_construction=200, M=32)
 - Implement vector quantization (float32 → int8)
 - Test matryoshka dimension reduction (768 → 512 → 384)
 - Benchmark retrieval speed vs quality tradeoffs

 6.2 Incremental Indexing Automation

 - Add file system watcher (watchdog library)
 - Implement auto-reindexing on file changes
 - Create incremental update queue with debouncing
 - Add webhook support for CI/CD integration

 6.3 Model Configuration Optimization

 - Pin Ollama models to specific quantization levels
 - Test qwen2.5-coder variants: q4_K_M vs q5_K_M vs q8_0
 - Tune LLM parameters (top_p=0.95, repetition_penalty=1.1)
 - Consider model specialization (7B Orchestrator, 32B Synthesizer)

 6.4 Monitoring & Observability

 - Add comprehensive metrics dashboard
 - Track: query latency (p50/p95/p99), success rate, cache hit rate
 - Implement agent-level performance profiling
 - Create alerting for anomalies (slow queries, failures)

 ---
 Phase 7: Testing & Documentation (Days 21-22)

 7.1 End-to-End Testing

 - Run full regression test suite
 - Benchmark performance improvements vs baseline
 - Validate answer quality on evaluation dataset
 - Load testing (concurrent queries)

 7.2 Documentation Updates

 - Update CLAUDE.md with new architecture
 - Create API documentation for agents
 - Document configuration options
 - Add troubleshooting guide

 ---
 Expected Outcomes

 | Metric               | Before      | After    | Improvement |
 |----------------------|-------------|----------|-------------|
 | Query Latency        | 30-90s      | 10-25s   | 3x faster   |
 | Indexing Time        | 30-60 min   | 5-10 min | 6x faster   |
 | Retrieval Precision  | ~70%        | ~90%     | +30%        |
 | Cache Hit Rate       | 0%          | 50%+     | Huge        |
 | GPU Utilization      | 0% (broken) | 80%+     | Unlocked    |
 | Code Maintainability | Low         | High     | 3x easier   |

 ---
 Files to Create/Modify

 New Files (~25):
 - config/settings.py, config/.env.example
 - src/agents/{base,indexer,orchestrator,synthesizer}.py
 - src/retrieval/{hybrid_search,reranker,chunking}.py
 - src/utils/{cache,gpu,logging}.py
 - src/prompts/*.j2 (template files)
 - tests/unit/test_*.py, tests/integration/test_pipeline.py
 - evaluation/dataset.json, evaluation/benchmark.py

 Modified Files (~10):
 - main.py, web_ui.py, agents.py (refactored)
 - index_academicon_lite.py, index_academicon.py
 - incremental_update.py
 - All batch files (update paths)
 - requirements.txt (new dependencies)