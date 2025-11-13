# Practical Guide to Building a Local, Code-Aware AI Assistant (Late 2025)

## Part 1: Choosing the Right Local LLM for Code

Top open-source LLMs optimized for code include Qwen2.5-Coder, DeepSeek-Coder-V3, and GLM-4.6, successors to DeepSeek-Coder-V2 and Phind-CodeLlama. These excel in code generation, understanding, and multi-language support (Python, JS, etc.).

### Top Model Recommendations

- **Qwen2.5-Coder**: Alibaba's SOTA code model. Variations: 1.5B, 7B, 32B. Strengths: Superior in coding tasks, agentic reasoning.
- **DeepSeek-Coder-V3**: DeepSeek's advanced iteration. Variations: 7B, 34B, 236B (focus on smaller for local). Strengths: Affordable, flexible for code navigation/refactoring.
- **GLM-4.6**: Strong in coding, outperforming predecessors. Variations: 9B, 130B. Strengths: 200K context, multi-language code.

| Model Name       | Parameters | Recommended VRAM (Q4 Quant) | Key Strengths                  | Link to Hugging Face                  |
|------------------|------------|-----------------------------|--------------------------------|---------------------------------------|
| Qwen2.5-Coder   | 7B        | 6-8GB                      | Python/JS, code gen/understanding | https://huggingface.co/Qwen/Qwen2.5-Coder-7B |
| DeepSeek-Coder-V3 | 7B       | 6-8GB                      | Multi-lang, refactoring        | https://huggingface.co/deepseek-ai/DeepSeek-Coder-V3-Base |
| GLM-4.6         | 9B        | 8-10GB                     | Coding reasoning, long context | https://huggingface.co/THUDM/glm-4-9b-chat |

### Quantization

Model quantization reduces precision (e.g., from FP16 to INT4) to lower VRAM use and speed inference, crucial for local runs on consumer hardware like RTX 5070Ti (16GB VRAM).

GGUF is the most popular and well-supported for code models in 2025, due to flexibility with llama.cpp, CPU/GPU offload, and broad compatibility. AWQ excels for activation-aware GPU inference; GPTQ for one-shot GPU quantization.

Choose quantization level by trade-off: Q4_K_M (balanced, ~4-6GB for 7B models, minor quality loss); Q8_0 (higher quality, ~8-10GB, better for complex code tasks). Test on hardware; start with Q4 for speed, upscale if accuracy drops.

## Part 2: Frameworks for Running Local LLMs

Leading frameworks: Ollama (easy CLI/API), vLLM (high-throughput inference), LM Studio (GUI-focused), Text Generation WebUI (Oobabooga, flexible customization).

| Framework              | Ease of Use | Performance (Tokens/sec) | API Compatibility | Key Features                          |
|------------------------|-------------|---------------------------|-------------------|---------------------------------------|
| Ollama                | High       | 50-100 (mid-range GPU)   | OpenAI-like      | Model library, quick setup, API server |
| vLLM                  | Medium     | 200-800 (high-scale)     | OpenAI-like      | PagedAttention, quantization, multi-GPU |
| LM Studio             | High       | 40-80                    | Partial          | GUI, model discovery, easy downloads  |
| Text Generation WebUI | Medium     | 60-120                   | OpenAI-like      | Custom UIs, extensions, fine-tune support |

For quick setup with robust OpenAI-compatible API: Ollama, due to one-line installs, developer-friendly CLI, and consistent performance.

## Part 3: "Training" the LLM on a Private Codebase (The RAG Pipeline)

Use RAG to augment LLM with codebase retrieval, avoiding full retraining.

### The RAG Workflow Explained

1. **Loading & Chunking**: Load repo files (.py, .jsx, .md) via directory readers. Chunk by function/class for code (preserves semantics); use recursive character splitting with overlap (20%) for text. Best practices: Semantic chunking (via LLMs for topics), fixed-size (400-800 tokens), or page-level for docs.

2. **Embedding**: Convert chunks to vectors for similarity search. Top open-source: Nomic-embed-code-v1 (code-specific, 768D), Voyage-code-3 (retrieval excellence), Jina-embeddings-v3 (multi-lang code).

3. **Vector Storage**: Vector DB stores embeddings for fast querying. Recommend: ChromaDB (easiest local setup, Python-native), FAISS (fastest search), LanceDB (scalable for large datasets). ChromaDB for beginners—pip install, auto-persist.

4. **Retrieval & Augmentation**: Embed user query, retrieve top-k chunks (cosine similarity), inject into LLM prompt (e.g., "Use this code: [chunks] to answer: [query]").

### Key Open-Source Libraries for Building the RAG Pipeline

LlamaIndex (data-focused, easy ingestion) and LangChain (flexible chains, integrations). Recommend LlamaIndex for simplicity in code RAG.

Example using LlamaIndex:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load embedding model
embed_model = HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v1")

# Load documents
documents = SimpleDirectoryReader('./my_codebase', recursive=True).load_data()

# Build index with custom embeddings
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("Explain the 'CIP' service in my codebase.")
print(response)
```

## Part 4: Fine-Tuning vs. RAG - A Practical Distinction

### RAG (Retrieval-Augmented Generation)

Analogy: Open-book exam—LLM pulls codebase info at runtime.

Pros: Fast setup, low cost, updates with code changes, no forgetting.

Cons: Retrieval quality varies; limited by context window.

Recommendation: Start here for 95% of cases; ideal for code knowledge.

### Fine-Tuning

Analogy: Teaching codebase style/patterns permanently.

Pros: Learns custom styles, architectures.

Cons: GPU-intensive, needs datasets, overfits, outdated quickly.

Recommendation: Use after RAG if needed for behavior (not just facts). Employ LoRA for efficient tuning on consumer hardware.

## Part 5: Putting It All Together - A Recommended "Starter Stack"

### Your First Local Code-Aware Assistant: A Recipe

- **Model**: Qwen2.5-Coder-7B-GGUF (balanced code performance, runs on 8GB VRAM).
- **Serving Framework**: Ollama (easy setup, OpenAI API).
- **RAG Library**: LlamaIndex (simple ingestion/retrieval).
- **Vector Database**: ChromaDB (local, easy Python integration).
- **Embedding Model**: Nomic-embed-text-v1 (strong code embeddings).

Steps: Install Ollama, pull model; set up LlamaIndex with ChromaDB; index codebase; query via Ollama API with RAG-augmented prompts. Test on hardware like Ryzen 9/RTX 5070Ti for 50+ tokens/sec.