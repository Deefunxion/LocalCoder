import os
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
import torch
import time

# Set cache directories to D: drive - FORCE all caches to D:
os.environ['HF_HOME'] = 'D:/AI-Models/huggingface-moved'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:/AI-Models/huggingface-moved/hub'
os.environ['TRANSFORMERS_CACHE'] = 'D:/AI-Models/transformers'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = 'D:/AI-Models/embeddings'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Disable symlink warnings

# Configuration - LITE VERSION (Python only, larger chunks)
ACADEMICON_PATH = "//wsl$/Ubuntu/home/deeznutz/projects/Academicon-Rebuild"
ALLOWED_EXTENSIONS = [".py"]  # Python only for faster indexing
DB_PATH = "./academicon_chroma_db"

print("="*60)
print("Academicon Codebase Indexing - LITE VERSION")
print("="*60)
print("[INFO] Python files only, larger chunks for speed")
print()

# Step 1: Initialize embedding model with GPU support
print("[1/5] Loading embedding model (Nomic Embed)...")

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128 if device == "cuda" else 32

if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"   [GPU ENABLED] {gpu_name} ({gpu_memory:.1f} GB VRAM)")
    print(f"   [INFO] PyTorch {torch.__version__} with CUDA {torch.version.cuda}")
    print(f"   [INFO] sm_120 support: {'sm_120' in torch.cuda.get_arch_list()}")
else:
    print("   [WARNING] No GPU detected, using CPU (slower)")

# Initialize with sentence-transformers directly for better GPU control
embed_model = HuggingFaceEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    cache_folder="D:/AI-Models/embeddings",
    trust_remote_code=True,
    embed_batch_size=batch_size,
    device=device  # GPU acceleration
)
Settings.embed_model = embed_model
print(f"   [OK] Embedding model loaded (768 dimensions)")
print(f"   [OK] Device: {device.upper()} | Batch size: {batch_size}")

# Step 2: Load documents
print("\n[2/5] Loading Python files from Academicon...")
print(f"   Source: {ACADEMICON_PATH}")
print(f"   Extensions: {', '.join(ALLOWED_EXTENSIONS)}")

start_time = time.time()

documents = SimpleDirectoryReader(
    input_dir=ACADEMICON_PATH,
    recursive=True,
    required_exts=ALLOWED_EXTENSIONS,
    exclude_hidden=True,
    exclude=["node_modules", ".git", "dist", "build", "__pycache__", "venv", ".venv", "migrations"]
).load_data()

load_time = time.time() - start_time
print(f"   [OK] Loaded {len(documents)} Python files in {load_time:.2f}s")

# Step 3: Split into LARGER chunks (fewer embeddings needed)
print("\n[3/5] Splitting code into chunks...")
start_time = time.time()

splitter = SentenceSplitter(
    chunk_size=2048,        # LARGER chunks (was 1024)
    chunk_overlap=256,      # Smaller overlap
)

nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

split_time = time.time() - start_time
print(f"   [OK] Created {len(nodes)} chunks in {split_time:.2f}s")
print(f"   [INFO] Estimated embedding time: ~{len(nodes) * 0.5 / 60:.1f} minutes")

# Step 4: Setup ChromaDB
print("\n[4/5] Setting up ChromaDB vector store...")
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Delete existing collection if it exists
try:
    chroma_client.delete_collection("academicon_code")
    print("   [INFO] Deleted existing collection")
except:
    pass

chroma_collection = chroma_client.create_collection("academicon_code")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
print(f"   [OK] ChromaDB initialized at {DB_PATH}")

# Step 5: Build index
print("\n[5/5] Building vector index...")
print(f"   [INFO] Processing {len(nodes)} chunks...")
start_time = time.time()

index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    show_progress=True
)

index_time = time.time() - start_time

# Final summary
print("\n" + "="*60)
print("Indexing Complete!")
print("="*60)
print(f"Total Python files:  {len(documents)}")
print(f"Total chunks:        {len(nodes)}")
print(f"Database path:       {DB_PATH}")
print(f"Total time:          {index_time/60:.2f} minutes")
print(f"Avg per chunk:       {index_time/len(nodes):.3f}s")
print("="*60)
print("\n[OK] Vector database ready for queries!")
print("\nNext step: Run 'python main.py' to test the assistant")
