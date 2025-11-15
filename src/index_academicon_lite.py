import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
import torch
import time

from config import Config, get_device, get_batch_size
from utils import setup_logger

# Setup environment
Config.setup_environment()

# Setup logging
logger = setup_logger(__name__)

logger.info("=" * 60)
logger.info("Academicon Codebase Indexing - LITE VERSION")
logger.info("=" * 60)
logger.info("Python files only, larger chunks for speed")

# Step 1: Initialize embedding model with GPU support
logger.info("[1/5] Loading embedding model (Nomic Embed)...")

# Check for GPU availability
device = get_device()
batch_size = get_batch_size()

if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"   [GPU ENABLED] {gpu_name} ({gpu_memory:.1f} GB VRAM)")
    logger.info(f"   [INFO] PyTorch {torch.__version__} with CUDA {torch.version.cuda}")
    logger.info(f"   [INFO] sm_120 support: {'sm_120' in torch.cuda.get_arch_list()}")
else:
    logger.warning("   No GPU detected, using CPU (slower)")

# Initialize with sentence-transformers directly for better GPU control
embed_model = HuggingFaceEmbedding(
    model_name=Config.EMBEDDING_MODEL_NAME,
    cache_folder=str(Config.SENTENCE_TRANSFORMERS_HOME),
    trust_remote_code=True,
    embed_batch_size=batch_size,
    device=device  # GPU acceleration
)
Settings.embed_model = embed_model
logger.info(f"   [OK] Embedding model loaded (768 dimensions)")
logger.info(f"   [OK] Device: {device.upper()} | Batch size: {batch_size}")

# Step 2: Load documents
logger.info("[2/5] Loading Python files from Academicon...")
logger.info(f"   Source: {Config.ACADEMICON_PATH}")
logger.info(f"   Extensions: {Config.INDEX_FILE_EXTENSIONS}")

start_time = time.time()

documents = SimpleDirectoryReader(
    input_dir=Config.ACADEMICON_PATH,
    recursive=True,
    required_exts=Config.INDEX_FILE_EXTENSIONS,
    exclude_hidden=True,
    exclude=Config.EXCLUDE_DIRS
).load_data()

load_time = time.time() - start_time
logger.info(f"   [OK] Loaded {len(documents)} Python files in {load_time:.2f}s")

# Step 3: Split into LARGER chunks (fewer embeddings needed)
logger.info("[3/5] Splitting code into chunks...")
start_time = time.time()

splitter = SentenceSplitter(
    chunk_size=Config.CHUNK_SIZE,
    chunk_overlap=Config.CHUNK_OVERLAP,
)

nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

split_time = time.time() - start_time
logger.info(f"   [OK] Created {len(nodes)} chunks in {split_time:.2f}s")
logger.info(f"   [INFO] Estimated embedding time: ~{len(nodes) * 0.5 / 60:.1f} minutes")

# Step 4: Setup ChromaDB
logger.info("[4/5] Setting up ChromaDB vector store...")
chroma_client = chromadb.PersistentClient(path=str(Config.DB_PATH))

# Delete existing collection if it exists
try:
    chroma_client.delete_collection(Config.DB_COLLECTION_NAME)
    logger.info("   [INFO] Deleted existing collection")
except:
    pass

chroma_collection = chroma_client.create_collection(Config.DB_COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
logger.info(f"   [OK] ChromaDB initialized at {Config.DB_PATH}")

# Step 5: Build index
logger.info("[5/5] Building vector index...")
logger.info(f"   [INFO] Processing {len(nodes)} chunks...")
start_time = time.time()

index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    show_progress=True
)

index_time = time.time() - start_time

# Final summary
logger.info("=" * 60)
logger.info("Indexing Complete!")
logger.info("=" * 60)
logger.info(f"Total Python files:  {len(documents)}")
logger.info(f"Total chunks:        {len(nodes)}")
logger.info(f"Database path:       {Config.DB_PATH}")
logger.info(f"Total time:          {index_time/60:.2f} minutes")
logger.info(f"Avg per chunk:       {index_time/len(nodes):.3f}s")
logger.info("=" * 60)
logger.info("[OK] Vector database ready for queries!")
logger.info("Next step: Run 'python src/main_openrouter.py' to test the assistant")
