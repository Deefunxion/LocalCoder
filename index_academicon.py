import os
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import CodeSplitter
import time

# Set cache directories to D: drive
os.environ['HF_HOME'] = 'D:/AI-Models/huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:/AI-Models/huggingface/hub'
os.environ['TRANSFORMERS_CACHE'] = 'D:/AI-Models/transformers'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = 'D:/AI-Models/embeddings'

# Configuration
ACADEMICON_PATH = "//wsl$/Ubuntu/home/deeznutz/projects/Academicon-Rebuild"
ALLOWED_EXTENSIONS = [".py", ".js", ".jsx", ".ts", ".tsx", ".vue"]
DB_PATH = "./academicon_chroma_db"

print("="*60)
print("Academicon Codebase Indexing")
print("="*60)

# Step 1: Initialize embedding model
print("\n[1/5] Loading embedding model (Nomic Embed)...")
embed_model = HuggingFaceEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    cache_folder="D:/AI-Models/embeddings",
    trust_remote_code=True
)
Settings.embed_model = embed_model
print("   [OK] Embedding model loaded (768 dimensions)")

# Step 2: Load documents
print("\n[2/5] Loading documents from Academicon...")
print(f"   Source: {ACADEMICON_PATH}")
print(f"   Extensions: {', '.join(ALLOWED_EXTENSIONS)}")

start_time = time.time()

documents = SimpleDirectoryReader(
    input_dir=ACADEMICON_PATH,
    recursive=True,
    required_exts=ALLOWED_EXTENSIONS,
    exclude_hidden=True,
    exclude=["node_modules", ".git", "dist", "build", "__pycache__", "venv", ".venv"]
).load_data()

load_time = time.time() - start_time
print(f"   [OK] Loaded {len(documents)} documents in {load_time:.2f}s")

# Step 3: Split into chunks
print("\n[3/5] Splitting code into chunks...")
start_time = time.time()

# Use a generic splitter that works for all languages
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=1024,        # Characters per chunk
    chunk_overlap=200,      # Overlap for context
)

nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

split_time = time.time() - start_time
print(f"   [OK] Created {len(nodes)} chunks in {split_time:.2f}s")

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

# Step 5: Build index (this will take time!)
print("\n[5/5] Building vector index...")
print("   [INFO] This may take 30-60 minutes for large codebases...")
print("   [INFO] Progress updates will show below:")
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
print(f"Total documents:  {len(documents)}")
print(f"Total chunks:     {len(nodes)}")
print(f"Database path:    {DB_PATH}")
print(f"Total time:       {index_time/60:.2f} minutes")
print(f"Avg per chunk:    {index_time/len(nodes):.3f}s")
print("="*60)
print("\n[OK] Vector database ready for queries!")
