"""
Incremental Index Update
Updates only changed files instead of re-indexing everything
Much faster than full re-index (seconds vs minutes)
"""

import os
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import torch
import time
from pathlib import Path
import hashlib

# Set cache directories
os.environ['HF_HOME'] = 'D:/AI-Models/huggingface-moved'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:/AI-Models/huggingface-moved/hub'
os.environ['TRANSFORMERS_CACHE'] = 'D:/AI-Models/transformers'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = 'D:/AI-Models/embeddings'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

ACADEMICON_PATH = "//wsl$/Ubuntu/home/deeznutz/projects/Academicon-Rebuild"
ALLOWED_EXTENSIONS = [".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".json", ".yml", ".yaml", ".md"]
DB_PATH = "./academicon_chroma_db"
HASH_FILE = "./last_index_state.txt"

print("="*60)
print("Incremental Index Update")
print("="*60)

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16 if device == "cuda" else 8  # Reduced for large datasets

print(f"\n[1/4] Loading embedding model...")

# Clear GPU cache first
if device == "cuda":
    torch.cuda.empty_cache()
    print(f"   [OK] GPU cache cleared")

embed_model = HuggingFaceEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    cache_folder="D:/AI-Models/embeddings",
    trust_remote_code=True,
    embed_batch_size=batch_size,
    device=device
)
Settings.embed_model = embed_model
print(f"   [OK] Device: {device.upper()}")

# Load existing database
print(f"\n[2/4] Loading existing database...")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
chroma_collection = chroma_client.get_collection("academicon_code")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

existing_count = chroma_collection.count()
print(f"   [OK] Current vectors: {existing_count}")

# Get file hashes to detect changes
print(f"\n[3/4] Detecting changed files...")

def get_file_hash(file_path):
    """Get MD5 hash of file content"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

# Load previous state
old_hashes = {}
if os.path.exists(HASH_FILE):
    with open(HASH_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if '|' in line:
                path, hash_val = line.strip().split('|')
                old_hashes[path] = hash_val

# Scan current files
changed_files = []
new_hashes = {}

for root, dirs, files in os.walk(ACADEMICON_PATH):
    # Skip excluded directories
    dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', 'dist', 'build', '__pycache__', 'venv', '.venv', 'migrations']]

    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in ALLOWED_EXTENSIONS:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, ACADEMICON_PATH)

            try:
                current_hash = get_file_hash(file_path)
                new_hashes[rel_path] = current_hash

                if rel_path not in old_hashes or old_hashes[rel_path] != current_hash:
                    changed_files.append(file_path)
            except Exception as e:
                print(f"   [SKIP] {rel_path}: {e}")

print(f"   [OK] Found {len(changed_files)} changed/new files")

if len(changed_files) == 0:
    print("\n[INFO] No changes detected. Index is up to date!")
else:
    # Index only changed files
    print(f"\n[4/4] Indexing {len(changed_files)} changed files...")
    start_time = time.time()

    # Load and split changed files
    documents = SimpleDirectoryReader(
        input_files=changed_files
    ).load_data()

    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

    # Add to existing index
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True
    )

    elapsed = time.time() - start_time
    new_count = chroma_collection.count()

    print(f"\n{'='*60}")
    print(f"Update Complete!")
    print(f"{'='*60}")
    print(f"Changed files:  {len(changed_files)}")
    print(f"New chunks:     {new_count - existing_count}")
    print(f"Total vectors:  {new_count}")
    print(f"Update time:    {elapsed:.2f}s")
    print(f"{'='*60}")

    # Save new state
    with open(HASH_FILE, 'w', encoding='utf-8') as f:
        for path, hash_val in sorted(new_hashes.items()):
            f.write(f"{path}|{hash_val}\n")

    print(f"\n[OK] Index state saved!")
