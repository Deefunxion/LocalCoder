import os
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set cache directories to D: drive
os.environ['HF_HOME'] = 'D:/AI-Models/huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:/AI-Models/huggingface/hub'
os.environ['TRANSFORMERS_CACHE'] = 'D:/AI-Models/transformers'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = 'D:/AI-Models/embeddings'

# Initialize Nomic Embed with GPU support
print("Loading Nomic Embed model...")
print("Cache directory: D:/AI-Models/embeddings")

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU detected: {gpu_name}")
else:
    print("No GPU detected, using CPU")

embed_model = HuggingFaceEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    cache_folder="D:/AI-Models/embeddings",
    trust_remote_code=True,
    device=device
)

# Test embedding
test_code = "def authenticate_user(username, password):"
embedding = embed_model.get_text_embedding(test_code)

print(f"[OK] Agent 1 (Indexer) working!")
print(f"   Device: {device.upper()}")
print(f"   Embedding dimension: {len(embedding)}")
print(f"   Sample values: {embedding[:5]}")
