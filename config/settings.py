"""
Centralized configuration module for LocalCoder

Handles all environment variables, paths, and settings with OS-agnostic defaults.
Eliminates 35+ hardcoded lines and provides single source of truth for configuration.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file at module import time
load_dotenv()


class Config:
    """Centralized configuration class with environment-aware defaults"""

    # ============================================================================
    # CACHE & MODEL DIRECTORIES (OS-Agnostic with Windows D: drive fallback)
    # ============================================================================

    @staticmethod
    def _get_cache_base() -> Path:
        """Get base cache directory with smart defaults"""
        env_base = os.getenv('CACHE_BASE_DIR')
        if env_base:
            return Path(env_base)
        
        # Default to D:/AI-Models on Windows (avoid C: drive space issues)
        if os.name == 'nt':  # Windows
            return Path('D:/AI-Models')
        else:  # Linux/macOS
            return Path.home() / '.cache' / 'academicon'

    CACHE_BASE = _get_cache_base()
    
    # HuggingFace caching
    HF_HOME = CACHE_BASE / 'huggingface-moved'
    HF_HUB_CACHE = HF_HOME / 'hub'
    TRANSFORMERS_CACHE = CACHE_BASE / 'transformers'
    SENTENCE_TRANSFORMERS_HOME = CACHE_BASE / 'embeddings'

    # ============================================================================
    # VECTOR DATABASE
    # ============================================================================
    
    DB_PATH = os.getenv('DB_PATH', './academicon_chroma_db')
    DB_COLLECTION_NAME = 'academicon_code'

    # ============================================================================
    # ACADEMICON CODEBASE PATH
    # ============================================================================
    
    ACADEMICON_PATH = os.getenv(
        'ACADEMICON_PATH',
        '//wsl$/Ubuntu/home/deeznutz/projects/Academicon-Rebuild'
    )

    # ============================================================================
    # EMBEDDING MODEL CONFIGURATION
    # ============================================================================
    
    EMBEDDING_MODEL_NAME = 'nomic-ai/nomic-embed-text-v1.5'
    EMBEDDING_DIMENSIONS = 768
    EMBEDDING_DEVICE = 'cpu'  # Temporarily use CPU due to RTX 5070 Ti compatibility issues

    # ============================================================================
    # OPENROUTER API CONFIGURATION
    # ============================================================================
    
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'

    # Model names for each agent (with defaults)
    # Using free OpenRouter models by default
    ORCHESTRATOR_MODEL = os.getenv('ORCHESTRATOR_MODEL', 'google/gemini-2.0-flash-exp-1219')
    SYNTHESIZER_MODEL = os.getenv('SYNTHESIZER_MODEL', 'openai/gpt-4-turbo-preview')
    GRAPH_ANALYST_MODEL = os.getenv('GRAPH_ANALYST_MODEL', 'openai/gpt-3.5-turbo')
    FALLBACK_MODEL = os.getenv('FALLBACK_MODEL', 'google/gemini-2.0-flash-exp-1219')

    # ============================================================================
    # AGENT CONFIGURATION (Timeouts, Temperature, etc.)
    # ============================================================================
    
    # Timeouts (in seconds)
    ORCHESTRATOR_TIMEOUT = 60.0
    GRAPH_ANALYST_TIMEOUT = 60.0
    SYNTHESIZER_TIMEOUT = 90.0
    
    # Temperature settings (0.0 = deterministic, 1.0 = random)
    ORCHESTRATOR_TEMPERATURE = 0.1  # Deterministic planning
    SYNTHESIZER_TEMPERATURE = 0.2   # Slightly creative synthesis
    GRAPH_ANALYST_TEMPERATURE = 0.1  # Deterministic analysis
    
    # Retrieval settings
    RETRIEVAL_TOP_K = 5  # Number of chunks to retrieve per query
    CONVERSATION_HISTORY_SIZE = 3  # Keep last N exchanges for context

    # ============================================================================
    # GPU & BATCH SIZE SETTINGS
    # ============================================================================
    
    GPU_BATCH_SIZE = 128  # For GPU-accelerated embedding
    CPU_BATCH_SIZE = 32   # For CPU-based embedding

    # ============================================================================
    # INDEXING CONFIGURATION
    # ============================================================================
    
    # File types to index
    ALLOWED_EXTENSIONS = ['.py']  # Python only for lite version
    
    # Chunk configuration
    CHUNK_SIZE = 2048
    CHUNK_OVERLAP = 256
    
    # Directories to exclude
    EXCLUDE_DIRS = [
        'node_modules', '.git', 'dist', 'build',
        '__pycache__', 'venv', '.venv', 'migrations',
        '.pytest_cache', '.mypy_cache'
    ]

    # ============================================================================
    # WEB UI CONFIGURATION
    # ============================================================================
    
    WEB_UI_HOST = '127.0.0.1'  # localhost only
    WEB_UI_PORT = None  # Auto-find available port
    WEB_UI_SHARE = False  # Don't create public link
    WEB_UI_SHOW_ERROR = True

    # ============================================================================
    # LOGGING CONFIGURATION
    # ============================================================================
    
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    LOG_DATE_FORMAT = '%H:%M:%S'

    # ============================================================================
    # ENVIRONMENT SETUP (Call this once at startup)
    # ============================================================================

    @classmethod
    def setup_environment(cls) -> None:
        """
        Set all environment variables for HuggingFace/Transformers caching.
        
        MUST be called once at application startup before importing transformers/torch.
        This prevents models from downloading to C: drive (Windows) and ensures
        consistent cache location across the application.
        """
        # HuggingFace caching
        os.environ['HF_HOME'] = str(cls.HF_HOME)
        os.environ['HUGGINGFACE_HUB_CACHE'] = str(cls.HF_HUB_CACHE)
        os.environ['TRANSFORMERS_CACHE'] = str(cls.TRANSFORMERS_CACHE)
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cls.SENTENCE_TRANSFORMERS_HOME)
        
        # Disable HuggingFace symlink warnings
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

    # ============================================================================
    # VALIDATION METHODS
    # ============================================================================

    @classmethod
    def validate(cls) -> tuple[bool, str]:
        """
        Validate critical configuration settings.
        
        Returns:
            (is_valid: bool, error_message: str)
        """
        # Check OpenRouter API key
        if not cls.OPENROUTER_API_KEY:
            return False, "OPENROUTER_API_KEY not set in .env file"
        
        if not cls.OPENROUTER_API_KEY.startswith('sk-or-v1-'):
            return False, "OPENROUTER_API_KEY format invalid (should start with 'sk-or-v1-')"
        
        # Check vector database path
        db_path = Path(cls.DB_PATH)
        if not db_path.exists():
            return False, f"Vector database not found at {cls.DB_PATH}. Run index_academicon_lite.py first."
        
        return True, "Configuration valid"

    @classmethod
    def get_summary(cls) -> str:
        """Get human-readable configuration summary"""
        return f"""
╔════════════════════════════════════════════════════════════╗
║           LocalCoder Configuration Summary                 ║
╚════════════════════════════════════════════════════════════╝

📁 PATHS:
   Cache Base:              {cls.CACHE_BASE}
   Vector Database:         {cls.DB_PATH}
   Academicon Codebase:     {cls.ACADEMICON_PATH}

🤖 MODELS:
   Orchestrator:            {cls.ORCHESTRATOR_MODEL}
   Synthesizer:             {cls.SYNTHESIZER_MODEL}
   Graph Analyst:           {cls.GRAPH_ANALYST_MODEL}
   Fallback:                {cls.FALLBACK_MODEL}

⚙️ SETTINGS:
   Embedding Model:         {cls.EMBEDDING_MODEL_NAME}
   Retrieval Top-K:         {cls.RETRIEVAL_TOP_K}
   History Size:            {cls.CONVERSATION_HISTORY_SIZE}
   Chunk Size:              {cls.CHUNK_SIZE}

🌐 WEB UI:
   Host:                    {cls.WEB_UI_HOST}
   Port:                    Auto-detect
   Share:                   {cls.WEB_UI_SHARE}
"""


# ============================================================================
# DEFAULT CONFIGURATION INSTANCE
# ============================================================================

config = Config()


# ============================================================================
# UTILITY FUNCTIONS (For backwards compatibility)
# ============================================================================

def get_device() -> str:
    """Smart GPU detection with fallback for RTX 5070 Ti compatibility"""
    try:
        import torch
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            return "cpu"
        
        # Get GPU info
        gpu_name = torch.cuda.get_device_name(0).lower()
        
        # RTX 5070 Ti has known compatibility issues with SentenceTransformer
        if "rtx 5070" in gpu_name:
            print("⚠️  RTX 5070 Ti detected - using CPU for SentenceTransformer compatibility")
            return "cpu"
        
        # For other GPUs, try CUDA
        return "cuda"
        
    except Exception as e:
        print(f"⚠️  GPU detection failed ({e}) - using CPU")
        return "cpu"


def get_batch_size() -> int:
    """Get appropriate batch size based on device"""
    device = get_device()
    return Config.GPU_BATCH_SIZE if device == "cuda" else Config.CPU_BATCH_SIZE
