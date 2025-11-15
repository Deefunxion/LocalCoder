# utils.py
# Shared utilities for Academicon Code Assistant

import json
import logging
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path


def extract_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM response, handling markdown code blocks.
    
    Handles multiple formats:
    - Plain JSON
    - JSON wrapped in ```json ... ```
    - JSON wrapped in ``` ... ```
    
    Args:
        response: LLM response text
        
    Returns:
        Parsed JSON dict, or empty dict if parsing fails
    """
    try:
        response_text = str(response).strip()
        
        # Try to extract JSON from markdown code blocks
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            if json_end > json_start:
                response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            if json_end > json_start:
                response_text = response_text[json_start:json_end].strip()
        
        # Remove common JSON prefixes
        response_text = response_text.lstrip("json ")
        
        # Parse and return
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to parse JSON from response: {e}")
        return {}
    except Exception as e:
        logging.error(f"Error extracting JSON: {e}")
        return {}


def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Setup structured logger with console and file handlers.
    
    Args:
        name: Logger name (usually __name__)
        log_level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Only add handlers if they don't exist (avoid duplicates)
    if logger.handlers:
        return logger
    
    # Console handler with colored output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # File handler
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        log_dir / f"academicon_{timestamp}.log",
        encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    
    # Formatter
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def format_code_context(chunks: list, max_chunks: int = 5) -> str:
    """
    Format code chunks for inclusion in prompts.
    
    Args:
        chunks: List of chunk dicts with 'text' and 'metadata' keys
        max_chunks: Maximum number of chunks to include
        
    Returns:
        Formatted code context string
    """
    if not chunks:
        return ""
    
    code_context = []
    for i, chunk in enumerate(chunks[:max_chunks], 1):
        file_path = chunk.get('metadata', {}).get('file_path', 'unknown')
        code_text = chunk.get('text', '')
        code_context.append(f"[Chunk {i} - {file_path}]\n{code_text}\n")
    
    return "\n".join(code_context)


def format_conversation_history(history: list, max_exchanges: int = 3) -> str:
    """
    Format conversation history for inclusion in prompts.
    
    Args:
        history: List of dicts with 'query' and 'answer' keys
        max_exchanges: Maximum number of exchanges to include
        
    Returns:
        Formatted conversation history string
    """
    if not history:
        return ""
    
    formatted = "Previous Conversation:\n"
    for i, exchange in enumerate(history[-max_exchanges:], 1):
        query = exchange.get('query', '')
        answer = exchange.get('answer', '')[:200]  # First 200 chars
        formatted += f"\nQ{i}: {query}\n"
        formatted += f"A{i}: {answer}...\n"
    
    return formatted


def validate_api_key(api_key: Optional[str], provider: str = "OpenRouter") -> bool:
    """
    Validate that API key is present and non-empty.
    
    Args:
        api_key: API key to validate
        provider: Name of the provider (for error messages)
        
    Returns:
        True if valid, False otherwise
    """
    if not api_key or not api_key.strip():
        logging.error(f"{provider} API key not found or empty")
        return False
    return True


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with optional suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def merge_metadata(chunks: list, key: str = 'file_path') -> Dict[str, int]:
    """
    Merge metadata from chunks and count occurrences.
    
    Args:
        chunks: List of chunks with metadata
        key: Metadata key to extract and count
        
    Returns:
        Dict with unique values and their counts
    """
    counts = {}
    for chunk in chunks:
        metadata = chunk.get('metadata', {})
        value = metadata.get(key)
        if value:
            counts[value] = counts.get(value, 0) + 1
    return counts


if __name__ == "__main__":
    # Test utilities
    logger = setup_logger(__name__)
    logger.info("Utils module loaded successfully")
    
    # Test JSON extraction
    test_json = '```json\n{"test": "value"}\n```'
    result = extract_json_from_response(test_json)
    logger.info(f"JSON extraction test: {result}")
