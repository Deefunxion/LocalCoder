# tests/unit/test_utils.py
# Unit tests for utility functions in src/utils.py

import pytest
import json
import logging
from unittest.mock import patch, mock_open
from pathlib import Path

from src.utils import (
    extract_json_from_response,
    validate_api_key,
    format_code_context,
    format_conversation_history,
    truncate_text,
    merge_metadata,
    setup_logger
)


class TestExtractJsonFromResponse:
    """Test JSON extraction from various response formats"""

    def test_extract_plain_json(self):
        """Test extracting plain JSON without code blocks"""
        response = '{"key": "value", "number": 42}'
        result = extract_json_from_response(response)
        assert result == {"key": "value", "number": 42}

    def test_extract_json_from_markdown_block(self):
        """Test extracting JSON from markdown code blocks"""
        response = '''Here is the result:
```json
{
  "functions": ["func1", "func2"],
  "dependencies": ["import1"]
}
```
That's all.'''
        result = extract_json_from_response(response)
        expected = {
            "functions": ["func1", "func2"],
            "dependencies": ["import1"]
        }
        assert result == expected

    def test_extract_json_from_generic_code_block(self):
        """Test extracting JSON from generic code blocks"""
        response = '''```
{"test": "value"}
```'''
        result = extract_json_from_response(response)
        assert result == {"test": "value"}

    def test_extract_json_with_json_prefix(self):
        """Test extracting JSON that has 'json ' prefix"""
        response = 'json {"key": "value"}'
        result = extract_json_from_response(response)
        assert result == {"key": "value"}

    def test_extract_invalid_json(self):
        """Test handling of invalid JSON"""
        response = '{"invalid": json}'
        result = extract_json_from_response(response)
        assert result == {}

    def test_extract_empty_response(self):
        """Test handling of empty response"""
        response = ""
        result = extract_json_from_response(response)
        assert result == {}


class TestValidateApiKey:
    """Test API key validation"""

    def test_valid_api_key(self):
        """Test valid API key"""
        assert validate_api_key("sk-or-v1-valid-key") == True

    def test_none_api_key(self):
        """Test None API key"""
        assert validate_api_key(None) == False

    def test_empty_api_key(self):
        """Test empty API key"""
        assert validate_api_key("") == False

    def test_whitespace_api_key(self):
        """Test whitespace-only API key"""
        assert validate_api_key("   ") == False


class TestFormatCodeContext:
    """Test code context formatting"""

    def test_format_code_chunks(self, sample_code_chunks):
        """Test formatting code chunks for prompts"""
        result = format_code_context(sample_code_chunks, max_chunks=2)

        assert "[Chunk 1 - utils/math.py]" in result
        assert "calculate_total" in result
        assert "[Chunk 2 - models/cart.py]" in result
        assert "ShoppingCart" in result
        # Should not include third chunk due to max_chunks=2
        assert "save_to_file" not in result

    def test_format_empty_chunks(self):
        """Test formatting with empty chunk list"""
        result = format_code_context([])
        assert result == ""

    def test_format_chunks_without_metadata(self):
        """Test formatting chunks missing metadata"""
        chunks = [{"text": "some code", "score": 0.8}]
        result = format_code_context(chunks)
        assert "[Chunk 1 - unknown]" in result


class TestFormatConversationHistory:
    """Test conversation history formatting"""

    def test_format_history(self, sample_conversation_history):
        """Test formatting conversation history"""
        result = format_conversation_history(sample_conversation_history, max_exchanges=2)

        assert "Previous Conversation:" in result
        assert "Q1: What functions are available?" in result
        assert "A1: The main functions are" in result
        assert "Q2: How does calculate_total work?" in result

    def test_format_empty_history(self):
        """Test formatting with empty history"""
        result = format_conversation_history([])
        assert result == ""

    def test_format_history_truncation(self, sample_conversation_history):
        """Test that answers are truncated"""
        # Make a long answer
        long_history = [{
            "query": "Test query",
            "answer": "A" * 300  # 300 character answer
        }]

        result = format_conversation_history(long_history)
        assert "..." in result
        assert len(result.split("A1: ")[1].split("...")[0]) <= 200


class TestTruncateText:
    """Test text truncation utility"""

    def test_truncate_short_text(self):
        """Test truncation of text shorter than max length"""
        text = "Short text"
        result = truncate_text(text, max_length=20)
        assert result == "Short text"

    def test_truncate_long_text(self):
        """Test truncation of text longer than max length"""
        text = "A" * 100
        result = truncate_text(text, max_length=50)
        assert len(result) == 50  # 47 + len("...")
        assert result.endswith("...")

    def test_truncate_with_custom_suffix(self):
        """Test truncation with custom suffix"""
        text = "A" * 100
        result = truncate_text(text, max_length=10, suffix="***")
        assert result.endswith("***")


class TestMergeMetadata:
    """Test metadata merging utility"""

    def test_merge_file_paths(self, sample_code_chunks):
        """Test merging file paths from chunks"""
        result = merge_metadata(sample_code_chunks, key='file_path')

        expected = {
            "utils/math.py": 1,
            "models/cart.py": 1,
            "utils/file_ops.py": 1
        }
        assert result == expected

    def test_merge_with_duplicates(self):
        """Test merging with duplicate values"""
        chunks = [
            {"metadata": {"file_path": "test.py"}},
            {"metadata": {"file_path": "test.py"}},
            {"metadata": {"file_path": "other.py"}}
        ]

        result = merge_metadata(chunks, key='file_path')
        assert result["test.py"] == 2
        assert result["other.py"] == 1

    def test_merge_missing_metadata(self):
        """Test merging with missing metadata"""
        chunks = [{"text": "code"}, {"metadata": {}}]
        result = merge_metadata(chunks, key='file_path')
        assert result == {}


class TestSetupLogger:
    """Test logger setup"""

    def test_setup_logger_creates_logger(self):
        """Test that setup_logger creates a properly configured logger"""
        logger = setup_logger("test_logger")

        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) >= 2  # Console and file handlers

    def test_setup_logger_idempotent(self):
        """Test that calling setup_logger multiple times doesn't add duplicate handlers"""
        logger1 = setup_logger("test_logger_2")
        handler_count_1 = len(logger1.handlers)

        logger2 = setup_logger("test_logger_2")
        handler_count_2 = len(logger2.handlers)

        assert handler_count_1 == handler_count_2

    @patch('pathlib.Path.mkdir')
    @patch('logging.FileHandler')
    def test_setup_logger_creates_log_directory(self, mock_file_handler, mock_mkdir, temp_log_dir):
        """Test that log directory is created"""
        with patch('src.utils.Path') as mock_path:
            mock_path.return_value.parent.parent = temp_log_dir
            mock_path.return_value.mkdir = mock_mkdir

            setup_logger("test_logger_3")

            # Verify directory creation was attempted
            mock_mkdir.assert_called_once_with(exist_ok=True)