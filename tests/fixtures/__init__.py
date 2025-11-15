# tests/fixtures/__init__.py
# Test fixtures and sample data for LocalCoder tests

import pytest
from unittest.mock import Mock
from pathlib import Path


# Sample code chunks for testing
@pytest.fixture
def sample_code_chunks():
    """Sample code chunks that might be returned by the indexer"""
    return [
        {
            "text": "def calculate_total(items):\n    return sum(item.price for item in items)",
            "score": 0.95,
            "metadata": {
                "file_path": "utils/math.py",
                "line_start": 10,
                "line_end": 12
            }
        },
        {
            "text": "class ShoppingCart:\n    def __init__(self):\n        self.items = []\n\n    def add_item(self, item):\n        self.items.append(item)",
            "score": 0.87,
            "metadata": {
                "file_path": "models/cart.py",
                "line_start": 1,
                "line_end": 6
            }
        },
        {
            "text": "import json\nfrom typing import List\n\ndef save_to_file(data, filename):\n    with open(filename, 'w') as f:\n        json.dump(data, f)",
            "score": 0.76,
            "metadata": {
                "file_path": "utils/file_ops.py",
                "line_start": 1,
                "line_end": 7
            }
        }
    ]


@pytest.fixture
def sample_user_query():
    """Sample user query for testing"""
    return "How do I calculate the total price of items in a shopping cart?"


@pytest.fixture
def sample_json_response():
    """Sample JSON response from LLM"""
    return '''```json
{
  "search_queries": ["calculate total price", "shopping cart methods"],
  "analysis_needed": true
}
```'''


@pytest.fixture
def sample_analysis_result():
    """Sample analysis result from GraphAnalystAgent"""
    return {
        "functions": ["calculate_total", "add_item", "save_to_file"],
        "dependencies": ["json", "typing"],
        "relationships": ["ShoppingCart uses calculate_total"]
    }


@pytest.fixture
def sample_conversation_history():
    """Sample conversation history"""
    return [
        {
            "query": "What functions are available?",
            "answer": "The main functions are calculate_total and add_item."
        },
        {
            "query": "How does calculate_total work?",
            "answer": "It sums the prices of all items using a generator expression."
        }
    ]


@pytest.fixture
def mock_vector_index():
    """Mock VectorStoreIndex for testing"""
    mock_index = Mock()
    mock_retriever = Mock()
    mock_node1 = Mock()
    mock_node1.node.text = "def test_function():\n    return 'hello'"
    mock_node1.node.metadata = {"file_path": "test.py"}
    mock_node1.score = 0.9

    mock_node2 = Mock()
    mock_node2.node.text = "class TestClass:\n    pass"
    mock_node2.node.metadata = {"file_path": "test.py"}
    mock_node2.score = 0.8

    mock_retriever.retrieve.return_value = [mock_node1, mock_node2]
    mock_index.as_retriever.return_value = mock_retriever

    return mock_index


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing"""
    mock_response = Mock()
    mock_response.__str__ = Mock(return_value='{"test": "response"}')
    return mock_response


@pytest.fixture
def temp_log_dir(tmp_path):
    """Temporary directory for log testing"""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir