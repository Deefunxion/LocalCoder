# tests/unit/test_agents.py
# Unit tests for agent classes in src/agents_openrouter.py

import pytest
from unittest.mock import Mock, patch, MagicMock
from llama_index.llms.openai_like import OpenAILike

from src.agents_openrouter import (
    IndexerAgent,
    GraphAnalystAgent,
    OrchestratorAgent,
    SynthesizerAgent
)


class TestIndexerAgent:
    """Test IndexerAgent functionality"""

    def test_init(self, mock_vector_index):
        """Test IndexerAgent initialization"""
        agent = IndexerAgent(mock_vector_index)
        assert agent.index == mock_vector_index

    def test_retrieve_chunks(self, mock_vector_index):
        """Test retrieving chunks from vector database"""
        agent = IndexerAgent(mock_vector_index)

        results = agent.retrieve("test query", top_k=2)

        assert len(results) == 2
        assert results[0]["text"] == "def test_function():\n    return 'hello'"
        assert results[0]["score"] == 0.9
        assert results[0]["metadata"] == {"file_path": "test.py"}

        # Verify the retriever was called correctly
        mock_vector_index.as_retriever.assert_called_once_with(similarity_top_k=2)
        retriever = mock_vector_index.as_retriever.return_value
        retriever.retrieve.assert_called_once_with("test query")

    def test_retrieve_empty_results(self):
        """Test handling of empty retrieval results"""
        mock_index = Mock()
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = []
        mock_index.as_retriever.return_value = mock_retriever

        agent = IndexerAgent(mock_index)
        results = agent.retrieve("empty query")

        assert results == []


class TestGraphAnalystAgent:
    """Test GraphAnalystAgent functionality"""

    @patch('src.agents_openrouter.validate_api_key')
    @patch('src.agents_openrouter.Config')
    def test_init_success(self, mock_config, mock_validate):
        """Test successful GraphAnalystAgent initialization"""
        mock_validate.return_value = True
        mock_config.OPENROUTER_API_KEY = "test-key"
        mock_config.GRAPH_ANALYST_MODEL = "test-model"
        mock_config.GRAPH_ANALYST_TEMPERATURE = 0.1
        mock_config.GRAPH_ANALYST_TIMEOUT = 60.0

        with patch('src.agents_openrouter.OpenAILike') as mock_llm_class:
            agent = GraphAnalystAgent()

            mock_llm_class.assert_called_once()
            args, kwargs = mock_llm_class.call_args
            assert kwargs['model'] == "test-model"
            assert kwargs['temperature'] == 0.1
            assert kwargs['timeout'] == 60.0

    @patch('src.agents_openrouter.validate_api_key')
    def test_init_missing_api_key(self, mock_validate):
        """Test initialization failure with missing API key"""
        mock_validate.return_value = False

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY not set"):
            GraphAnalystAgent()

    def test_analyze_relationships_empty_chunks(self):
        """Test analysis with empty code chunks"""
        agent = GraphAnalystAgent.__new__(GraphAnalystAgent)  # Create without __init__

        result = agent.analyze_relationships([])

        expected = {"functions": [], "dependencies": [], "relationships": []}
        assert result == expected

    @patch('src.agents_openrouter.extract_json_from_response')
    def test_analyze_relationships_success(self, mock_extract, sample_code_chunks):
        """Test successful code analysis"""
        agent = GraphAnalystAgent.__new__(GraphAnalystAgent)
        mock_llm = Mock()
        agent.llm = mock_llm

        mock_response = Mock()
        mock_response.__str__ = Mock(return_value='{"functions": ["test"], "dependencies": [], "relationships": []}')
        mock_llm.complete.return_value = mock_response
        mock_extract.return_value = {"functions": ["test"], "dependencies": [], "relationships": []}

        result = agent.analyze_relationships(sample_code_chunks)

        assert "functions" in result
        mock_llm.complete.assert_called_once()
        # Verify prompt contains expected content
        prompt = mock_llm.complete.call_args[0][0]
        assert "Analyze these code chunks" in prompt
        assert "Main function/class names" in prompt
        assert "Import dependencies" in prompt

    @patch('src.agents_openrouter.extract_json_from_response')
    def test_analyze_relationships_llm_failure(self, mock_extract, sample_code_chunks):
        """Test handling of LLM failure during analysis"""
        agent = GraphAnalystAgent.__new__(GraphAnalystAgent)
        mock_llm = Mock()
        agent.llm = mock_llm

        mock_llm.complete.side_effect = Exception("API Error")
        mock_extract.return_value = {}  # Failed extraction

        result = agent.analyze_relationships(sample_code_chunks)

        expected = {"functions": [], "dependencies": [], "relationships": []}
        assert result == expected


class TestOrchestratorAgent:
    """Test OrchestratorAgent functionality"""

    @patch('src.agents_openrouter.validate_api_key')
    @patch('src.agents_openrouter.Config')
    def test_init_success(self, mock_config, mock_validate):
        """Test successful OrchestratorAgent initialization"""
        mock_validate.return_value = True
        mock_config.OPENROUTER_API_KEY = "test-key"
        mock_config.ORCHESTRATOR_MODEL = "test-model"
        mock_config.ORCHESTRATOR_TEMPERATURE = 0.1
        mock_config.ORCHESTRATOR_TIMEOUT = 60.0

        with patch('src.agents_openrouter.OpenAILike') as mock_llm_class:
            agent = OrchestratorAgent()

            mock_llm_class.assert_called_once()
            args, kwargs = mock_llm_class.call_args
            assert kwargs['model'] == "test-model"
            assert kwargs['temperature'] == 0.1

    @patch('src.agents_openrouter.validate_api_key')
    def test_init_missing_api_key(self, mock_validate):
        """Test initialization failure with missing API key"""
        mock_validate.return_value = False

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY not set"):
            OrchestratorAgent()

    @patch('src.agents_openrouter.extract_json_from_response')
    def test_plan_query_success(self, mock_extract, sample_user_query):
        """Test successful query planning"""
        agent = OrchestratorAgent.__new__(OrchestratorAgent)
        mock_llm = Mock()
        agent.llm = mock_llm

        mock_response = Mock()
        mock_response.__str__ = Mock(return_value='{"search_queries": ["test query"], "analysis_needed": true}')
        mock_llm.complete.return_value = mock_response
        mock_extract.return_value = {"search_queries": ["test query"], "analysis_needed": True}

        result = agent.plan_query(sample_user_query)

        assert result["search_queries"] == ["test query"]
        assert result["analysis_needed"] == True

        # Verify prompt content
        prompt = mock_llm.complete.call_args[0][0]
        assert sample_user_query in prompt
        assert "search queries" in prompt.lower()

    @patch('src.agents_openrouter.extract_json_from_response')
    def test_plan_query_llm_failure(self, mock_extract, sample_user_query):
        """Test handling of LLM failure during planning"""
        agent = OrchestratorAgent.__new__(OrchestratorAgent)
        mock_llm = Mock()
        agent.llm = mock_llm

        mock_llm.complete.side_effect = Exception("API Error")
        mock_extract.return_value = {}  # Failed extraction

        result = agent.plan_query(sample_user_query)

        # Should return fallback result
        assert "search_queries" in result
        assert "analysis_needed" in result
        assert result["search_queries"] == [sample_user_query]
        assert result["analysis_needed"] == False


class TestSynthesizerAgent:
    """Test SynthesizerAgent functionality"""

    @patch('src.agents_openrouter.validate_api_key')
    @patch('src.agents_openrouter.Config')
    def test_init_success(self, mock_config, mock_validate):
        """Test successful SynthesizerAgent initialization"""
        mock_validate.return_value = True
        mock_config.OPENROUTER_API_KEY = "test-key"
        mock_config.SYNTHESIZER_MODEL = "test-model"
        mock_config.SYNTHESIZER_TEMPERATURE = 0.2
        mock_config.SYNTHESIZER_TIMEOUT = 90.0

        with patch('src.agents_openrouter.OpenAILike') as mock_llm_class:
            agent = SynthesizerAgent()

            mock_llm_class.assert_called_once()
            args, kwargs = mock_llm_class.call_args
            assert kwargs['model'] == "test-model"
            assert kwargs['temperature'] == 0.2

    @patch('src.agents_openrouter.validate_api_key')
    def test_init_missing_api_key(self, mock_validate):
        """Test initialization failure with missing API key"""
        mock_validate.return_value = False

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY not set"):
            SynthesizerAgent()

    def test_synthesize_no_chunks(self):
        """Test synthesis with no code chunks"""
        agent = SynthesizerAgent.__new__(SynthesizerAgent)

        result = agent.synthesize("test query", {"code_chunks": []})

        assert "couldn't find relevant code" in result.lower()

    def test_synthesize_success(self, sample_code_chunks, sample_analysis_result):
        """Test successful answer synthesis"""
        agent = SynthesizerAgent.__new__(SynthesizerAgent)
        mock_llm = Mock()
        agent.llm = mock_llm

        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="This is the synthesized answer.")
        mock_llm.complete.return_value = mock_response

        context = {
            "code_chunks": sample_code_chunks,
            "analysis": sample_analysis_result,
            "conversation_history": []
        }

        result = agent.synthesize("How does calculate_total work?", context)

        assert result == "This is the synthesized answer."

        # Verify prompt construction
        prompt = mock_llm.complete.call_args[0][0]
        assert "calculate_total" in prompt
        assert "How does calculate_total work?" in prompt
        assert "Code Analysis:" in prompt
        assert "functions" in prompt  # Should be in the JSON analysis

    def test_synthesize_with_history(self, sample_code_chunks, sample_conversation_history):
        """Test synthesis including conversation history"""
        agent = SynthesizerAgent.__new__(SynthesizerAgent)
        mock_llm = Mock()
        agent.llm = mock_llm

        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="Answer with history context.")
        mock_llm.complete.return_value = mock_response

        context = {
            "code_chunks": sample_code_chunks,
            "analysis": {},
            "conversation_history": sample_conversation_history
        }

        result = agent.synthesize("Follow-up question", context)

        prompt = mock_llm.complete.call_args[0][0]
        assert "Previous Conversation:" in prompt
        assert "What functions are available?" in prompt

    @patch('src.agents_openrouter.logger')
    def test_synthesize_llm_failure(self, mock_logger, sample_code_chunks):
        """Test handling of LLM failure during synthesis"""
        agent = SynthesizerAgent.__new__(SynthesizerAgent)
        mock_llm = Mock()
        agent.llm = mock_llm

        mock_llm.complete.side_effect = Exception("API Error")

        context = {"code_chunks": sample_code_chunks}

        result = agent.synthesize("test query", context)

        assert "Error generating answer" in result
        mock_logger.error.assert_called_once()