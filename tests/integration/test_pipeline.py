# tests/integration/test_pipeline.py
# Integration tests for the full agent pipeline

import pytest
from unittest.mock import Mock, patch

from src.agents_openrouter import IndexerAgent, OrchestratorAgent, SynthesizerAgent


@pytest.mark.integration
class TestAgentPipeline:
    """Test the full agent pipeline working together"""

    @patch('src.agents_openrouter.validate_api_key')
    @patch('src.agents_openrouter.Config')
    def test_full_pipeline_success(self, mock_config, mock_validate, mock_vector_index,
                                  sample_user_query, sample_code_chunks):
        """Test successful execution of full agent pipeline"""
        # Setup mocks
        mock_validate.return_value = True
        mock_config.OPENROUTER_API_KEY = "test-key"
        mock_config.ORCHESTRATOR_MODEL = "test-model"
        mock_config.SYNTHESIZER_MODEL = "test-model"

        # Mock Orchestrator response
        with patch('src.agents_openrouter.OpenAILike') as mock_llm_class:
            mock_orchestrator_llm = Mock()
            mock_synthesizer_llm = Mock()

            # Orchestrator returns planning result
            orchestrator_response = Mock()
            orchestrator_response.__str__ = Mock(return_value='''
            {
              "search_queries": ["calculate total", "shopping cart"],
              "analysis_needed": false
            }
            ''')

            # Synthesizer returns final answer
            synthesizer_response = Mock()
            synthesizer_response.__str__ = Mock(return_value="The calculate_total function sums item prices.")

            # Configure LLM instances
            mock_llm_class.side_effect = [mock_orchestrator_llm, mock_synthesizer_llm]
            mock_orchestrator_llm.complete.return_value = orchestrator_response
            mock_synthesizer_llm.complete.return_value = synthesizer_response

            # Create agents
            indexer = IndexerAgent(mock_vector_index)
            orchestrator = OrchestratorAgent()
            synthesizer = SynthesizerAgent()

            # Execute pipeline
            plan = orchestrator.plan_query(sample_user_query)
            chunks = indexer.retrieve(plan["search_queries"][0])
            context = {
                "code_chunks": chunks,
                "analysis": {},
                "conversation_history": []
            }
            answer = synthesizer.synthesize(sample_user_query, context)

            # Verify results
            assert "search_queries" in plan
            assert len(plan["search_queries"]) > 0
            assert len(chunks) > 0
            assert "calculate_total" in answer or "function" in answer

    @patch('src.agents_openrouter.validate_api_key')
    def test_pipeline_error_handling(self, mock_validate, mock_vector_index, sample_user_query):
        """Test pipeline error handling when API key is missing"""
        mock_validate.return_value = False

        # Should fail at Orchestrator initialization
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY not set"):
            OrchestratorAgent()

        # Indexer should still work (doesn't need API key)
        indexer = IndexerAgent(mock_vector_index)
        chunks = indexer.retrieve("test query")
        assert len(chunks) > 0

    @pytest.mark.slow
    def test_pipeline_with_real_dependencies_disabled(self):
        """Placeholder for tests that would use real dependencies (disabled for CI)"""
        # This test would load real vector database and make real API calls
        # Currently disabled to avoid API costs and dependencies in CI
        pytest.skip("Real API integration tests disabled for CI")

        # Future implementation:
        # - Load actual vector database
        # - Make real API calls (with rate limiting)
        # - Test end-to-end functionality
        # - Validate response quality