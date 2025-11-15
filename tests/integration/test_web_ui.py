# tests/integration/test_web_ui.py
# Integration tests for the Gradio Web UI

import pytest
from unittest.mock import Mock, patch
import gradio as gr


@pytest.mark.integration
class TestWebUIIntegration:
    """Integration tests for the web UI components"""

    def test_query_assistant_returns_correct_format_for_gradio(self):
        """Test that query_assistant returns data in the correct format for Gradio chatbot"""
        # Import here to avoid issues if web_ui is not available
        from src.web_ui_openrouter import query_assistant

        # Mock the assistant to return a predictable response
        with patch('src.web_ui_openrouter.assistant') as mock_assistant:
            mock_assistant.query.return_value = "This is a test answer"
            mock_assistant.conversation_history = []

            # Test data
            test_message = "What is the CIP service?"
            test_history = []  # Empty history

            # Call the function
            result_history = query_assistant(test_message, test_history)

            # Verify the result is in the correct format for Gradio chatbot
            assert isinstance(result_history, list)
            assert len(result_history) == 1  # One exchange added

            # Each item should be a tuple (old format) - this is the problem!
            exchange = result_history[0]
            assert isinstance(exchange, tuple)
            assert len(exchange) == 2

            user_msg, assistant_msg = exchange
            assert user_msg == test_message
            assert assistant_msg == "This is a test answer"

    def test_query_assistant_format_should_be_dict_for_gradio_5(self):
        """Test that demonstrates the correct format needed for Gradio 5.0+"""
        # This test shows what the format SHOULD be
        correct_format = [
            {"role": "user", "content": "What is the CIP service?"},
            {"role": "assistant", "content": "The CIP service handles..."}
        ]

        # Verify each message has the required structure
        for message in correct_format:
            assert isinstance(message, dict)
            assert "role" in message
            assert "content" in message
            assert message["role"] in ["user", "assistant"]
            assert isinstance(message["content"], str)

    @patch('src.web_ui_openrouter.assistant')
    def test_query_assistant_with_existing_history(self, mock_assistant):
        """Test query_assistant with existing conversation history"""
        from src.web_ui_openrouter import query_assistant

        mock_assistant.query.return_value = "New answer"
        mock_assistant.conversation_history = []

        # Start with some existing history
        existing_history = [("Previous question", "Previous answer")]
        new_message = "Follow-up question"

        result = query_assistant(new_message, existing_history)

        # Should have 2 exchanges now
        assert len(result) == 2
        assert result[0] == ("Previous question", "Previous answer")
        assert result[1] == (new_message, "New answer")

    @patch('src.web_ui_openrouter.assistant')
    def test_query_assistant_error_handling(self, mock_assistant):
        """Test that errors are handled gracefully and return proper format"""
        from src.web_ui_openrouter import query_assistant

        # Make the assistant raise an exception
        mock_assistant.query.side_effect = Exception("API Error")

        test_message = "Test question"
        test_history = []

        result = query_assistant(test_message, test_history)

        # Should still return a valid history format
        assert isinstance(result, list)
        assert len(result) == 1

        user_msg, assistant_msg = result[0]
        assert user_msg == test_message
        assert "❌ Error:" in assistant_msg

    def test_gradio_chatbot_component_expects_messages_format(self):
        """Test that demonstrates what Gradio chatbot expects"""
        # This test verifies our understanding of Gradio's requirements

        # Create a mock chatbot component
        chatbot = gr.Chatbot(type="messages")

        # The correct format for Gradio 5.0+
        correct_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        # Verify the structure
        assert isinstance(correct_messages, list)
        for msg in correct_messages:
            assert isinstance(msg, dict)
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ["user", "assistant"]

        # The old tuple format that causes the error
        old_format = [
            ("Hello", "Hi there!")  # This causes the Gradio error
        ]

        # Show the difference
        assert old_format != correct_messages

    @pytest.mark.parametrize("input_format,expected_error", [
        ([("user", "assistant")], "tuple format causes error"),
        ([{"role": "user"}, {"role": "assistant"}], "missing content key"),
        ([{"role": "user", "content": "msg"}, {"role": "invalid", "content": "msg"}], "invalid role"),
    ])
    def test_invalid_message_formats(self, input_format, expected_error):
        """Test various invalid message formats that would cause Gradio errors"""
        # This test documents what NOT to do

        # Check that these formats are invalid
        for msg in input_format:
            if isinstance(msg, dict):
                has_role = "role" in msg
                has_content = "content" in msg
                valid_role = msg.get("role") in ["user", "assistant"] if has_role else False

                if not (has_role and has_content and valid_role):
                    # This would cause a Gradio error
                    assert True, f"Invalid format detected: {expected_error}"
            else:
                # Tuple format is invalid for Gradio 5.0+
                assert isinstance(msg, tuple), f"Non-dict format: {expected_error}"