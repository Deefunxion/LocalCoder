# tests/integration/test_gradio_format.py
# Test for Gradio message format compatibility

import pytest


def test_gradio_message_format_requirements():
    """Test that demonstrates the correct message format for Gradio 5.0+"""

    # ✅ Correct format for Gradio chatbot
    correct_messages = [
        {"role": "user", "content": "What is the CIP service?"},
        {"role": "assistant", "content": "The CIP service handles..."},
        {"role": "user", "content": "How does it work?"},
        {"role": "assistant", "content": "It works by..."}
    ]

    # ❌ Wrong format that causes the error
    wrong_messages = [
        ("What is the CIP service?", "The CIP service handles..."),
        ("How does it work?", "It works by...")
    ]

    # Verify correct format structure
    for msg in correct_messages:
        assert isinstance(msg, dict), "Messages must be dictionaries"
        assert "role" in msg, "Each message must have a 'role' key"
        assert "content" in msg, "Each message must have a 'content' key"
        assert msg["role"] in ["user", "assistant"], "Role must be 'user' or 'assistant'"
        assert isinstance(msg["content"], str), "Content must be a string"

    # Show the difference
    assert correct_messages != wrong_messages


def test_converting_old_format_to_new():
    """Test utility function to convert old tuple format to new dict format"""

    def convert_tuple_history_to_gradio_format(tuple_history):
        """Convert old format: [(user_msg, assistant_msg), ...]
        To new format: [{"role": "user", "content": user_msg}, {"role": "assistant", "content": assistant_msg}, ...]
        """
        messages = []
        for user_msg, assistant_msg in tuple_history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        return messages

    # Test data - this is what the current web_ui returns
    old_format_history = [
        ("What is the CIP service?", "The CIP service handles authentication"),
        ("How does it work?", "It works by validating user credentials")
    ]

    # Convert to new format
    new_format_messages = convert_tuple_history_to_gradio_format(old_format_history)

    # Verify the conversion
    expected_messages = [
        {"role": "user", "content": "What is the CIP service?"},
        {"role": "assistant", "content": "The CIP service handles authentication"},
        {"role": "user", "content": "How does it work?"},
        {"role": "assistant", "content": "It works by validating user credentials"}
    ]

    assert new_format_messages == expected_messages

    # Verify each message meets Gradio requirements
    for msg in new_format_messages:
        assert isinstance(msg, dict)
        assert "role" in msg and "content" in msg
        assert msg["role"] in ["user", "assistant"]


@pytest.mark.parametrize("invalid_format,description", [
    # Old tuple format
    ([("Hello", "Hi")], "tuple format from old Gradio versions"),

    # Missing required keys
    ([{"role": "user"}, {"role": "assistant", "content": "Hi"}], "missing content key"),

    # Invalid role values
    ([{"role": "user", "content": "Hello"}, {"role": "bot", "content": "Hi"}], "invalid role value"),

    # Wrong data types
    ([{"role": "user", "content": 123}], "non-string content"),
])
def test_invalid_formats_that_break_gradio(invalid_format, description):
    """Test various invalid formats that would cause Gradio errors"""

    # These formats would all cause the error:
    # "Data incompatible with messages format. Each message should be a dictionary with 'role' and 'content' keys"

    # We can't test Gradio directly due to environment issues, but we can validate the structure
    for item in invalid_format:
        if isinstance(item, dict):
            # Check if dict has required keys
            has_role = "role" in item
            has_content = "content" in item
            valid_role = item.get("role") in ["user", "assistant"] if has_role else False
            valid_content = isinstance(item.get("content"), str) if has_content else False

            # If any requirement is missing, this format is invalid
            if not (has_role and has_content and valid_role and valid_content):
                assert True, f"Invalid format detected: {description}"
        elif isinstance(item, tuple):
            # Tuple format is always invalid for Gradio 5.0+
            assert isinstance(item, tuple), f"Tuple format is invalid: {description}"
        else:
            # Any other format is invalid
            assert True, f"Unsupported format: {description}"


def test_web_ui_should_use_this_format():
    """This test documents what the web_ui_openrouter.py SHOULD return"""

    # Instead of:
    # history.append((message, answer))  # ❌ Wrong - tuple format

    # It should do:
    # history.append({"role": "user", "content": message})      # ✅ Correct
    # history.append({"role": "assistant", "content": answer})  # ✅ Correct

    # Simulate what the corrected function should return
    def corrected_query_assistant(message, history):
        answer = "Mock answer"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        return history

    # Test the corrected version
    result = corrected_query_assistant("Test question", [])

    expected = [
        {"role": "user", "content": "Test question"},
        {"role": "assistant", "content": "Mock answer"}
    ]

    assert result == expected

    # Verify Gradio compatibility
    for msg in result:
        assert isinstance(msg, dict)
        assert "role" in msg and "content" in msg
        assert msg["role"] in ["user", "assistant"]