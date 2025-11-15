# web_ui.py
# Gradio Web UI for Academicon Code Assistant (DEPRECATED - Use web_ui_openrouter.py)
# This file is maintained for backwards compatibility with Ollama

import gradio as gr
from main import AcademiconAssistant
import os

# Set cache directories
os.environ['HF_HOME'] = 'D:/AI-Models/huggingface-moved'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:/AI-Models/huggingface-moved/hub'
os.environ['TRANSFORMERS_CACHE'] = 'D:/AI-Models/transformers'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = 'D:/AI-Models/embeddings'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Initialize assistant (done once at startup)
print("Initializing Academicon Assistant...")
assistant = AcademiconAssistant()
print("Ready!")

def query_assistant(message, history):
    """
    Process user query and return answer

    Args:
        message: User's question
        history: Chat history

    Returns:
        Updated history with new message and answer
    """
    if not message.strip():
        return history

    import time
    start_time = time.time()
    
    print(f"\n[{time.strftime('%H:%M:%S')}] User query: {message}")

    try:
        # Get answer from assistant with timeout protection
        print(f"   [DEBUG] Starting query processing...")
        answer = assistant.query(message, verbose=False)
        
        elapsed = time.time() - start_time
        print(f"   [OK] Query completed in {elapsed:.1f}s")

        # Append to history in correct format: [(user_msg, bot_msg), ...]
        history.append((message, answer))
        return history
        
    except TimeoutError as e:
        elapsed = time.time() - start_time
        print(f"   [ERROR] Query timeout after {elapsed:.1f}s")
        error_msg = f"⏱️ Query timed out after {elapsed:.0f} seconds.\n\nTry:\n- A simpler question\n- More specific terms\n- Restart Ollama: `taskkill /IM ollama.exe /F && ollama serve`"
        history.append((message, error_msg))
        return history
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   [ERROR] Query failed after {elapsed:.1f}s: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        history.append((message, f"❌ Error: {str(e)}"))
        return history


# Create Gradio interface
with gr.Blocks(title="Academicon Code Assistant") as demo:
    gr.Markdown("""
    # 🤖 Academicon Code Assistant

    Ask questions about the Academicon codebase. The assistant uses 4 AI agents:
    - **Orchestrator**: Plans the search strategy
    - **Indexer**: Retrieves relevant code
    - **Graph Analyst**: Analyzes relationships
    - **Synthesizer**: Generates the final answer

    **Examples:**
    - What is the CIP service?
    - How does authentication work?
    - Show me the database models
    - Explain the task queue implementation
    """)

    chatbot = gr.Chatbot(
        height=500,
        placeholder="Ask me anything about the Academicon codebase...",
        show_label=False
    )

    msg = gr.Textbox(
        placeholder="Type your question here...",
        show_label=False,
        scale=9
    )

    with gr.Row():
        submit = gr.Button("Send", variant="primary", scale=1)
        clear = gr.Button("Clear", scale=1)

    # Example questions
    gr.Examples(
        examples=[
            "What is the CIP service in Academicon?",
            "How does user authentication work?",
            "What API endpoints are available?",
            "Show me the database schema for publications",
            "Explain the task queue implementation"
        ],
        inputs=msg
    )

    # Event handlers
    def submit_and_clear(message, history):
        new_history = query_assistant(message, history)
        return new_history, ""  # Return updated history and clear input

    msg.submit(submit_and_clear, [msg, chatbot], [chatbot, msg])
    submit.click(submit_and_clear, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: ([], ""), None, [chatbot, msg], queue=False)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",  # localhost only
        server_port=None,  # Auto-find available port
        share=False,  # Don't create public link
        show_error=True
    )
