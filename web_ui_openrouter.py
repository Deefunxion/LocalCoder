# web_ui_openrouter.py
# Gradio Web UI for Academicon Code Assistant - OpenRouter Version

import gradio as gr
from main_openrouter import AcademiconAssistant
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set cache directories
os.environ['HF_HOME'] = os.getenv('HF_HOME', 'D:/AI-Models/huggingface-moved')
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:/AI-Models/huggingface-moved/hub'
os.environ['TRANSFORMERS_CACHE'] = os.getenv('TRANSFORMERS_CACHE', 'D:/AI-Models/transformers')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.getenv('SENTENCE_TRANSFORMERS_HOME', 'D:/AI-Models/embeddings')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Initialize assistant (done once at startup)
print("Initializing Academicon Assistant with OpenRouter...")
try:
    assistant = AcademiconAssistant()
    print("Ready!")
except Exception as e:
    print(f"[ERROR] Failed to initialize: {e}")
    print("\nMake sure:")
    print("1. You have OPENROUTER_API_KEY set in .env")
    print("2. You have credits at https://openrouter.ai/credits")
    raise

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
        # Get answer from assistant
        print(f"   [DEBUG] Starting query processing with OpenRouter...")
        answer = assistant.query(message, verbose=False)
        
        elapsed = time.time() - start_time
        print(f"   [OK] Query completed in {elapsed:.1f}s")
        print(f"   [COST] Estimated: ~${elapsed * 0.0001:.4f}")

        # Append to history
        history.append((message, answer))
        return history
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   [ERROR] Query failed after {elapsed:.1f}s: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_msg = f"❌ Error: {str(e)}\n\n"
        if "401" in str(e) or "Unauthorized" in str(e):
            error_msg += "Check your OPENROUTER_API_KEY in .env file"
        elif "402" in str(e) or "credits" in str(e).lower():
            error_msg += "Add credits at https://openrouter.ai/credits"
        elif "rate limit" in str(e).lower():
            error_msg += "Rate limited. Wait a moment and try again."
        else:
            error_msg += "Please try again or check the logs."
        
        history.append((message, error_msg))
        return history


# Get model names for display
orchestrator_model = os.getenv('ORCHESTRATOR_MODEL', 'unknown')
graph_analyst_model = os.getenv('GRAPH_ANALYST_MODEL', 'unknown')
synthesizer_model = os.getenv('SYNTHESIZER_MODEL', 'unknown')

# Create Gradio interface
with gr.Blocks(title="Academicon Code Assistant (OpenRouter)", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # 🤖 Academicon Code Assistant (OpenRouter)

    **Multi-Model Setup (All FREE!):**
    - 🎯 **Orchestrator**: `{orchestrator_model}` (Planning)
    - 🔍 **Graph Analyst**: `{graph_analyst_model}` (Code Analysis) - Disabled for speed
    - ✍️ **Synthesizer**: `{synthesizer_model}` (Answer Generation)
    
    Ask questions about the Academicon codebase. Total query time: **~60-80s** 🚀
    - **Orchestrator**: Plans the search strategy ⚡ (~2-3s)
    - **Indexer**: Retrieves relevant code 📚
    - **Graph Analyst**: Analyzes relationships (disabled for speed)
    - **Synthesizer**: Generates the final answer ✍️ (~5-8s)

    **Total query time: ~10-15 seconds** 🚀 (vs 250s+ with local Ollama!)

    **Examples:**
    - What is the CIP service?
    - How does authentication work?
    - Show me the database models
    - Explain the task queue implementation
    """)

    chatbot = gr.Chatbot(
        height=500,
        placeholder="Ask me anything about the Academicon codebase...",
        show_label=False,
        type="tuples"  # Explicitly set to avoid deprecation warning
    )

    msg = gr.Textbox(
        placeholder="Type your question here...",
        show_label=False,
        scale=9
    )

    with gr.Row():
        submit = gr.Button("Send", variant="primary", scale=1)
        clear = gr.Button("Clear Chat", scale=1)
        clear_history = gr.Button("Clear Memory", scale=1, variant="secondary")

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

    gr.Markdown(f"""
    ---
    💡 **Tips:**
    - Ask follow-up questions - conversation history is maintained
    - "Clear Chat" = Clear visual chat only
    - "Clear Memory" = Clear conversation history (start fresh)
    - More specific questions = better answers
    - Queries cost ~$0.01-0.02 each
    - Check credits: [OpenRouter Dashboard](https://openrouter.ai/credits)
    
    ⚙️ **Change model:** Edit agent models in `.env` file
    """)

    # Event handlers
    def submit_and_clear(message, history):
        new_history = query_assistant(message, history)
        return new_history, ""  # Return updated history and clear input

    def clear_memory():
        assistant.clear_history()
        return [], ""  # Clear chat and input

    msg.submit(submit_and_clear, [msg, chatbot], [chatbot, msg])
    submit.click(submit_and_clear, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: ([], ""), None, [chatbot, msg], queue=False)
    clear_history.click(clear_memory, None, [chatbot, msg], queue=False)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",  # localhost only
        server_port=None,  # Auto-find available port
        share=False,  # Don't create public link
        show_error=True
    )
