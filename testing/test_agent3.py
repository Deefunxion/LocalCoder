from llama_index.llms.ollama import Ollama

# Initialize Orchestrator
print("Initializing Orchestrator...")
orchestrator = Ollama(
    model="qwen2.5-coder:14b",  # Using available qwen model
    base_url="http://localhost:11434",
    temperature=0.3,
    request_timeout=300.0  # Increased timeout
)

# Test planning ability
test_prompt = """
You are a code analysis orchestrator. Break down this task into steps:

"Find all authentication-related functions in the codebase and explain their relationships."

Provide a numbered step-by-step plan.
"""

response = orchestrator.complete(test_prompt)
print(f"[OK] Agent 3 (Orchestrator) working!")
print(f"   Plan:\n{response}")
