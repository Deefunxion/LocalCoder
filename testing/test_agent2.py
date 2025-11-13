from llama_index.llms.ollama import Ollama

# Initialize Graph Analyst
print("Initializing Graph Analyst...")
graph_analyst = Ollama(
    model="qwen2.5-coder:14b",  # Using available qwen model
    base_url="http://localhost:11434",
    temperature=0.1
)

# Test structured output
test_prompt = """
Extract function names from this code as JSON array:

def login(username, password):
    pass

def logout(session_id):
    pass

Return only JSON: {"functions": ["name1", "name2"]}
"""

response = graph_analyst.complete(test_prompt)
print(f"[OK] Agent 2 (Graph Analyst) working!")
print(f"   Response: {response}")
