from llama_index.llms.ollama import Ollama

# Initialize Synthesizer
print("Initializing Synthesizer...")
synthesizer = Ollama(
    model="qwen2.5-coder:14b",  # Using the model you already have
    base_url="http://localhost:11434",
    temperature=0.7,
    request_timeout=300.0  # Increased timeout
)

# Test code explanation
test_prompt = """
Explain this code in simple terms:

def calculate_cip_score(publications, citations):
    weighted_score = sum([p.impact * c.count for p, c in zip(publications, citations)])
    return weighted_score / len(publications) if publications else 0

Focus on what it does and why.
"""

response = synthesizer.complete(test_prompt)
print(f"[OK] Agent 4 (Synthesizer) working!")
print(f"   Explanation:\n{response}")
