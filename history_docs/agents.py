# agents.py
# Multi-Agent System for Academicon Code Assistant

from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex
from typing import List, Dict, Any
import json


class IndexerAgent:
    """Agent 1: Retrieves relevant code chunks from vector database"""

    def __init__(self, index: VectorStoreIndex):
        self.index = index
        print("[OK] Agent 1 (Indexer) initialized")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant code chunks for a query

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of dicts with text, score, and metadata
        """
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        results = []
        for node in nodes:
            results.append({
                "text": node.node.text,
                "score": float(node.score) if node.score else 0.0,
                "metadata": node.node.metadata
            })

        return results


class GraphAnalystAgent:
    """Agent 2: Analyzes code relationships and structure"""

    def __init__(self):
        self.llm = Ollama(
            model="qwen2.5-coder:14b",
            base_url="http://localhost:11434",
            temperature=0.1,
            request_timeout=300.0
        )
        print("[OK] Agent 2 (Graph Analyst) initialized")

    def analyze_relationships(self, code_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze code chunks to extract functions, dependencies, and relationships

        Args:
            code_chunks: List of code chunks from Indexer

        Returns:
            Dict with functions, dependencies, and relationships
        """
        if not code_chunks:
            return {"functions": [], "dependencies": [], "relationships": []}

        # Limit context to avoid overwhelming the model
        limited_chunks = [c['text'][:800] for c in code_chunks[:3]]

        prompt = f"""Analyze these code chunks and extract:
1. Main function/class names
2. Import dependencies
3. Relationships between components

Code chunks:
{json.dumps(limited_chunks, indent=2)}

Return ONLY valid JSON with this structure:
{{
  "functions": ["function1", "function2"],
  "dependencies": ["import1", "import2"],
  "relationships": ["func1 calls func2"]
}}"""

        try:
            response = self.llm.complete(prompt)
            response_text = str(response).strip()

            # Try to extract JSON from markdown code blocks
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            return json.loads(response_text)
        except Exception as e:
            print(f"   [WARN] Graph analysis failed: {e}")
            return {"functions": [], "dependencies": [], "relationships": []}


class OrchestratorAgent:
    """Agent 3: Plans search strategy and coordinates other agents"""

    def __init__(self):
        self.llm = Ollama(
            model="qwen2.5-coder:14b",
            base_url="http://localhost:11434",
            temperature=0.3,
            request_timeout=120.0  # Reduced from 300s to 120s
        )
        print("[OK] Agent 3 (Orchestrator) initialized")

    def plan_query(self, user_query: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Create a search plan for the user query

        Args:
            user_query: User's question
            timeout: Max seconds to wait for response

        Returns:
            Dict with search_queries, analysis_needed, expected_file_types
        """
        # Simplified prompt - shorter and more direct
        prompt = f"""Given query: "{user_query}"

Return JSON:
{{"search_queries": ["query1"], "analysis_needed": false, "expected_file_types": [".py"]}}

Be concise."""

        try:
            import time
            start = time.time()
            print(f"   [DEBUG] Orchestrator starting... (timeout: {timeout}s)")
            
            response = self.llm.complete(prompt)
            elapsed = time.time() - start
            print(f"   [DEBUG] Orchestrator completed in {elapsed:.1f}s")
            
            response_text = str(response).strip()

            # Extract JSON from markdown
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            return json.loads(response_text)
        except Exception as e:
            print(f"   [WARN] Orchestration failed: {e}")
            # Fallback: skip orchestration, just search directly
            return {
                "search_queries": [user_query],
                "analysis_needed": False,
                "expected_file_types": []
            }


class SynthesizerAgent:
    """Agent 4: Generates final answer from all collected context"""

    def __init__(self):
        self.llm = Ollama(
            model="qwen2.5-coder:14b",
            base_url="http://localhost:11434",
            temperature=0.7,
            request_timeout=180.0  # Reduced from 300s to 180s
        )
        print("[OK] Agent 4 (Synthesizer) initialized")

    def synthesize(self, user_query: str, context: Dict[str, Any], timeout: int = 120) -> str:
        """
        Generate final answer based on retrieved context

        Args:
            user_query: Original user question
            context: Dict with code_chunks and analysis
            timeout: Max seconds to wait

        Returns:
            Final answer string
        """
        import time
        start = time.time()
        print(f"   [DEBUG] Synthesizer starting... (timeout: {timeout}s)")
        
        code_chunks = context.get('code_chunks', [])[:2]  # Reduced from 3 to 2
        analysis = context.get('analysis', {})

        # Build context summary - MUCH shorter
        code_context = "\n\n".join([
            f"[Chunk {i+1}]\n{c['text'][:500]}"  # Reduced from 800 to 500 chars
            for i, c in enumerate(code_chunks)
        ])

        # Shorter prompt
        prompt = f"""Question: {user_query}

Code:
{code_context}

Answer concisely based on this code."""

        try:
            response = self.llm.complete(prompt)
            elapsed = time.time() - start
            print(f"   [DEBUG] Synthesizer completed in {elapsed:.1f}s")
            return str(response)
        except Exception as e:
            elapsed = time.time() - start
            print(f"   [ERROR] Synthesizer failed after {elapsed:.1f}s: {e}")
            return f"Error generating answer: {e}"
