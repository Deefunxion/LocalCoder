# agents_openrouter.py
# Multi-Agent System for Academicon Code Assistant - OpenRouter Version

from llama_index.llms.openai_like import OpenAILike
from llama_index.core import VectorStoreIndex
from typing import List, Dict, Any
import json
import sys
import logging
from pathlib import Path

# Add parent directory to path for config imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from utils import extract_json_from_response, validate_api_key, format_code_context, setup_logger

# Setup logger
logger = setup_logger(__name__)


class IndexerAgent:
    """Agent 1: Retrieves relevant code chunks from vector database"""

    def __init__(self, index: VectorStoreIndex):
        self.index = index
        logger.info("Agent 1 (Indexer) initialized")

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
        api_key = Config.OPENROUTER_API_KEY
        model_name = Config.GRAPH_ANALYST_MODEL
        
        if not validate_api_key(api_key):
            raise ValueError("OPENROUTER_API_KEY not set in .env file")
        
        self.llm = OpenAILike(
            model=model_name,
            api_key=api_key,
            api_base=Config.OPENROUTER_BASE_URL,
            is_chat_model=True,
            temperature=Config.GRAPH_ANALYST_TEMPERATURE,
            timeout=Config.GRAPH_ANALYST_TIMEOUT,
            max_retries=2
        )
        logger.info(f"Agent 2 (Graph Analyst) initialized with {model_name}")

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

            # Extract JSON using utility function
            result = extract_json_from_response(response_text)
            return result if result else {"functions": [], "dependencies": [], "relationships": []}
        except Exception as e:
            logger.warning(f"Graph analysis failed: {e}")
            return {"functions": [], "dependencies": [], "relationships": []}


class OrchestratorAgent:
    """Agent 3: Plans search strategy based on user query"""

    def __init__(self):
        api_key = Config.OPENROUTER_API_KEY
        model_name = Config.ORCHESTRATOR_MODEL
        
        if not validate_api_key(api_key):
            raise ValueError("OPENROUTER_API_KEY not set in .env file")
        
        self.llm = OpenAILike(
            model=model_name,
            api_key=api_key,
            api_base=Config.OPENROUTER_BASE_URL,
            is_chat_model=True,
            temperature=Config.ORCHESTRATOR_TEMPERATURE,
            timeout=Config.ORCHESTRATOR_TIMEOUT,
            max_retries=2
        )
        logger.info(f"Agent 3 (Orchestrator) initialized with {model_name}")

    def plan_query(self, user_query: str) -> Dict[str, Any]:
        """
        Plan search strategy for user query

        Args:
            user_query: User's question

        Returns:
            Dict with search_queries and analysis_needed flag
        """
        prompt = f"""Given this user query about a codebase:
"{user_query}"

Generate 1-2 search queries to find relevant code and determine if code relationship analysis is needed.

Return ONLY valid JSON:
{{
  "search_queries": ["query1", "query2"],
  "analysis_needed": true/false
}}"""

        try:
            response = self.llm.complete(prompt)
            response_text = str(response).strip()

            # Extract JSON using utility function
            result = extract_json_from_response(response_text)
            return result if result else {"search_queries": [user_query], "analysis_needed": False}
        except Exception as e:
            logger.warning(f"Planning failed: {e}, using fallback")
            return {
                "search_queries": [user_query],
                "analysis_needed": False
            }


class SynthesizerAgent:
    """Agent 4: Synthesizes final answer from retrieved context"""

    def __init__(self):
        api_key = Config.OPENROUTER_API_KEY
        model_name = Config.SYNTHESIZER_MODEL
        
        if not validate_api_key(api_key):
            raise ValueError("OPENROUTER_API_KEY not set in .env file")
        
        self.llm = OpenAILike(
            model=model_name,
            api_key=api_key,
            api_base=Config.OPENROUTER_BASE_URL,
            is_chat_model=True,
            temperature=Config.SYNTHESIZER_TEMPERATURE,
            timeout=Config.SYNTHESIZER_TIMEOUT,
            max_retries=2
        )
        logger.info(f"Agent 4 (Synthesizer) initialized with {model_name}")

    def synthesize(self, user_query: str, context: Dict[str, Any]) -> str:
        """
        Generate final answer based on retrieved code and analysis

        Args:
            user_query: Original user question
            context: Dict with code_chunks, optional analysis, and conversation_history

        Returns:
            Final answer string
        """
        code_chunks = context.get("code_chunks", [])
        analysis = context.get("analysis", {})
        conversation_history = context.get("conversation_history", [])

        if not code_chunks:
            return "I couldn't find relevant code in the database. Please try rephrasing your question."

        # Format code chunks using utility
        code_str = format_code_context(code_chunks, max_chunks=5)

        # Build prompt with conversation history
        prompt = f"""You are a code assistant. Answer the user's question based on the provided code chunks."""
        
        # Add conversation history if exists
        if conversation_history:
            prompt += "\n\nPrevious Conversation:\n"
            for i, exchange in enumerate(conversation_history[-3:], 1):
                prompt += f"\nQ{i}: {exchange['query']}\n"
                prompt += f"A{i}: {exchange['answer'][:200]}...\n"  # First 200 chars
            prompt += "\n(Use this context to maintain conversation continuity)\n"

        prompt += f"""

Current Question: {user_query}

Code Context:
{code_str}
"""

        if analysis and any(analysis.values()):
            prompt += f"\nCode Analysis:\n{json.dumps(analysis, indent=2)}\n"

        prompt += """
Instructions:
- Answer clearly and concisely
- Reference specific code when relevant
- If this is a follow-up question, use previous context
- If code doesn't fully answer the question, say so
- Use markdown formatting for code snippets

Answer:"""

        try:
            response = self.llm.complete(prompt)
            return str(response).strip()
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return f"Error generating answer: {str(e)}\n\nPlease try again with a simpler question."
