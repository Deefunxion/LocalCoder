# main_openrouter.py
# Academicon Multi-Agent Code Assistant - OpenRouter Version

import os
import chromadb
import torch
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from agents_openrouter import IndexerAgent, GraphAnalystAgent, OrchestratorAgent, SynthesizerAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set cache directories to D: drive
os.environ['HF_HOME'] = os.getenv('HF_HOME', 'D:/AI-Models/huggingface-moved')
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:/AI-Models/huggingface-moved/hub'
os.environ['TRANSFORMERS_CACHE'] = os.getenv('TRANSFORMERS_CACHE', 'D:/AI-Models/transformers')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.getenv('SENTENCE_TRANSFORMERS_HOME', 'D:/AI-Models/embeddings')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


class AcademiconAssistant:
    """Multi-Agent Code Assistant for Academicon Project - OpenRouter Version"""

    def __init__(self, db_path="./academicon_chroma_db"):
        print("="*60)
        print("Academicon Code Assistant (OpenRouter)")
        print("="*60)
        
        # Initialize conversation history
        self.conversation_history = []

        # Load embedding model with GPU support
        print("\n[1/5] Loading embedding model...")

        # Check for GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 128 if device == "cuda" else 32

        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   [GPU] {gpu_name}")

        embed_model = HuggingFaceEmbedding(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME', 'D:/AI-Models/embeddings'),
            trust_remote_code=True,
            embed_batch_size=batch_size,
            device=device
        )
        Settings.embed_model = embed_model
        print(f"   [OK] Embedding model loaded ({device.upper()})")

        # Load existing vector index
        print("\n[2/5] Loading vector index from ChromaDB...")
        chroma_client = chromadb.PersistentClient(path=db_path)
        chroma_collection = chroma_client.get_collection("academicon_code")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        self.index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model
        )
        print(f"   [OK] Loaded index from {db_path}")

        # Initialize all 4 agents
        print("\n[3/5] Initializing agents with OpenRouter...")
        self.indexer = IndexerAgent(self.index)
        self.graph_analyst = GraphAnalystAgent()
        self.orchestrator = OrchestratorAgent()
        self.synthesizer = SynthesizerAgent()

        print("\n[4/5] Running health check...")
        self._health_check()

        print("\n[5/5] Ready!")
        print("="*60)

    def _health_check(self):
        """Quick test to ensure all agents are working"""
        try:
            # Test retrieval
            test_results = self.indexer.retrieve("authentication", top_k=1)
            print(f"   [OK] Indexer: Retrieved {len(test_results)} results")

            # Other agents will be tested during actual queries
            print("   [OK] All agents ready")
        except Exception as e:
            print(f"   [ERROR] Health check failed: {e}")

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("   [OK] Conversation history cleared")

    def query(self, user_query: str, verbose: bool = True) -> str:
        """
        Process a user query through the multi-agent pipeline

        Args:
            user_query: User's question about the codebase
            verbose: Print progress messages

        Returns:
            Final answer string
        """
        import time
        start_time = time.time()
        
        if verbose:
            print("\n" + "="*60)
            print(f"Query: {user_query}")
            print("="*60)

        try:
            # Step 1: Orchestrator plans the search
            if verbose:
                print("\n[1/4] Orchestrator: Planning search strategy...")

            plan = self.orchestrator.plan_query(user_query)
            
            elapsed = time.time() - start_time
            if verbose:
                print(f"   [OK] Planning complete in {elapsed:.1f}s")
                print(f"   Search queries: {plan.get('search_queries', [])}")
                print(f"   Analysis needed: {plan.get('analysis_needed', False)}")

            # Step 2: Indexer retrieves relevant code
            if verbose:
                print("\n[2/4] Indexer: Retrieving relevant code...")

            all_chunks = []
            for search_query in plan.get('search_queries', [user_query])[:2]:
                chunks = self.indexer.retrieve(search_query, top_k=3)
                all_chunks.extend(chunks)

            # Remove duplicates based on text content
            seen_texts = set()
            unique_chunks = []
            for chunk in all_chunks:
                text_hash = hash(chunk['text'][:200])
                if text_hash not in seen_texts:
                    seen_texts.add(text_hash)
                    unique_chunks.append(chunk)

            if verbose:
                print(f"   [OK] Retrieved {len(unique_chunks)} unique code chunks")

            # Step 3: SKIP Graph Analyst - it's slow and optional
            analysis = {}
            if verbose:
                print("\n[3/4] Graph Analyst: Skipped (disabled for speed)")

            # Step 4: Synthesizer creates final answer
            if verbose:
                print("\n[4/4] Synthesizer: Generating answer...")

            context = {
                "code_chunks": unique_chunks[:5],
                "analysis": analysis,
                "conversation_history": self.conversation_history[-3:]  # Last 3 exchanges
            }
            answer = self.synthesizer.synthesize(user_query, context)
            
            # Add to conversation history
            self.conversation_history.append({
                "query": user_query,
                "answer": answer
            })
            
            total_elapsed = time.time() - start_time
            if verbose:
                print(f"\n   [OK] Total query time: {total_elapsed:.1f}s")
                print(f"   [COST] ~${total_elapsed * 0.0001:.4f} (estimated)")
                if len(self.conversation_history) > 1:
                    print(f"   [HISTORY] {len(self.conversation_history)} exchanges in memory")
                print("\n" + "="*60)
                print("Answer:")
                print("="*60)

            return answer
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n   [ERROR] Query failed after {elapsed:.1f}s: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"


def main():
    """CLI interface for the assistant"""
    try:
        assistant = AcademiconAssistant()
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize assistant: {e}")
        print("\nMake sure:")
        print("1. You've run index_academicon.py first")
        print("2. You have a .env file with OPENROUTER_API_KEY")
        return

    print("\nType your questions about the Academicon codebase.")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            query = input("\nYou: ").strip()

            if query.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break

            if not query:
                continue

            answer = assistant.query(query, verbose=True)
            print(f"\n{answer}\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    main()
