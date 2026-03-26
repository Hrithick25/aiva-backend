"""
Interactive Query Tester for FAISS Vector Retrieval
==================================================

Test vector embeddings and retrieval responses with custom input queries.
Provides detailed analysis of embedding similarity, chunk retrieval, and performance metrics.

Usage:
    python query_tester.py
"""

import os
import sys
import time
import numpy as np
import pickle
import faiss
from typing import List, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from rag_faiss.config import (
        FAISS_INDEX_PATH,
        INDEX_MAP_PATH,
        PICKLES_DIR,
        TOP_K,
        HNSW_EF_SEARCH,
    )
    from rag_faiss.embedder import embed_query as _embed_query_fn
except ImportError:
    from config import (
        FAISS_INDEX_PATH,
        INDEX_MAP_PATH,
        PICKLES_DIR,
        TOP_K,
        HNSW_EF_SEARCH,
    )
    from embedder import embed_query as _embed_query_fn

class QueryTester:
    def __init__(self):
        """Initialize the Query Tester with FAISS index and local BGE embedder."""
        print("🚀 Initializing Query Tester (local BGE embedder — no API keys needed)...")
        self.faiss_index = None
        self.index_map   = None
        self.pickle_cache = {}
        self._load_index()
        print(f"✅ FAISS index loaded: {self.faiss_index.ntotal} vectors")
        print(f"✅ Index map: {len(self.index_map)} mappings")
        print("-" * 60)

    def _load_index(self):
        """Load FAISS index and index mapping."""
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")
            
        if not os.path.exists(INDEX_MAP_PATH):
            raise FileNotFoundError(f"Index map not found at {INDEX_MAP_PATH}")
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        self.faiss_index.hnsw.efSearch = HNSW_EF_SEARCH
        
        # Load index mapping
        with open(INDEX_MAP_PATH, "rb") as f:
            self.index_map = pickle.load(f)

    def _embed_query(self, text: str) -> np.ndarray:
        """Embed using local BGE model — no API, no rate limits."""
        return _embed_query_fn(text)

    def _load_pickle(self, pickle_filename: str) -> List[str]:
        """Load and cache pickle file contents."""
        if pickle_filename not in self.pickle_cache:
            path = os.path.join(PICKLES_DIR, pickle_filename)
            with open(path, "rb") as f:
                self.pickle_cache[pickle_filename] = pickle.load(f)
        return self.pickle_cache[pickle_filename]

    def query_detailed(self, query: str, top_k: int = None) -> Dict:
        """
        Perform detailed query analysis with comprehensive results.
        
        Returns:
            Dict containing embedding info, similarity scores, chunks, timing, etc.
        """
        if top_k is None:
            top_k = TOP_K
            
        print(f"\n🔍 Query: '{query}'")
        print(f"📊 Retrieving top {top_k} results...")
        
        # Timing - Embedding generation
        start_embed = time.perf_counter()
        query_vec = self._embed_query(query)
        embed_time = (time.perf_counter() - start_embed) * 1000
        
        # Timing - FAISS search
        start_search = time.perf_counter()
        distances, ids = self.faiss_index.search(query_vec, top_k)
        search_time = (time.perf_counter() - start_search) * 1000
        
        # Timing - Chunk retrieval
        start_chunks = time.perf_counter()
        chunks = []
        sources = []
        detailed_results = []
        
        for i, faiss_id in enumerate(ids[0]):
            if faiss_id == -1:
                continue
                
            # Get chunk info
            pickle_filename, chunk_idx = self.index_map[int(faiss_id)]
            pickle_chunks = self._load_pickle(pickle_filename)
            chunk_text = pickle_chunks[chunk_idx]
            chunks.append(chunk_text)
            
            # Track source
            source_txt = pickle_filename.replace(".pkl", ".txt")
            if source_txt not in sources:
                sources.append(source_txt)
            
            # Detailed result info
            similarity_score = 1 - distances[0][i]  # Convert distance to similarity
            detailed_results.append({
                "rank": i + 1,
                "faiss_id": int(faiss_id), 
                "distance": float(distances[0][i]),
                "similarity": float(similarity_score),
                "source": source_txt,
                "chunk_idx": chunk_idx,
                "chunk_preview": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
            })
        
        chunks_time = (time.perf_counter() - start_chunks) * 1000
        total_time = embed_time + search_time + chunks_time
        
        return {
            "query": query,
            "embedding_dim": query_vec.shape[1],
            "total_results": len(chunks),
            "context": "\n\n".join(chunks),
            "sources": sources,
            "detailed_results": detailed_results,
            "timing": {
                "embedding_ms": embed_time,
                "search_ms": search_time, 
                "chunks_ms": chunks_time,
                "total_ms": total_time
            }
        }

    def print_results(self, results: Dict):
        """Print formatted query results."""
        print(f"\n📈 Results Summary:")
        print(f"   • Total chunks found: {results['total_results']}")
        print(f"   • Sources: {', '.join(results['sources'])}")
        print(f"   • Embedding dimension: {results['embedding_dim']}")
        
        print(f"\n⏱️  Performance:")
        timing = results['timing']
        print(f"   • Embedding: {timing['embedding_ms']:.1f}ms")
        print(f"   • FAISS search: {timing['search_ms']:.1f}ms") 
        print(f"   • Chunk loading: {timing['chunks_ms']:.1f}ms")
        print(f"   • Total: {timing['total_ms']:.1f}ms")
        
        print(f"\n🎯 Top Results:")
        for result in results['detailed_results'][:3]:  # Show top 3
            print(f"   {result['rank']}. Similarity: {result['similarity']:.3f} | Source: {result['source']}")
            print(f"      Preview: {result['chunk_preview']}")
            print()
        
        print(f"\n📝 Full Context ({len(results['context'])} chars):")
        print(f"   {results['context'][:500]}{'...' if len(results['context']) > 500 else ''}")

    def interactive_mode(self):
        """Run interactive query testing."""
        print("🎮 Interactive Mode - Enter queries to test retrieval")
        print("Commands: 'quit' to exit, 'batch' for batch testing")
        print("-" * 60)
        
        while True:
            try:
                query = input("\n💬 Enter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif query.lower() == 'batch':
                    self.batch_test_mode()
                    continue
                elif not query:
                    continue
                
                # Process query
                results = self.query_detailed(query)
                self.print_results(results)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def batch_test_mode(self):
        """Test with predefined batch queries."""
        test_queries = [
            "What are the CSE department cutoff marks?",
            "Tell me about student achievements and awards", 
            "What facilities are available in the hostel?",
            "What is the fee structure for different courses?",
            "Which companies visit for placements?",
            "What are the eligibility criteria for admission?",
            "Tell me about the campus infrastructure",
            "What research opportunities are available?"
        ]
        
        print(f"\n🔄 Running batch test with {len(test_queries)} queries...")
        print("-" * 60)
        
        total_time = 0
        for i, query in enumerate(test_queries, 1):
            print(f"\n[{i}/{len(test_queries)}] Testing: '{query}'")
            
            results = self.query_detailed(query, top_k=3)
            total_time += results['timing']['total_ms'] 
            
            # Brief results
            print(f"   ⚡ {results['timing']['total_ms']:.1f}ms | "
                  f"{results['total_results']} chunks | "
                  f"Top similarity: {results['detailed_results'][0]['similarity']:.3f}")
        
        avg_time = total_time / len(test_queries)
        print(f"\n📊 Batch Summary:")
        print(f"   • Total time: {total_time:.1f}ms")
        print(f"   • Average per query: {avg_time:.1f}ms")
        print(f"   • Queries per second: {1000/avg_time:.1f}")

def main():
    """Main function to run the query tester."""
    print("=" * 60)
    print("    🧠 FAISS Vector Embedding Query Tester")  
    print("=" * 60)
    
    try:
        tester = QueryTester()
        tester.interactive_mode()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())