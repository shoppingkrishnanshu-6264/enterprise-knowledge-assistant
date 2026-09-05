"""
Vector Search Tool for the Enterprise Knowledge Assistant.

Wraps the ChromaDB collection built during ingestion into a simple function
the agent can call: vector_search(query) -> list of relevant chunks with sources.
"""

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "company_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)
_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
_collection = _client.get_collection(name=COLLECTION_NAME, embedding_function=_embedding_fn)


def vector_search(query: str, n_results: int = 3, relevance_gap: float = 0.08) -> list[dict]:
    """
    Searches the company documents vector store for chunks relevant to the query.
    Returns a list of dicts: {"text": ..., "source": ..., "distance": ...}
    Lower distance = more relevant.

    relevance_gap: after fetching n_results candidates, any chunk whose distance is more
    than `relevance_gap` worse than the single best match is dropped. This prevents a
    strongly-matching top chunk from being diluted by a weaker, less relevant chunk that
    happens to still rank in the top-N (e.g., a chunk about an unrelated exception clause
    that shares some vocabulary with the query). The best match is always kept regardless.
    """
    results = _collection.query(query_texts=[query], n_results=n_results)

    candidates = []
    for doc, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        candidates.append({
            "text": doc,
            "source": metadata["source"],
            "distance": distance,
        })

    if not candidates:
        return candidates

    best_distance = min(c["distance"] for c in candidates)
    filtered = [c for c in candidates if c["distance"] <= best_distance + relevance_gap]

    return filtered


if __name__ == "__main__":
    # Quick manual test
    test_query = "What is the refund cooling-off period?"
    results = vector_search(test_query)
    for r in results:
        print(f"[{r['source']}] (distance={r['distance']:.4f})\n{r['text'][:200]}...\n")
