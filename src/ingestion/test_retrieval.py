"""
Quick test script to confirm the ChromaDB vector store is working after ingestion.
Run this AFTER running src/ingestion/ingest_docs.py at least once.
"""

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "company_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

print(f"Collection '{COLLECTION_NAME}' contains {collection.count()} chunks.\n")

test_queries = [
    "What is the refund cooling-off period for enterprise customers?",
    "How many days of sick leave do employees get?",
    "What is the home office stipend for remote employees?",
]

for query in test_queries:
    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    results = collection.query(query_texts=[query], n_results=2)

    for i, (doc, metadata, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        print(f"\n-- Result {i + 1} (source: {metadata['source']}, distance: {distance:.4f}) --")
        print(doc[:300] + ("..." if len(doc) > 300 else ""))

    print("\n")
