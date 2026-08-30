"""
Ingestion pipeline for the Enterprise Knowledge Assistant.

What this script does:
1. Loads all .txt policy documents from data/docs/
2. Splits each document into overlapping chunks (so context isn't cut mid-sentence)
3. Generates embeddings locally using sentence-transformers (no API key needed)
4. Stores chunks + embeddings + metadata into a persistent ChromaDB collection

Run this whenever you add/change documents in data/docs/.
"""

import os
import glob
import chromadb
from chromadb.utils import embedding_functions

# ---------- CONFIG ----------
DOCS_DIR = "data/docs"
CHROMA_DB_DIR = "chroma_db"          # ChromaDB will persist data here on disk
COLLECTION_NAME = "company_docs"
CHUNK_SIZE = 800                      # characters per chunk (not tokens, kept simple)
CHUNK_OVERLAP = 150                   # overlap between consecutive chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, runs locally, good quality for this use case


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits text into overlapping chunks by character count.
    Overlap ensures a sentence/idea split across a chunk boundary isn't lost entirely.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap  # move forward, but re-include the overlap window

    return [c for c in chunks if c]  # drop any empty chunks


def load_documents(docs_dir: str) -> list[dict]:
    """
    Loads all .txt files from docs_dir.
    Returns a list of dicts: {"filename": ..., "text": ...}
    """
    documents = []
    filepaths = glob.glob(os.path.join(docs_dir, "*.txt"))

    if not filepaths:
        print(f"WARNING: No .txt files found in '{docs_dir}'. Check the path.")
        return documents

    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        filename = os.path.basename(filepath)
        documents.append({"filename": filename, "text": text})
        print(f"Loaded: {filename} ({len(text)} characters)")

    return documents


def build_vector_store():
    """
    Main ingestion routine: load -> chunk -> embed -> store in ChromaDB.
    """
    print("=" * 60)
    print("STEP 1: Loading documents")
    print("=" * 60)
    documents = load_documents(DOCS_DIR)

    if not documents:
        print("No documents loaded. Exiting.")
        return

    print("\n" + "=" * 60)
    print("STEP 2: Chunking documents")
    print("=" * 60)

    all_chunks = []
    all_metadatas = []
    all_ids = []

    for doc in documents:
        chunks = chunk_text(doc["text"])
        print(f"  {doc['filename']} -> {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": doc["filename"],
                "chunk_index": i,
            })
            all_ids.append(f"{doc['filename']}_chunk_{i}")

    print(f"\nTotal chunks created across all documents: {len(all_chunks)}")

    print("\n" + "=" * 60)
    print("STEP 3: Setting up embedding function (local, sentence-transformers)")
    print("=" * 60)
    print(f"Model: {EMBEDDING_MODEL} (first run will download the model, ~90MB)")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    print("\n" + "=" * 60)
    print("STEP 4: Storing chunks + embeddings in ChromaDB")
    print("=" * 60)

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # If the collection already exists from a previous run, delete it so we don't get duplicates
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        print(f"Collection '{COLLECTION_NAME}' already exists — deleting old version to avoid duplicates.")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    # Chroma can be picky about batch size for very large ingestions; 511 sales rows aside,
    # our doc chunk count is small, so we add everything in one call.
    collection.add(
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=all_ids,
    )

    print(f"\nSuccessfully stored {collection.count()} chunks in ChromaDB collection '{COLLECTION_NAME}'.")
    print(f"Persisted to disk at: ./{CHROMA_DB_DIR}")
    print("\nIngestion complete.")


if __name__ == "__main__":
    build_vector_store()
