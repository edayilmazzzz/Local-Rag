import os

# Limit CPU threads to prevent overheating/shutdown
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["ONNXRUNTIME_NUM_THREADS"] = "2"

import math
from typing import List, Dict, Any, Tuple
import database


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Calculates cosine similarity between two vectors using pure Python.
    Returns a float in range [-1.0, 1.0].
    A score of 1.0 means identical direction (most similar).
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def retrieve_top_k(
    db_path: str,
    query_embedding: List[float],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieves the top-k most similar chunks from the database.

    Steps:
      1. Load all chunks (with embeddings) from SQLite.
      2. Compute cosine similarity between query and each chunk.
      3. Sort descending by similarity score.
      4. Return top-k results as a list of dicts.
    """
    all_chunks = database.get_all_chunks(db_path)

    if not all_chunks:
        return []

    # Compute similarity scores
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for chunk in all_chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored.append((score, chunk))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Build result list without embedding vectors (keep response lean)
    results = []
    for score, chunk in scored[:top_k]:
        results.append({
            "id": chunk["id"],
            "document_id": chunk["document_id"],
            "filename": chunk["filename"],
            "chunk_index": chunk["chunk_index"],
            "content": chunk["content"],
            "score": round(score, 6),
        })

    return results


if __name__ == "__main__":
    from foundry_local_sdk import FoundryLocalManager, Configuration

    DB_PATH = "data/rag_database.db"

    print("Loading embedding model for retrieval test...")
    config = Configuration(app_name="local-rag-assistant")
    manager = FoundryLocalManager(config)

    model = manager.catalog.get_model("qwen3-embedding-0.6b")
    model.load()
    emb_client = model.get_embedding_client()

    query = "What is Microsoft Foundry Local?"
    print(f"\nQuery: '{query}'")

    # Generate embedding for the query
    res = emb_client.generate_embedding(input_text=query)
    query_embedding = res.data[0].embedding

    # Retrieve top-k matching chunks
    results = retrieve_top_k(DB_PATH, query_embedding, top_k=3)

    print(f"\n--- Top {len(results)} Results ---")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']:.4f} | File: {r['filename']} | Chunk #{r['chunk_index']}")
        print(f"    Content: {r['content'][:200]}...")
    print("\nRetrieval test complete.")
