import sqlite3
import json
import os
from typing import List, Dict, Tuple, Any

def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Establishes a connection to the SQLite database and enables foreign keys.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(db_path: str) -> None:
    """
    Initializes the database schema if it doesn't already exist.
    Creates 'documents' and 'chunks' tables.
    """
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Table to track uploaded documents
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL UNIQUE,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Table to track document chunks and their embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,  -- JSON-serialized list of floats
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            );
        """)
        conn.commit()

def insert_document(db_path: str, filename: str, file_path: str) -> int:
    """
    Inserts a document metadata record. If document already exists, returns its ID.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO documents (filename, file_path) VALUES (?, ?)",
                (filename, file_path)
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Document already exists, retrieve and return its ID
            cursor.execute("SELECT id FROM documents WHERE file_path = ?", (file_path,))
            result = cursor.fetchone()
            return result[0] if result else -1

def insert_chunks(db_path: str, document_id: int, chunks: List[Tuple[int, str, List[float]]]) -> None:
    """
    Inserts multiple chunks in bulk.
    chunks: list of tuples (chunk_index, content, embedding_vector)
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        # Prepare data: serialize embedding list to JSON string
        prepared_chunks = [
            (document_id, index, content, json.dumps(embedding))
            for index, content, embedding in chunks
        ]
        cursor.executemany(
            "INSERT INTO chunks (document_id, chunk_index, content, embedding) VALUES (?, ?, ?, ?)",
            prepared_chunks
        )
        conn.commit()

def get_all_chunks(db_path: str) -> List[Dict[str, Any]]:
    """
    Retrieves all chunks from the database along with document info.
    Deserializes embeddings from JSON.
    """
    with get_connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.id, c.document_id, c.chunk_index, c.content, c.embedding, d.filename
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
        """)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "document_id": row["document_id"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "embedding": json.loads(row["embedding"]),  # deserialize JSON to list of floats
                "filename": row["filename"]
            })
        return results

def clear_database(db_path: str) -> None:
    """
    Deletes all records from the database tables.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chunks")
        cursor.execute("DELETE FROM documents")
        conn.commit()
