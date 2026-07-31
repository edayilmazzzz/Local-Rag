import os

# Limit CPU threads to prevent overheating/shutdown
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["ONNXRUNTIME_NUM_THREADS"] = "2"

import pypdf
from typing import List, Tuple
from foundry_local_sdk import FoundryLocalManager, Configuration
import database

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF file page-by-page.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    reader = pypdf.PdfReader(pdf_path)
    extracted_text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            extracted_text.append(page_text)
    return "\n".join(extracted_text)

def extract_text_from_txt(txt_path: str) -> str:
    """
    Extracts all text from a TXT file.
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"TXT file not found: {txt_path}")
        
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Splits text into chunks of specified size with overlap.
    Attempts to avoid breaking words by splitting on whitespaces when possible.
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Search backward for space/newline to prevent splitting a word
        if end < text_len:
            boundary = -1
            # Search within the last 100 characters of the window
            search_start = max(start, end - 100)
            for i in range(end - 1, search_start - 1, -1):
                if text[i] in (' ', '\n', '\t'):
                    boundary = i
                    break
            if boundary != -1:
                end = boundary + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        next_start = end - chunk_overlap
        # Guarantee progression to prevent infinite loop
        if next_start <= start:
            start = end
        else:
            start = next_start
            
    return chunks

def ingest_file(db_path: str, file_path: str, embedding_client) -> None:
    """
    Extracts text, chunks it, generates embeddings, and saves to the SQLite database.
    """
    print(f"\nProcessing file: {file_path}")
    filename = os.path.basename(file_path)
    extension = os.path.splitext(filename)[1].lower()
    
    # 1. Extract text
    if extension == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif extension == ".txt":
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {extension}. Only PDF and TXT are supported.")
        
    # 2. Chunk text
    print("Splitting text into chunks...")
    chunks = split_text(text, chunk_size=800, chunk_overlap=150)
    print(f"Generated {len(chunks)} chunks.")
    
    # 3. Generate embeddings & format chunks for database insert
    db_chunks: List[Tuple[int, str, List[float]]] = []
    print("Generating embeddings (this may take a moment)...")
    
    for index, chunk_content in enumerate(chunks):
        # Generate embedding for single chunk
        res = embedding_client.generate_embedding(input_text=chunk_content)
        embedding_vector = res.data[0].embedding
        db_chunks.append((index, chunk_content, embedding_vector))
        
    # 4. Save to database
    print("Saving to database...")
    doc_id = database.insert_document(db_path, filename, file_path)
    if doc_id == -1:
        print("Warning: Document metadata could not be updated or retrieved.")
        return
        
    # Clear old chunks if re-uploading the same file to prevent duplicates
    with database.get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        conn.commit()
        
    database.insert_chunks(db_path, doc_id, db_chunks)
    print(f"Successfully ingested and saved {len(db_chunks)} chunks for '{filename}'.")

if __name__ == "__main__":
    # Test script to run standalone ingestion
    DB_PATH = "data/rag_database.db"
    database.init_db(DB_PATH)
    
    print("Loading Foundry Local Manager for testing...")
    config = Configuration(app_name="local-rag-assistant")
    manager = FoundryLocalManager(config)
    
    model_name = "qwen3-embedding-0.6b"
    model = manager.catalog.get_model(model_name)
    model.load()
    emb_client = model.get_embedding_client()
    
    # Create a dummy test file
    test_file_path = "test_doc.txt"
    test_content = (
        "Microsoft Foundry Local is an offline model execution framework.\n"
        "It runs on-device Large Language Models without cloud dependencies.\n"
        "This project uses SQLite to store text chunks and vector embeddings.\n"
        "We are building a Q&A chatbot using character-based paragraph chunking."
    )
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(test_content)
        
    try:
        ingest_file(DB_PATH, test_file_path, emb_client)
        
        # Verify db contents
        all_chunks = database.get_all_chunks(DB_PATH)
        print(f"\nVerification: Total chunks stored in DB: {len(all_chunks)}")
        if all_chunks:
            print(f"First chunk content: '{all_chunks[0]['content']}'")
            print(f"First chunk embedding length: {len(all_chunks[0]['embedding'])}")
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
