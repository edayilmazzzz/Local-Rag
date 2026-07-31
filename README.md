# Local RAG Assistant

**A fully offline, locally-running Retrieval-Augmented Generation (RAG) system built with Microsoft Foundry Local and SQLite.**

Upload your PDF or TXT documents, ask questions about their content — no internet, no cloud, no API keys required.

---

## Overview

This project is a complete RAG pipeline built from scratch using:
- **Microsoft Foundry Local** for running AI models entirely on-device
- **SQLite** as a lightweight, zero-dependency vector and text store
- **Pure Python cosine similarity** for semantic search (no FAISS or NumPy required)

When a document is uploaded, the system:
1. Reads the file (PDF or TXT)
2. Splits the text into overlapping chunks
3. Generates a 1024-dimensional embedding vector for each chunk using a local embedding model
4. Persists the chunks and their vectors in SQLite

When a question is asked, the system:
1. Embeds the query using the same local model
2. Computes cosine similarity between the query and all stored chunk vectors
3. Retrieves the top-K most relevant chunks
4. Builds a grounded prompt from the retrieved context
5. Calls a local LLM to generate a response — based only on the retrieved document content

---

## Features

- Fully offline — works without any internet connection after model download
- Supports PDF and TXT documents
- Local embedding generation with `qwen3-embedding-0.6b`
- Local LLM inference with `qwen3-0.6b` via OpenAI-compatible API
- SQLite-based persistent document store
- Cosine similarity semantic search in pure Python
- CPU thread limiting to prevent overheating on laptops
- Simple and interactive CLI interface

---

## Project Structure

```
local_rag_assistant/
├── main.py            # CLI entry point
├── rag_system.py      # Central RAG coordinator class
├── ingestion.py       # Document loading, chunking, embedding & storage
├── retrieval.py       # Cosine similarity-based chunk retrieval
├── database.py        # SQLite schema and CRUD operations
├── requirements.txt   # Python dependencies
├── data/              # SQLite database (auto-created at first run)
└── .venv/             # Python virtual environment
```

---

## Requirements

- Windows 10/11
- Python 3.14+
- [Microsoft Foundry Local](https://foundrylocal.ai) CLI installed
- ~1 GB free disk space for models

---

## Setup

### 1. Download models (requires internet — one time only)

```powershell
foundry model download qwen3-0.6b
foundry model download qwen3-embedding-0.6b
```

After this step, the system runs fully offline.

### 2. Navigate to the project directory

```powershell
cd C:\path\to\local_rag_assistant
```

### 3. Create a virtual environment and install dependencies

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

---

## Usage

```powershell
.venv\Scripts\python main.py
```

### Available Commands

| Command | Description |
|---|---|
| `upload <path>` | Upload and index a PDF or TXT document |
| `ask <question>` | Ask a question about uploaded documents |
| `docs` | List all indexed documents |
| `clear` | Clear the database (removes all documents and chunks) |
| `help` | Show help text |
| `exit` / `quit` | Exit the application |

### Example Session

```
>>> upload C:\Users\YourName\Desktop\report.pdf
Processing file: report.pdf
Splitting text into chunks...
Generated 34 chunks.
Generating embeddings (this may take a moment)...
Saving to database...
Successfully ingested and saved 34 chunks for 'report.pdf'.

>>> ask What are the main findings of this report?
Generating answer...
---------------------------------------------------------
The report highlights three key findings...
---------------------------------------------------------

>>> docs
ID    Filename                            Upload Date
---------------------------------------------------------
1     report.pdf                          2026-07-27 15:00:00

>>> exit
```

> **Tip:** You can paste Windows "Copy as path" style paths (with surrounding quotes) directly — the system strips them automatically.

---

## Architecture

```
User Query
    │
    ▼
[Embedding Model]  ←  qwen3-embedding-0.6b (local, offline)
    │
    ▼
[Cosine Similarity Search]  ←  SQLite chunk vectors
    │
    ▼
[Top-K Chunks Retrieved]
    │
    ▼
[Prompt Builder]  →  System prompt + context + question
    │
    ▼
[Chat LLM]  ←  qwen3-0.6b via OpenAI-compatible local API
    │
    ▼
Answer (grounded only in uploaded documents)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.14 |
| Local Runtime | Microsoft Foundry Local v0.8.x |
| LLM | qwen3-0.6b (CPU, generic) |
| Embedding Model | qwen3-embedding-0.6b (1024 dimensions) |
| Vector Store | SQLite (JSON-serialized float arrays) |
| PDF Parsing | pypdf |
| Similarity Search | Pure Python cosine similarity |
| LLM Client | openai (pointed to local endpoint) |
| OS | Windows |

---

## Limitations

- The LLM only answers based on uploaded documents — it does not use its general knowledge
- Large documents (50+ pages) may take longer to embed on first upload
- The first `ask` command triggers chat model loading (~15–30 seconds); subsequent queries are faster
- Currently supports PDF and TXT formats only

---

## License

This project is for personal and educational use.
