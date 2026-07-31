<div align="center">

# Local RAG Assistant

### Fully Offline Document Q&A — Powered by Microsoft Foundry Local

[![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Foundry Local](https://img.shields.io/badge/Foundry_Local-v0.8-purple?style=for-the-badge&logo=microsoft&logoColor=white)](https://foundrylocal.ai)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://microsoft.com/windows)

> Upload your PDFs and TXT files. Ask questions. Get grounded answers.
> **No internet. No cloud. No API keys. Ever.**

</div>

---

## What Is This?

**Local RAG Assistant** is a command-line application that lets you build a personal, private knowledge base from your own documents — and query it using a local large language model running entirely on your machine.

It implements a full **Retrieval-Augmented Generation (RAG)** pipeline from scratch:

```
Your Documents
      |
      v
Local Embedding Model   -->  SQLite Vector Store
      |
      v
Cosine Similarity Search
      |
      v
Context-Grounded Prompt
      |
      v
Local LLM (Foundry Local)
      |
      v
Answer — based only on YOUR documents
```

Everything runs **on-device** using [Microsoft Foundry Local](https://foundrylocal.ai). Your data never leaves your computer.

---

## Features

| Feature | Details |
|---|---|
| Document Support | PDF and TXT files |
| Local Embeddings | `qwen3-embedding-0.6b` — 1024 dimensions |
| Local LLM | `qwen3-0.6b` via OpenAI-compatible local API |
| Vector Store | SQLite — no external database required |
| Semantic Search | Pure Python cosine similarity |
| Offline | Works 100% without internet after model download |
| CLI Interface | Simple interactive terminal commands |
| Private | Your documents never leave your machine |

---

## Project Structure

```
local_rag_assistant/
|
|-- main.py           # CLI entry point and command loop
|-- rag_system.py     # Central RAG coordinator (lazy model loading)
|-- ingestion.py      # Document reading, chunking and embedding
|-- retrieval.py      # Cosine similarity search
|-- database.py       # SQLite schema and CRUD operations
|
|-- requirements.txt  # Python dependencies
|-- data/             # SQLite database (auto-created at first run)
|-- .venv/            # Python virtual environment
```

---

## Setup

### Prerequisites

- Windows 10/11
- Python 3.14+
- [Microsoft Foundry Local](https://foundrylocal.ai) CLI installed

### Step 1 — Download AI Models

Internet is required for this step only. After this, everything runs offline.

```powershell
foundry model download qwen3-0.6b
foundry model download qwen3-embedding-0.6b
```

### Step 2 — Create Virtual Environment and Install Dependencies

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

---

## Usage

```powershell
.venv\Scripts\python main.py
```

### Commands

```
upload <path>    Upload and index a PDF or TXT document
ask    <query>   Ask a question about your documents
docs             List all indexed documents
clear            Wipe the database
help             Show help
exit             Quit
```

### Example Session

```
+-------------------------------------------------------+
|     Local RAG Assistant  |  Powered by Foundry Local  |
|          Fully Offline                                 |
+-------------------------------------------------------+

>>> upload C:\Users\You\Desktop\research_paper.pdf
Processing file: research_paper.pdf
Splitting text into chunks...
Generated 52 chunks.
Generating embeddings (this may take a moment)...
Successfully ingested and saved 52 chunks for 'research_paper.pdf'.

>>> ask What methodology was used in this study?
Generating answer...
---------------------------------------------------------
The study employed a mixed-methods approach combining
quantitative surveys (n=500) and qualitative interviews...
---------------------------------------------------------

>>> docs
ID    Filename                  Upload Date
---------------------------------------------
1     research_paper.pdf        2026-07-27 15:00:00

>>> exit
Goodbye!
```

> **Tip:** You can paste Windows "Copy as path" quoted paths directly — the CLI strips quotes automatically.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.14 |
| Local AI Runtime | Microsoft Foundry Local v0.8.x |
| LLM | qwen3-0.6b (CPU, Apache-2.0) |
| Embedding Model | qwen3-embedding-0.6b (1024 dimensions, Apache-2.0) |
| Vector and Text Store | SQLite |
| PDF Parsing | pypdf |
| Similarity Search | Cosine similarity — pure Python, no NumPy |
| LLM Client | openai SDK pointed to local OpenAI-compatible endpoint |
| Platform | Windows |

---

## How It Works

### Document Ingestion

1. **Read** — `pypdf` for PDFs, native `open()` for TXT files
2. **Chunk** — Sliding window split (`chunk_size=800`, `overlap=150` characters) with word-boundary detection
3. **Embed** — Each chunk is passed to `qwen3-embedding-0.6b`, producing a 1024-float vector
4. **Store** — Chunk text and JSON-serialized embedding saved to SQLite

### Query and Answer

1. **Embed query** — Same embedding model converts the question to a vector
2. **Search** — Cosine similarity computed between query and all stored chunk vectors
3. **Retrieve** — Top-K chunks selected by similarity score
4. **Prompt** — Retrieved context and question assembled into a structured prompt
5. **Generate** — `qwen3-0.6b` generates an answer using only the retrieved context (temperature=0.2)

### Lazy Model Loading

To reduce startup time and avoid CPU overload, the system uses lazy loading:

- **Startup** — Only the embedding model loads
- **First ask command** — The chat model loads on demand, once per session

---

## Limitations

- The LLM answers only from uploaded documents — it ignores its general knowledge by design
- Very large documents (50+ pages) may take a few minutes to embed on first upload
- The first ask command in a session takes approximately 15-30 seconds while the chat model loads
- Currently supports PDF and TXT formats only

---

## License

This project is for personal and educational use.
