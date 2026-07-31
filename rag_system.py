import os

# Limit CPU threads to prevent overheating/shutdown
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["ONNXRUNTIME_NUM_THREADS"] = "2"

from typing import List, Dict, Any
from openai import OpenAI
from foundry_local_sdk import FoundryLocalManager, Configuration
import retrieval
import database

# Foundry Local model identifiers
EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"
CHAT_MODEL_ALIAS = "qwen3-0.6b"
CHAT_MODEL_ID = "qwen3-0.6b-generic-cpu"


class RAGSystem:
    """
    Coordinates the full RAG pipeline using lazy loading:
      - Embedding model loads at initialize() → needed for both upload and ask.
      - Chat model (web service) loads on first ask() call → avoids double load at startup.
      - SQLite is used for chunk storage and retrieval.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._manager: FoundryLocalManager | None = None
        self._emb_client = None
        self._chat_client: OpenAI | None = None
        self._chat_model_loaded: bool = False
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_manager(self) -> FoundryLocalManager:
        """Creates the FoundryLocalManager singleton if not yet created."""
        if self._manager is None:
            print("Foundry Local Manager baslatiliyor...")
            config = Configuration(app_name="local-rag-assistant")
            self._manager = FoundryLocalManager(config)
        return self._manager

    def _load_chat_model(self) -> None:
        """
        Lazily loads the chat model and starts the OpenAI-compatible web service.
        Called on the first ask() invocation.
        """
        if self._chat_model_loaded:
            return

        manager = self._ensure_manager()
        print(f"Chat modeli yukleniyor: {CHAT_MODEL_ALIAS} (ilk soru icin bir kere yapilir)...")
        chat_model = manager.catalog.get_model(CHAT_MODEL_ALIAS)
        chat_model.load()

        manager.start_web_service()
        base_url = manager.urls[0] + "/v1"
        self._chat_client = OpenAI(base_url=base_url, api_key="none")
        self._chat_model_loaded = True
        print("Chat modeli hazir.\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Phase 1 of startup: initializes the manager and embedding model only.
        Chat model is deferred to first ask() call to reduce startup time.
        """
        if self._initialized:
            return

        manager = self._ensure_manager()

        print(f"Embedding modeli yukleniyor: {EMBEDDING_MODEL_ALIAS}...")
        emb_model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
        emb_model.load()
        self._emb_client = emb_model.get_embedding_client()

        # Initialize DB schema
        database.init_db(self.db_path)

        self._initialized = True
        print("Sistem hazir. Belge yukleyebilir veya soru sorabilirsiniz.\n")

    def ingest(self, file_path: str) -> None:
        """Ingests a document (PDF or TXT) into the database."""
        if not self._initialized:
            raise RuntimeError("RAGSystem not initialized. Call initialize() first.")

        from ingestion import ingest_file
        ingest_file(self.db_path, file_path, self._emb_client)

    def ask(self, question: str, top_k: int = 3, max_tokens: int = 512) -> str:
        """
        Answers a question using the RAG pipeline:
          1. Embeds the question using the embedding model.
          2. Retrieves top-k relevant chunks from SQLite via cosine similarity.
          3. Builds a context-grounded prompt.
          4. Calls the local LLM and returns its response.
        """
        if not self._initialized:
            raise RuntimeError("RAGSystem not initialized. Call initialize() first.")

        # Lazily load the chat model on first use
        self._load_chat_model()

        # 1. Embed the query
        res = self._emb_client.generate_embedding(input_text=question)
        query_embedding = res.data[0].embedding

        # 2. Retrieve relevant chunks
        chunks = retrieval.retrieve_top_k(self.db_path, query_embedding, top_k=top_k)

        if not chunks:
            return (
                "Veritabaninda belge bulunamadi. "
                "Lutfen once bir belge yukleyin ('upload' komutu)."
            )

        # 3. Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Kaynak {i} - {chunk['filename']}, Bolum {chunk['chunk_index']}]\n"
                f"{chunk['content']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # 4. Build prompts (ASCII-safe for Windows console compatibility)
        system_prompt = (
            "Sen yardimci bir AI asistanisin. "
            "Asagidaki belge bolumlerini kullanarak soruyu Turkce veya soruyla ayni dilde yanitla. "
            "Eger belgeler soruyu yanitlamak icin yeterli bilgi icermiyorsa, bunu acikca belirt. "
            "Yanitinda yalnizca belgelerden elde ettigin bilgileri kullan."
        )

        user_prompt = (
            f"Asagidaki belge bolumleri saglanmistir:\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"Soru: {question}\n\n"
            f"Yanit:"
        )

        # 5. Call local LLM
        response = self._chat_client.chat.completions.create(
            model=CHAT_MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns a list of all ingested documents from the database."""
        with database.get_connection(self.db_path) as conn:
            conn.row_factory = __import__("sqlite3").Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, filename, file_path, uploaded_at "
                "FROM documents ORDER BY uploaded_at DESC"
            )
            return [dict(row) for row in cursor.fetchall()]
