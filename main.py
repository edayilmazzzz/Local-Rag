import os
import sys

# Limit CPU threads to prevent overheating/shutdown
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["ONNXRUNTIME_NUM_THREADS"] = "2"

# Force UTF-8 output on Windows to avoid charmap encoding errors
if sys.stdout.encoding != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
if sys.stderr.encoding != "utf-8":
    sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1)

from rag_system import RAGSystem

DB_PATH = "data/rag_database.db"

BANNER = """
+-------------------------------------------------------+
|     Local RAG Assistant  |  Powered by Foundry Local  |
|          Tamamen Cevrimdisi (Offline)                  |
+-------------------------------------------------------+
"""

HELP_TEXT = """
Komutlar:
  upload <dosya_yolu>   -> PDF veya TXT belge yukle ve isle
  docs                  -> Yuklenmis belgeleri listele
  ask <soru>            -> Belgeler hakkinda soru sor
  clear                 -> Veritabanini sifirla (tum belge ve chunk'lari sil)
  help                  -> Bu yardim metnini goster
  exit | quit           -> Uygulamadan cik
"""


def print_separator():
    print("-" * 57)


def cmd_upload(rag: RAGSystem, args: list[str]) -> None:
    if not args:
        print("Kullanim: upload <dosya_yolu>")
        return
    file_path = " ".join(args)  # allow spaces in path

    # Strip surrounding quotes added by Windows "Copy as path"
    file_path = file_path.strip().strip('"').strip("'")

    # Normalize path separators
    file_path = os.path.normpath(file_path)

    if not os.path.exists(file_path):
        print(f"Hata: Dosya bulunamadi -> {file_path}")
        print("Ipucu: Yolu dogrudan terminale suruklemeyi deneyin,")
        print("       ya da tam yolu elle yazin.")
        return
    try:
        rag.ingest(file_path)
    except ValueError as e:
        print(f"Hata: {e}")


def cmd_docs(rag: RAGSystem) -> None:
    docs = rag.list_documents()
    if not docs:
        print("Henüz hiç belge yüklenmemiş.")
        return
    print(f"\n{'ID':<5} {'Dosya Adı':<35} {'Yüklenme Tarihi'}")
    print_separator()
    for doc in docs:
        print(f"{doc['id']:<5} {doc['filename']:<35} {doc['uploaded_at']}")
    print()


def cmd_ask(rag: RAGSystem, args: list[str]) -> None:
    if not args:
        print("Kullanım: ask <sorunuz>")
        return
    question = " ".join(args)
    print(f"\nSoru: {question}")
    print("Yanıt üretiliyor...\n")
    answer = rag.ask(question)
    print_separator()
    print(answer)
    print_separator()
    print()


def cmd_clear(rag: RAGSystem) -> None:
    confirm = input("Tüm belge ve chunk'lar silinecek. Emin misiniz? (evet/hayır): ").strip().lower()
    if confirm == "evet":
        from database import clear_database
        clear_database(DB_PATH)
        print("Veritabanı temizlendi.")
    else:
        print("İptal edildi.")


def main():
    print(BANNER)
    print("Sistem başlatılıyor, lütfen bekleyin...")

    rag = RAGSystem(db_path=DB_PATH)
    rag.initialize()

    print(HELP_TEXT)

    while True:
        try:
            raw = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÇıkılıyor...")
            sys.exit(0)

        if not raw:
            continue

        parts = raw.split()
        command = parts[0].lower()
        args = parts[1:]

        if command in ("exit", "quit"):
            print("Görüşmek üzere!")
            sys.exit(0)
        elif command == "upload":
            cmd_upload(rag, args)
        elif command == "docs":
            cmd_docs(rag)
        elif command == "ask":
            cmd_ask(rag, args)
        elif command == "clear":
            cmd_clear(rag)
        elif command == "help":
            print(HELP_TEXT)
        else:
            print(f"Tanınmayan komut: '{command}'. Yardım için 'help' yazın.")


if __name__ == "__main__":
    main()
