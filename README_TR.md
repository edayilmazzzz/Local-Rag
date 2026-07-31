# Local RAG Assistant

**Tamamen çevrimdışı çalışan, yerel yapay zeka destekli belge soru-cevap sistemi.**

PDF ve TXT belgelerinizi yükleyin, içerikleri hakkında Türkçe sorular sorun — internet olmadan, bulut olmadan.

---

## Proje Hakkında

Bu proje, Microsoft Foundry Local çalışma zamanı ve SQLite veritabanı kullanılarak sıfırdan inşa edilmiş bir **RAG (Retrieval-Augmented Generation)** sistemidir.

Kullanıcı bir belge yüklediğinde sistem:
1. Belgeyi okur (PDF veya TXT)
2. Metni küçük parçalara (chunk) böler
3. Her parça için yerel bir embedding modeli ile vektör üretir
4. Vektörleri ve metinleri SQLite veritabanına kaydeder

Kullanıcı soru sorduğunda sistem:
1. Soruyu da aynı embedding modeliyle vektöre dönüştürür
2. Veritabanındaki chunk vektörleriyle kosinüs benzerliği hesaplar
3. En ilgili parçaları bulur
4. Bu parçaları bağlam olarak yerel LLM'e gönderir
5. LLM yalnızca bu bağlamı kullanarak yanıt üretir

---

## Özellikler

- Tamamen çevrimdışı — internet bağlantısı gerektirmez
- PDF ve TXT belge desteği
- Yerel embedding üretimi (qwen3-embedding-0.6b)
- Yerel LLM ile yanıt üretimi (qwen3-0.6b)
- SQLite tabanlı kalıcı belge deposu
- Kosinüs benzerliği ile semantik arama (NumPy veya FAISS gerektirmez)
- İşlemci aşırı yüklenmesini önleyen thread sınırlaması

---

## Proje Yapısı

```
local_rag_assistant/
├── main.py            # CLI arayüzü — programı buradan çalıştırın
├── rag_system.py      # RAG pipeline koordinatörü (merkezi sınıf)
├── ingestion.py       # Belge okuma, chunking ve embedding kayıt
├── retrieval.py       # Kosinüs benzerliği ile chunk arama
├── database.py        # SQLite şeması ve CRUD işlemleri
├── requirements.txt   # Python bağımlılıkları
├── data/              # SQLite veritabanı (otomatik oluşturulur)
└── .venv/             # Sanal Python ortamı
```

---

## Kurulum

### Gereksinimler

- Windows 10/11
- Python 3.14
- [Microsoft Foundry Local](https://foundrylocal.ai) (CLI kurulu olmalı)
- Visual Studio Code (önerilen)

### Adımlar

**1. Foundry Local modellerini indirin**

```powershell
foundry model download qwen3-0.6b
foundry model download qwen3-embedding-0.6b
```

> Bu işlem internet bağlantısı gerektirir ve sadece bir kez yapılır. Sonrasında tamamen çevrimdışı çalışır.

**2. Depoyu klonlayın veya projeyi açın**

```powershell
cd C:\Users\LENOVO\.gemini\antigravity\scratch\local_rag_assistant
```

**3. Sanal ortam oluşturun ve bağımlılıkları yükleyin**

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

---

## Kullanım

```powershell
.venv\Scripts\python main.py
```

Program başladıktan sonra şu komutları kullanabilirsiniz:

| Komut | Açıklama |
|---|---|
| `upload <yol>` | PDF veya TXT belge yükle |
| `ask <soru>` | Yüklü belgeler hakkında soru sor |
| `docs` | Yüklü belgeleri listele |
| `clear` | Veritabanını sıfırla |
| `help` | Yardım metnini göster |
| `exit` | Programdan çık |

### Örnek Kullanım

```
>>> upload C:\Users\LENOVO\Desktop\rapor.pdf
>>> ask Bu rapordaki ana bulgular nelerdir?
>>> docs
>>> exit
```

> **Not:** `upload` komutuna Windows'tan kopyalanan tırnaklı yolları yapıştırabilirsiniz — sistem otomatik temizler.

---

## Teknik Detaylar

| Bileşen | Teknoloji |
|---|---|
| Platform | Windows |
| Dil | Python 3.14 |
| Çalışma zamanı | Microsoft Foundry Local v0.8.x |
| LLM | qwen3-0.6b (CPU) |
| Embedding modeli | qwen3-embedding-0.6b (1024 boyut) |
| Veritabanı | SQLite |
| PDF okuma | pypdf |
| Benzerlik | Kosinüs benzerliği (saf Python) |

---

## Sınırlamalar

- LLM yalnızca yüklü belgelerdeki bilgileri kullanır; genel bilgiye başvurmaz
- Çok büyük PDF'lerde (50+ sayfa) embedding üretimi zaman alabilir
- İlk `ask` komutunda chat modeli yükleneceğinden 15-30 saniye bekleme yaşanabilir
