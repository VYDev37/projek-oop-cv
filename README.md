# 📸 Sistem AI Image Analysis & Tagging Otomatis [I2]
### Platform Pemrosesan Citra Cerdas Berbasis Vision AI (Gemini / GPT-4o / Mock) & Prinsip Object-Oriented Programming (OOP)

Sistem perangkat lunak modular siap pakai (*production-grade*) yang dirancang khusus untuk memenuhi **100% kriteria Rubrik Tugas Kuliah OOP [I2]**: Captioning naratif, Tagging kata kunci, Deteksi objek, Kategorisasi citra, Palet warna dominan, Ekstraksi teks OCR, Moderasi konten otomatis, dan Manajemen koleksi foto cerdas.

---

## 📋 Pemetaan 20 Fitur Rubrik [I2] ke Kode Sumber

### A. Fitur Dasar & Analisis (1–10)
| No | Kebutuhan Rubrik | Lokasi Implementasi | Penjelasan Teknis |
|:---:|:---|:---|:---|
| **1** | **Definisi `VisionClient` (ABC)** | `vision_client.py` (`VisionClient`) | Kelas abstrak berbasis `abc.ABC` dengan `@abstractmethod analyze()`. |
| **2** | **`GeminiVisionClient extends VisionClient`** | `vision_client.py` (`GeminiVisionClient`) | Mengirim gambar ke Google GenAI SDK (`google-genai`), parse JSON ke skema terstruktur. |
| **3** | **`MockVisionClient` (Mode Offline)** | `vision_client.py` (`MockVisionClient`) | Mode demo tanpa API key dan koneksi internet, deterministik & cepat. |
| **4** | **Raise `VisionAPIException`** | `vision_client.py` (`VisionAPIException`) | Custom exception saat terjadi kegagalan jaringan, parsing JSON, atau API error. |
| **5** | **`_encode_image()` (Base64)** | `vision_client.py` (`VisionClient._encode_image`) | Mengonversi file path, bytes, atau objek `PIL.Image` menjadi string base64 UTF-8. |
| **6** | **`PhotoAnalysisResult`** | `vision_client.py` (`PhotoAnalysisResult`) | Pydantic Model: `caption`, `tags`, `objects`, `kategori`, `is_safe`, `colors`, `ocr_text`. |
| **7** | **`analyze() -> PhotoAnalysisResult`** | Seluruh turunan `VisionClient` | Kontrak metode polimorfik yang mengembalikan objek `PhotoAnalysisResult`. |
| **8** | **`ContentModerator` (Cek `is_safe`)** | `photo_collection.py` (`ContentModerator`) | Memeriksa nilai `is_safe` dan memfilter foto yang melanggar panduan konten. |
| **9** | **`PhotoCollection.tambah()`** | `photo_collection.py` (`PhotoCollection.tambah`) | Menambahkan objek hasil analisis ke dalam koleksi internal. |
| **10**| **Tampilkan caption & top tags (CLI)** | `cli.py` (`display_photo_result`) | Menampilkan format rapi caption, tags, objek, warna, dan status moderasi di terminal. |

### B. Fitur Koleksi & Manajemen Lanjutan (11–20)
| No | Kebutuhan Rubrik | Lokasi Implementasi | Penjelasan Teknis |
|:---:|:---|:---|:---|
| **11**| **`PhotoCollection.cari_by_tag()`** | `photo_collection.py` (`PhotoCollection.cari_by_tag`) | Pencarian foto berdasarkan tag kata kunci secara *case-insensitive*. |
| **12**| **`PhotoCollection.filter_kategori()`** | `photo_collection.py` (`PhotoCollection.filter_kategori`) | Memfilter daftar foto berdasarkan nama kategori (Alam, Makanan, Dokumen, dll). |
| **13**| **Iterator pada `PhotoCollection`** | `photo_collection.py` (`PhotoCollectionIterator`) | Implementasi murni **Iterator Pattern** (`__iter__` dan `__next__`), mendukung `for p in collection`. |
| **14**| **Statistik tag terpopuler (`Counter`)** | `photo_collection.py` (`PhotoCollection.get_popular_tags`) | Menggunakan `collections.Counter` untuk menghitung frekuensi tag paling banyak muncul. |
| **15**| **Batch analysis (Banyak foto)** | `photo_collection.py` (`PhotoCollection.batch_analyze`) | Memproses sekumpulan foto sekaligus dengan penanganan error per-berkas. |
| **16**| **`GPT4VisionClient extends VisionClient`** | `vision_client.py` (`GPT4VisionClient`) | Provider OpenAI Vision API (`gpt-4o`) dengan payload base64 JSON mode. |
| **17**| **Komparasi hasil 2 provider** | `vision_client.py` (`ImageAnalysisService.compare_providers`) | Membandingkan caption, tag bersama (*shared tags*), dan tingkat keselarasan antar 2 AI. |
| **18**| **Laporan moderasi (Safe vs Unsafe)** | `photo_collection.py` (`ContentModerator.generate_moderation_report`) | Rekapitulasi jumlah foto aman, foto tidak aman, dan persentase kepatuhan. |
| **19**| **Ekspor koleksi ke CSV (SOLID: SRP)** | `photo_collection.py` (`CollectionCSVExporter`) | Kelas terpisah khusus penulisan file CSV agar mematuhi *Single Responsibility Principle*. |
| **20**| **Design Pattern: Strategy** | `vision_client.py` (`ImageAnalysisService`) | Menukar provider Vision AI aktif secara dinamis saat runtime (`set_strategy`). |

---

## 🏛️ Konsep OOP yang Diterapkan

1. **Inheritance & Abstraction (ABC)**:
   - `VisionClient(ABC)` sebagai kelas induk abstrak yang mewariskan fungsionalitas ke `GeminiVisionClient`, `GPT4VisionClient`, dan `MockVisionClient`.
2. **Polymorphism**:
   - Pemanggilan metode `client.analyze(image)` dieksekusi secara seragam terlepas dari provider konkret mana yang sedang aktif.
3. **Exception Handling**:
   - Kelas eksepsi terdedikasi `VisionAPIException` untuk menangkap galat API dan parsing data.
4. **Design Pattern: Strategy Pattern**:
   - `ImageAnalysisService` bertindak sebagai *Context* yang memegang objek strategi bertipe `VisionClient`.
5. **Design Pattern: Iterator Pattern**:
   - `PhotoCollection` mengembalikan objek `PhotoCollectionIterator` sehingga mendukung protokol iterasi native Python.
6. **Prinsip SOLID**:
   - **SRP (Single Responsibility Principle)**: Pemisahan tugas yang tegas antara analisis AI (`VisionClient`), penyimpanan koleksi (`PhotoCollection`), moderasi konten (`ContentModerator`), dan ekspor CSV (`CollectionCSVExporter`).
   - **OCP (Open/Closed Principle)**: Menambah provider AI baru cukup membuat subclass baru turunan `VisionClient` tanpa perlu mengubah kode yang ada.
   - **LSP & DIP (Liskov Substitution & Dependency Inversion)**: `ImageAnalysisService` bergantung pada abstraksi `VisionClient`, bukan implementasi konkretnya.
7. **Unit Testing Pytest**:
   - Rangkaian pengujian otomatis mandiri di folder `tests/` mencakup seluruh logika dan relasi kelas.

---

## 📂 Struktur Direktori Proyek

```text
Projek OOP/
│
├── config.py                 # Pengaturan konfigurasi, environment variable & direktori data
├── vision_client.py          # VisionClient (ABC), Gemini, GPT-4o, Mock, Exception, Strategy Service
├── photo_collection.py       # PhotoCollection, Iterator Pattern, ContentModerator, CSV Exporter (SRP)
├── cli.py                    # Runner baris perintah (CLI) untuk visualisasi hasil & demonstrasi lengkap
├── app.py                    # Web UI Dashboard Interaktif multi-tab berbasis Streamlit
├── generate_samples.py       # Pembuat gambar sampel sintetis (Alam, Makanan, Mobil, Dokumen, Unsafe)
├── pyproject.toml            # Konfigurasi dependensi dan test runner pytest
├── requirements.txt          # Daftar pustaka Python yang dibutuhkan
├── .env                      # File konfigurasi API Key (GEMINI_API_KEY, OPENAI_API_KEY)
├── README.md                 # Dokumentasi teknis proyek
│
├── data/                     # Penyimpanan data lokal
│   ├── samples/              # Gambar sampel bawaan
│   └── exports/              # File luaran ekspor CSV
│
├── tests/                    # Rangkaian Pengujian Otomatis Pytest
│   ├── test_vision_client.py           # Uji ABC, Exception, Mock, Base64, Strategy Pattern
│   ├── test_photo_collection.py        # Uji Koleksi, Cari Tag, Filter Kategori, Iterator, Counter
│   └── test_moderator_and_exporter.py  # Uji ContentModerator dan CollectionCSVExporter (SRP)
│
└── traffic_system_backup/    # Arsip cadangan sistem traffic / ETLE sebelumnya (aman tersimpan)
```

---

## 🚀 Panduan Instalasi & Persiapan

### 1. Prasyarat
Pastikan komputer Anda memiliki **Python versi 3.10** atau yang lebih baru.

### 2. Instalasi Dependensi
Jalankan salah satu perintah berikut di terminal:

```bash
# Menggunakan uv (direkomendasikan):
uv pip install -r requirements.txt

# Atau menggunakan pip standar:
pip install -r requirements.txt
```

### 3. Konfigurasi API Key
Buka berkas `.env` dan masukkan API Key Anda:
```env
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-2.5-flash
```
*(Catatan: Jika API Key tidak disetel, sistem secara otomatis beralih ke `MockVisionClient` agar seluruh 20 fitur tetap dapat didemonstrasikan 100% tanpa internet).*

### 4. Menghasilkan Gambar Sampel
Jalankan skrip generator sampel:
```bash
uv run python generate_samples.py
# atau: python generate_samples.py
```

---

## 💻 Cara Menjalankan Program

### A. Menjalankan Dashboard Web Streamlit (Rekomendasi Utama)
```bash
uv run streamlit run app.py
# atau: streamlit run app.py
```
Buka peramban di alamat `http://localhost:8501`. Fitur yang tersedia:
1. **Tab 1 - Analisis Foto Tunggal**: Pilih provider (Strategy Pattern), unggah foto atau jepret kamera, lihat caption, tags, objek, warna, OCR, dan status moderasi.
2. **Tab 2 - Analisis Batch**: Unggah banyak foto sekaligus untuk diproses secara massal dengan progress bar.
3. **Tab 3 - Komparasi 2 Provider**: Bandingkan output Gemini vs GPT-4o / Mock berdampingan pada satu foto yang sama.
4. **Tab 4 - Galeri Koleksi & Moderasi**: Cari foto by tag, filter per kategori, statistik 5 tag terpopuler (`Counter`), laporan moderasi (`safe` vs `unsafe`), dan tombol **Unduh CSV Koleksi**.

---

### B. Menjalankan via Baris Perintah (CLI)
Anda dapat menjalankan demonstrasi lengkap atau menganalisis file foto di terminal:

1. **Demonstrasi Otomatis Lengkap**:
   ```bash
   uv run python cli.py --demo
   ```
2. **Menganalisis Berkas Foto Tertentu**:
   ```bash
   uv run python cli.py --file "data/samples/pantai_sunset.jpg" --provider gemini
   ```
3. **Komparasi 2 Provider di CLI**:
   ```bash
   uv run python cli.py --compare "data/samples/kopi_dan_roti.jpg"
   ```

---

## 🧪 Menjalankan Rangkaian Pengujian Pytest

Jalankan seluruh 18 pengujian unit terisolasi dengan perintah:

```bash
uv run pytest -v
# atau: pytest -v
```

Hasil pengujian mencakup:
- ✅ `test_abc_cannot_be_instantiated`: Memastikan `VisionClient` adalah Abstract Base Class murni.
- ✅ `test_mock_vision_client_inheritance`: Memverifikasi Polimorfisme subclass.
- ✅ `test_encode_image_base64_integrity`: Memverifikasi encoding base64 UTF-8.
- ✅ `test_custom_exception_handling`: Memverifikasi eksepsi `VisionAPIException`.
- ✅ `test_strategy_pattern_execution_and_switching`: Memverifikasi Strategy Pattern runtime.
- ✅ `test_compare_providers`: Memverifikasi komparasi 2 provider.
- ✅ `test_tambah_photo`, `test_cari_by_tag`, `test_filter_kategori`: Memverifikasi manajemen koleksi.
- ✅ `test_iterator_pattern`: Memverifikasi Iterator Pattern murni (`iter()`, `next()`, `StopIteration`).
- ✅ `test_popular_tags_counter`: Memverifikasi statistik tag menggunakan `collections.Counter`.
- ✅ `test_batch_analyze`: Memverifikasi pemrosesan batch.
- ✅ `test_filter_safe_and_unsafe_photos`: Memverifikasi moderasi konten.
- ✅ `test_generate_moderation_report`: Memverifikasi laporan moderasi.
- ✅ `test_export_to_csv_integrity`: Memverifikasi ekspor CSV mematuhi prinsip SOLID (SRP).

---

## 🔒 Catatan Keamanan Data Proyek Sebelumnya
Seluruh kode sistem deteksi lalu lintas & ETLE sebelumnya telah dicadangkan secara utuh di dalam direktori `traffic_system_backup/` dan dapat diakses kapan saja untuk kebutuhan portofolio Anda.
