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
| **16**| **`YOLOv8VisionClient extends VisionClient`** | `vision_client.py` (`YOLOv8VisionClient`) | Provider Computer Vision lokal (YOLOv8 + EasyOCR) untuk deteksi objek, plat nomor, dan evaluasi helm. |
| **17**| **Komparasi hasil 2 provider / engine** | `vision_client.py` (`ImageAnalysisService.compare_providers`) | Membandingkan caption, tag bersama (*shared tags*), dan tingkat keselarasan antar 2 AI/engine. |
| **18**| **Laporan moderasi (Safe vs Unsafe)** | `photo_collection.py` (`ContentModerator.generate_moderation_report`) | Rekapitulasi jumlah foto tertib/aman, pelanggaran, dan persentase kepatuhan. |
| **19**| **Ekspor koleksi ke CSV (SOLID: SRP)** | `photo_collection.py` (`CollectionCSVExporter`) | Kelas terpisah khusus penulisan file CSV agar mematuhi *Single Responsibility Principle*. |
| **20**| **Design Pattern: Strategy** | `vision_client.py` (`ImageAnalysisService`) | Menukar engine Vision AI aktif secara dinamis saat runtime (`set_strategy`). |

---

## 🏛️ Konsep OOP yang Diterapkan

1. **Inheritance & Abstraction (ABC)**:
   - `VisionClient(ABC)` sebagai kelas induk abstrak yang mewariskan fungsionalitas ke `YOLOv8VisionClient`, `GeminiVisionClient`, `GPT4VisionClient`, dan `MockVisionClient`.
2. **Polymorphism**:
   - Pemanggilan metode `client.analyze(image)` dieksekusi secara seragam terlepas dari provider/engine mana yang aktif.
3. **Exception Handling**:
   - Kelas eksepsi terdedikasi `VisionAPIException` untuk menangkap galat inferensi, API, dan parsing data.
4. **Design Pattern: Strategy Pattern**:
   - `ImageAnalysisService` bertindak sebagai *Context* yang memegang objek strategi bertipe `VisionClient`.
5. **Design Pattern: Iterator Pattern**:
   - `PhotoCollection` mengembalikan objek `PhotoCollectionIterator` sehingga mendukung protokol iterasi native Python (`__iter__` dan `__next__`).
6. **Prinsip SOLID**:
   - **SRP (Single Responsibility Principle)**: Pemisahan tugas yang tegas antara analisis AI (`VisionClient`), penyimpanan koleksi (`PhotoCollection`), evaluasi keselamatan/moderasi (`ContentModerator`), dan ekspor CSV (`CollectionCSVExporter`).
   - **OCP (Open/Closed Principle)**: Menambah engine baru cukup membuat subclass baru turunan `VisionClient` tanpa merombak kode yang ada.
   - **LSP & DIP (Liskov Substitution & Dependency Inversion)**: `ImageAnalysisService` bergantung pada abstraksi `VisionClient`, bukan implementasi konkretnya.
7. **Unit Testing Pytest**:
   - Rangkaian 20 pengujian otomatis mandiri di folder `tests/` mencakup seluruh logika dan relasi kelas.

---

## 📂 Struktur Direktori Proyek

```text
Projek OOP/
│
├── config.py                 # Pengaturan konfigurasi, YOLO, OCR, dan direktori data
├── vision_client.py          # VisionClient (ABC), YOLOv8, Gemini, Mock, Strategy Service
├── photo_collection.py       # PhotoCollection, Iterator Pattern, ContentModerator, CSV Exporter (SRP)
├── pipeline.py               # Pipeline kontinu pelacakan kendaraan, crossing line & auto-log
├── detector.py               # Detector YOLOv8 + ByteTrack untuk video streaming
├── visualizer.py             # Renderer visualizer anotasi bounding box (SRP)
├── ocr_engine.py             # Optical Character Recognition (EasyOCR) plat nomor
├── database.py               # SQLite logging & pelaporan
├── cli.py                    # Runner baris perintah (CLI) untuk demonstrasi lengkap
├── app.py                    # Web UI Dashboard Interaktif 5 Tab berbasis Streamlit
├── yolov8n.pt                # Bobot model YOLOv8 Nano
├── generate_samples.py       # Pembuat gambar sampel sintetis
├── pyproject.toml            # Konfigurasi dependensi dan test runner pytest
├── requirements.txt          # Daftar pustaka Python yang dibutuhkan
├── .env                      # File konfigurasi API Key
├── README.md                 # Dokumentasi teknis proyek
│
├── data/                     # Penyimpanan data lokal
│   ├── samples/              # Gambar sampel bawaan
│   ├── annotated/            # Citra hasil deteksi beranotasi bounding box
│   ├── evidence/             # Tangkapan bukti pengendara dan plat nomor
│   └── exports/              # File luaran ekspor CSV
│
├── tests/                    # Rangkaian Pengujian Otomatis Pytest (20 Unit Tests)
│   ├── test_vision_client.py           # Uji ABC, Exception, Mock, YOLOv8, Base64, Strategy Pattern
│   ├── test_photo_collection.py        # Uji Koleksi, Cari Tag, Filter Kategori, Iterator, Counter
│   └── test_moderator_and_exporter.py  # Uji ContentModerator dan CollectionCSVExporter (SRP)
│
└── traffic_system_backup/    # Arsip cadangan sistem traffic sebelumnya
```

---

## 🚀 Panduan Instalasi & Persiapan

### 1. Prasyarat
Pastikan komputer Anda memiliki **Python versi 3.10** atau yang lebih baru.

### 2. Instalasi Dependensi
Jalankan perintah berikut di terminal:

```bash
uv pip install -r requirements.txt
# atau: pip install -r requirements.txt
```

### 3. Konfigurasi API Key (Opsional)
Buka berkas `.env` dan masukkan API Key Google Gemini (opsional, sistem memiliki engine lokal YOLOv8 dan Mock):
```env
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-2.5-flash
```

---

## 💻 Cara Menjalankan Program

### A. Menjalankan Dashboard Web Streamlit (Rekomendasi Utama)
```bash
uv run streamlit run app.py
# atau: streamlit run app.py
```
Buka peramban di alamat `http://localhost:8501`. Fitur yang tersedia:
1. **Tab 1 - Analisis Citra (Foto / Kamera)**: Upload gambar atau ambil foto langsung lewat webcam, pilih engine (YOLOv8 Local CV, Gemini, Mock), lihat bounding box, plat nomor OCR, caption, dan evaluasi kepatuhan aturan.
2. **Tab 2 - Pemantauan Video & Live Stream**: Jalankan tracking video rekaman atau live webcam secara kontinu dengan garis hitung virtual (*virtual line crossing*) dan auto-logging ke koleksi.
3. **Tab 3 - Analisis Batch**: Unggah banyak foto sekaligus untuk diproses secara massal dengan progress bar.
4. **Tab 4 - Komparasi 2 Engine**: Bandingkan output YOLOv8 Local CV vs Gemini Multimodal AI pada citra yang sama.
5. **Tab 5 - Galeri Koleksi & Moderasi**: Cari foto by tag, filter per kategori, statistik 5 tag terpopuler (`Counter`), laporan kepatuhan keselamatan, dan tombol **Unduh CSV Koleksi** (SOLID: SRP).

---

### B. Menjalankan via Baris Perintah (CLI)
Anda dapat menjalankan demonstrasi lengkap atau menganalisis file foto di terminal:

1. **Demonstrasi Otomatis Lengkap**:
   ```bash
   uv run python cli.py --demo
   ```
2. **Menganalisis Berkas Foto Tertentu dengan YOLOv8**:
   ```bash
   uv run python cli.py --file "data/samples/mobil_sedan.jpg" --provider yolo
   ```
3. **Komparasi 2 Engine di CLI**:
   ```bash
   uv run python cli.py --compare "data/samples/kopi_dan_roti.jpg"
   ```
4. **Pemantauan Video di Terminal**:
   ```bash
   uv run python cli.py --video "path/ke/video.mp4"
   ```

---

## 🧪 Menjalankan Rangkaian Pengujian Pytest

Jalankan seluruh 20 pengujian unit terisolasi dengan perintah:

```bash
uv run pytest -v
# atau: pytest -v
```

Hasil pengujian mencakup:
- ✅ `test_abc_cannot_be_instantiated`: Memastikan `VisionClient` adalah Abstract Base Class murni.
- ✅ `test_mock_vision_client_inheritance`: Memverifikasi Polimorfisme subclass Mock.
- ✅ `test_yolo_vision_client_inheritance`: Memverifikasi Polimorfisme subclass YOLOv8.
- ✅ `test_encode_image_base64_integrity`: Memverifikasi encoding base64 UTF-8.
- ✅ `test_custom_exception_handling`: Memverifikasi eksepsi `VisionAPIException`.
- ✅ `test_mock_vision_client_analyze_safe` & `test_mock_vision_client_analyze_unsafe`: Verifikasi inferensi mock.
- ✅ `test_yolo_vision_client_analyze`: Verifikasi inferensi deteksi objek YOLOv8 & anotasi bounding box.
- ✅ `test_strategy_pattern_execution_and_switching`: Memverifikasi penukaran strategi runtime.
- ✅ `test_strategy_type_enforcement`: Memverifikasi penolakan strategi yang bukan turunan `VisionClient`.
- ✅ `test_compare_providers`: Memverifikasi komparasi 2 engine.
- ✅ `test_tambah_photo`, `test_cari_by_tag`, `test_filter_kategori`: Memverifikasi manajemen koleksi.
- ✅ `test_iterator_pattern`: Memverifikasi Iterator Pattern murni (`iter()`, `next()`, `StopIteration`).
- ✅ `test_popular_tags_counter`: Memverifikasi statistik tag menggunakan `collections.Counter`.
- ✅ `test_batch_analyze`: Memverifikasi pemrosesan batch.
- ✅ `test_filter_safe_and_unsafe_photos`: Memverifikasi pemisahan data aman dan pelanggaran.
- ✅ `test_generate_moderation_report`: Memverifikasi laporan kepatuhan/keselamatan.
- ✅ `test_export_to_csv_integrity`: Memverifikasi ekspor CSV mematuhi prinsip SOLID (SRP).

---

## 🔒 Catatan Keamanan Data Proyek Sebelumnya
Seluruh kode sistem deteksi lalu lintas & ETLE sebelumnya telah dicadangkan secara utuh di dalam direktori `traffic_system_backup/` dan dapat diakses kapan saja untuk kebutuhan portofolio Anda.
