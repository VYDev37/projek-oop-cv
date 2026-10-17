# 📸 Sistem AI Image Analysis & Tagging Otomatis [I2]
### Platform Pemrosesan Citra Cerdas Berbasis Vision AI & Object-Oriented Programming (OOP)

Sistem perangkat lunak modular yang dirancang secara presisi untuk memenuhi **seluruh 20 butir Rubrik Tugas Besar Pemrograman Berorientasi Objek (OOP) Topik [I2]**: Captioning naratif, Tagging kata kunci, Deteksi objek, Kategorisasi citra, Palet warna dominan, Ekstraksi teks OCR, Moderasi konten otomatis, dan Manajemen koleksi foto cerdas.

---

## 🏛️ Arsitektur Proyek: Modul Inti Rubrik vs Modul Ekstensi

Untuk memudahkan pengujian dan penilaian akademis oleh dosen/penilai, struktur kode dipisahkan menjadi dua bagian yang jelas:

### 📦 1. Modul Inti Rubrik [I2] (Fokus Utama Penilaian Rubrik)
Seluruh 20 butir rubrik penilaian tugas besar diimplementasikan pada modul inti ini:
- **`vision_client.py`**:
  - `VisionClient(ABC)`: Abstract Base Class dengan kontrak polimorfik `@abstractmethod def analyze()`.
  - `GeminiVisionClient`: Integrasi Google GenAI SDK dengan structured JSON response.
  - `MockVisionClient`: Mode offline deterministik tanpa memerlukan API key untuk demo/testing.
  - `GPT4VisionClient`: Integrasi OpenAI Vision API via HTTP payload dengan fallback simulasi cerdas.
  - `YOLOv8VisionClient`: Engine Computer Vision lokal berbasis YOLOv8 + EasyOCR.
  - `VisionAPIException`: Custom exception handling untuk kegagalan komunikasi, input invalid, dan error parsing JSON.
  - `PhotoAnalysisResult`: Pydantic data model terstruktur (`caption`, `tags`, `objects`, `kategori`, `is_safe`, `colors`, `ocr_text`).
  - `VisionClientFactory`: Penerapan **Factory Pattern** (`create("mock" | "gemini" | "gpt4" | "yolo")`, `register()` untuk provider baru).
  - `ImageAnalysisService`: Penerapan **Strategy Pattern** untuk penukaran engine AI secara dinamis saat runtime (`set_strategy`) dan komparasi hasil 2 provider (`compare_providers`).
- **`photo_collection.py`**:
  - `PhotoCollection`: Kelas agregat penyimpan koleksi foto, pencarian tag (`cari_by_tag`), filter kategori (`filter_kategori`), dan pemrosesan batch (`batch_analyze`).
  - `PhotoCollectionIterator`: Penerapan **Iterator Pattern** native (`__iter__` dan `__next__`).
  - `get_popular_tags()`: Statistik frekuensi tag terpopuler berbasis `collections.Counter`.
  - `ContentModerator`: Pengecekan kelayakan publik (`is_safe`), filter foto unsafe, dan penyusunan laporan moderasi (`generate_moderation_report`).
  - `CollectionCSVExporter`: Penulisan berkas ekspor CSV yang terisolasi mematuhi prinsip **SOLID (Single Responsibility Principle)**.
- **`cli.py` & `main.py`**:
  - Antarmuka baris perintah (CLI) interaktif dan otomatis.
  - Fungsi `display_photo_result(result, top_n_tags=5)`: Menampilkan caption naratif dan top tag teratas per foto sesuai spesifikasi rubrik.
- **`tests/`**:
  - Rangkaian pengujian otomatis **Unit Tests di Pytest** (32 test inti rubrik + test tambahan Factory/Database) yang menguji seluruh fungsionalitas dan skenario kasus negatif (*pytest.raises*).

### 🚀 2. Modul Ekstensi Sistem Lalu Lintas Cerdas (Studi Kasus Nyata / Nilai Tambah)
Implementasi terintegrasi ke dalam studi kasus nyata pemantauan lalu lintas dan penegakan hukum elektronik (ETLE):
- **`pipeline.py`**: Pipeline pemrosesan frame video, deteksi garis hitung virtual, dan auto-logging ke koleksi foto.
- **`detector.py`**: Pelacakan objek multi-kelas (mobil, motor, bus, truk, orang) berbasis YOLOv8 dan ByteTrack.
- **`ocr_engine.py`**: Pipeline pra-pemrosesan citra plat nomor (Grayscale, Upscale, CLAHE, Denoising, Otsu) dan EasyOCR.
- **`visualizer.py`**: Visualizer anotasi bounding box dan label visual.
- **`database.py`**: Manajemen basis data SQLite multi-tabel (`traffic_logs`, `violations`, `document_validations`).
- **`app.py`**: Dashboard Web interaktif Streamlit 4 Tab otentik.

---

## 📋 Pemetaan 20 Fitur Rubrik [I2] ke Kode Sumber

### A. Fitur Dasar & Analisis (1–10)
| No | Kebutuhan Rubrik | Lokasi Implementasi | Penjelasan Teknis |
|:---:|:---|:---|:---|
| **1** | **Definisi `VisionClient` (ABC)** | `vision_client.py` (`VisionClient`) | Kelas abstrak berbasis `abc.ABC` dengan `@abstractmethod analyze()`. |
| **2** | **`GeminiVisionClient extends VisionClient`** | `vision_client.py` (`GeminiVisionClient`) | Mengirim gambar ke Google GenAI SDK (`google-genai`), parse JSON ke skema terstruktur. |
| **3** | **`MockVisionClient` (Mode Offline)** | `vision_client.py` (`MockVisionClient`) | Mode demo tanpa API key dan tanpa internet, deterministik & cepat. |
| **4** | **Raise `VisionAPIException` saat error / parse gagal** | `vision_client.py` (`VisionAPIException`) | Custom exception saat terjadi kegagalan jaringan, parsing JSON rusak (422), berkas hilang, atau input tidak didukung. |
| **5** | **`_encode_image()` (Base64)** | `vision_client.py` (`VisionClient._encode_image`) | Mengonversi file path, bytes, atau objek `PIL.Image` menjadi string base64 UTF-8. |
| **6** | **`PhotoAnalysisResult`** | `vision_client.py` (`PhotoAnalysisResult`) | Pydantic Model: `caption`, `tags`, `objects`, `kategori`, `is_safe`, `colors`, `ocr_text`. |
| **7** | **`analyze() -> PhotoAnalysisResult`** | Seluruh turunan `VisionClient` | Kontrak metode polimorfik yang mengembalikan objek `PhotoAnalysisResult`. |
| **8** | **`ContentModerator` (Cek `is_safe` & filter unsafe)** | `photo_collection.py` (`ContentModerator`) | Evaluasi kelayakan publik (aman vs konten tidak pantas/sensitif/berbahaya) dan filter foto unsafe. |
| **9** | **`PhotoCollection.tambah()`** | `photo_collection.py` (`PhotoCollection.tambah`) | Menambahkan objek hasil analisis ke dalam koleksi internal dengan validasi tipe. |
| **10**| **Tampilkan caption & top tags per foto (CLI)** | `cli.py` / `main.py` (`display_photo_result`) | Menampilkan format rapi caption naratif dan top-N tag teratas di terminal. |

### B. Fitur Koleksi & Manajemen Lanjutan (11–20)
| No | Kebutuhan Rubrik | Lokasi Implementasi | Penjelasan Teknis |
|:---:|:---|:---|:---|
| **11**| **`PhotoCollection.cari_by_tag()`** | `photo_collection.py` (`PhotoCollection.cari_by_tag`) | Pencarian foto berdasarkan tag kata kunci secara *case-insensitive*. |
| **12**| **`PhotoCollection.filter_kategori()`** | `photo_collection.py` (`PhotoCollection.filter_kategori`) | Memfilter daftar foto berdasarkan nama kategori (Alam, Makanan, Dokumen, Kendaraan, dll). |
| **13**| **Iterator pada `PhotoCollection`** | `photo_collection.py` (`PhotoCollectionIterator`) | Implementasi murni **Iterator Pattern** (`__iter__` dan `__next__`), mendukung loop native `for p in collection`. |
| **14**| **Statistik tag terpopuler (`Counter`)** | `photo_collection.py` (`PhotoCollection.get_popular_tags`) | Menggunakan `collections.Counter` untuk menghitung frekuensi tag terbanyak. |
| **15**| **Batch analysis (Banyak foto)** | `photo_collection.py` (`PhotoCollection.batch_analyze`) | Memproses sekumpulan foto sekaligus dengan penanganan error per-berkas yang *graceful*. |
| **16**| **`GPT4VisionClient extends VisionClient`** | `vision_client.py` (`GPT4VisionClient`) | Provider OpenAI GPT-4o Vision API via HTTP payload JSON dengan fallback simulasi cerdas. |
| **17**| **Komparasi hasil 2 provider / engine** | `vision_client.py` (`ImageAnalysisService.compare_providers`) | Membandingkan caption, tag bersama (*shared tags*), dan tingkat keselarasan antar 2 engine. |
| **18**| **Laporan moderasi (Safe vs Unsafe)** | `photo_collection.py` (`ContentModerator.generate_moderation_report`) | Rekapitulasi jumlah foto aman, foto tidak aman (unsafe), dan persentase kepatuhan kelayakan publik. |
| **19**| **Ekspor koleksi ke CSV (SOLID: SRP)** | `photo_collection.py` (`CollectionCSVExporter`) | Kelas terpisah khusus penulisan file CSV agar mematuhi *Single Responsibility Principle*. |
| **20**| **Design Pattern: Strategy** | `vision_client.py` (`ImageAnalysisService`) | Menukar engine Vision AI aktif secara dinamis saat runtime (`set_strategy`). |

---

## 🏛️ Konsep OOP yang Diterapkan

1. **Inheritance & Abstraction (ABC)**:
   - `VisionClient(ABC)` sebagai superclass abstrak dengan metode `@abstractmethod analyze()` yang diturunkan oleh `GeminiVisionClient`, `GPT4VisionClient`, `YOLOv8VisionClient`, dan `MockVisionClient`.
2. **Polymorphism**:
   - Pemanggilan metode seragam `client.analyze(image)` dapat dieksekusi pada instansiasi provider apa pun tanpa membedakan implementasi internalnya.
3. **Exception Handling**:
   - Kelas eksepsi terdedikasi `VisionAPIException` untuk menangkap galat inferensi, API error, parsing JSON cacat, maupun berkas yang tidak ditemukan.
4. **Design Pattern: Strategy Pattern**:
   - `ImageAnalysisService` bertindak sebagai *Context* yang memegang objek strategi `VisionClient` dan dapat ditukar kapan saja saat runtime.
5. **Design Pattern: Iterator Pattern**:
   - `PhotoCollection` mengembalikan objek `PhotoCollectionIterator` sehingga mendukung protokol iterasi native Python (`__iter__` dan `__next__`, melempar `StopIteration` saat habis).
6. **Prinsip SOLID**:
   - **SRP (Single Responsibility Principle)**: `ContentModerator` hanya bertanggung jawab atas kepatuhan/moderasi, `CollectionCSVExporter` hanya bertanggung jawab atas I/O ekspor CSV, dan `PhotoCollection` hanya mengelola struktur data koleksi.
   - **OCP (Open/Closed Principle)**: Menambah engine baru cukup membuat subclass baru turunan `VisionClient` tanpa mengubah kode layanan (`ImageAnalysisService`).
   - **LSP & DIP (Liskov Substitution & Dependency Inversion)**: `ImageAnalysisService` bergantung pada abstraksi `VisionClient`, bukan implementasi konkretnya.

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
Buka berkas `.env` dan masukkan API Key Google Gemini atau OpenAI (opsional, sistem memiliki engine lokal YOLOv8 dan Mock offline):
```env
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-2.5-flash
OPENAI_API_KEY=sk-...
```

---

## 💻 Cara Menjalankan Program

### A. Menjalankan via Baris Perintah (CLI)
Anda dapat menjalankan CLI interaktif atau menganalisis file foto langsung di terminal:

1. **Menganalisis Berkas Foto (Caption & Top Tags - Rubrik Item 10)**:
   ```bash
   uv run python cli.py --file "data/samples/kopi_dan_roti.jpg" --provider mock
   ```
2. **Komparasi 2 Engine Vision AI Berdampingan (Rubrik Item 17)**:
   ```bash
   uv run python cli.py --compare "data/samples/kopi_dan_roti.jpg"
   ```
3. **Menganalisis Foto dengan Engine YOLOv8 Local Computer Vision**:
   ```bash
   uv run python cli.py --file "data/samples/mobil_sedan.jpg" --provider yolo
   ```
4. **Menjalankan Pemantauan Video Real-Time**:
   ```bash
   uv run python cli.py --demo
   # atau: uv run python cli.py --video "path/ke/video.mp4"
   ```
5. **Menu Interaktif Lengkap**:
   ```bash
   uv run python cli.py
   ```

---

### B. Menjalankan Dashboard Web Streamlit
```bash
uv run streamlit run app.py
# atau: streamlit run app.py
```
Akses di browser pada alamat `http://localhost:8501`. Fitur 4 Tab mencakup:
1. **🎥 Tab 1 - Live Stream & Monitoring**: Pemantauan video CCTV / webcam / demo video dengan garis hitung virtual (*line crossing*), serta mode snapshot inspeksi citra statis.
2. **🚨 Tab 2 - Log Pelanggaran & Bukti**: Buku register pelanggaran lalu lintas terintegrasi database SQLite, galeri foto forensik pengendara & plat nomor, dan tombol ekspor CSV/Excel.
3. **📄 Tab 3 - Validasi Dokumen (SIM/STNK)**: Verifikasi keabsahan dokumen berkendara via multimodal OCR dan *cross-check* pencocokan nomor polisi terhadap tangkapan kamera.
4. **📊 Tab 4 - Statistik & Galeri Koleksi**: Visualisasi metrik arus lalu lintas, galeri foto hasil analisis (`PhotoCollection`), pencarian tag (`cari_by_tag`), filter kategori (`filter_kategori`), statistik 8 tag terpopuler (`collections.Counter`), laporan moderasi keselamatan (`ContentModerator`), dan tombol ekspor data koleksi ke CSV (SOLID: SRP).

---

## 🧪 Rangkaian Pengujian Pytest (32 Unit Tests)

Rangkaian 32 pengujian otomatis terisolasi telah disiapkan pada folder `tests/` untuk memverifikasi fungsionalitas normal maupun **skenario kasus salah (Negative Testing / Exception Handling)**:

```bash
uv run pytest -v
# atau: pytest -v
```

### Rincian Cakupan Pengujian:
1. **Hierarki & Polimorfisme `VisionClient`**:
   - `test_abc_cannot_be_instantiated`: Memastikan kelas abstrak tidak dapat diinstansiasi langsung (`TypeError`).
   - `test_mock_vision_client_inheritance` & `test_yolo_vision_client_inheritance`: Memverifikasi pewarisan polimorfik.
   - `test_encode_image_base64_integrity`: Memverifikasi konversi citra ke string base64.
2. **Skenario Kasus Negatif & Pembuktian Exception (`VisionAPIException`)**:
   - `test_encode_image_nonexistent_file_raises_exception`: Memastikan `VisionAPIException` dilempar jika path berkas tidak ditemukan.
   - `test_encode_image_invalid_type_raises_exception`: Memastikan `VisionAPIException` dilempar jika tipe input bukan gambar.
   - `test_strategy_pattern_rejects_invalid_type`: Memastikan `TypeError` dilempar jika objek bukan turunan `VisionClient`.
   - `test_photo_collection_tambah_invalid_type`: Memastikan `TypeError` dilempar jika objek bukan `PhotoAnalysisResult`.
   - `test_batch_analyze_handles_exceptions_gracefully`: Memverifikasi batch analysis tidak *crash* saat menemui berkas yang korup/hilang.
3. **Pengujian `GeminiVisionClient`**:
   - `test_gemini_vision_client_missing_client_raises_exception`: Memverifikasi pelemparan `VisionAPIException` status 401 jika API key belum disetel.
   - `test_gemini_vision_client_valid_json_response`: Memverifikasi parsing structured JSON yang berhasil.
   - `test_gemini_vision_client_corrupt_json_raises_exception`: **Uji respons JSON rusak / korup** yang memicu `VisionAPIException` status 422.
   - `test_gemini_vision_client_empty_response_raises_exception`: Memverifikasi penanganan respons kosong (500).
4. **Pengujian `GPT4VisionClient`**:
   - `test_gpt4_vision_client_simulated_fallback`: Memverifikasi mode simulasi saat tanpa API key.
   - `test_gpt4_vision_client_api_error_raises_exception`: Memverifikasi respons error dari API OpenAI memicu `VisionAPIException`.
5. **Pengujian Koleksi, Moderasi, & Ekspor (SOLID)**:
   - `test_iterator_pattern`: Memverifikasi protokol `__iter__`, `__next__`, dan `StopIteration`.
   - `test_cari_by_tag` & `test_filter_kategori`: Memverifikasi pencarian dan filter data.
   - `test_popular_tags_counter`: Memverifikasi penghitungan tag via `collections.Counter`.
   - `test_filter_safe_and_unsafe_photos`: Memverifikasi moderasi konten (`is_safe`).
   - `test_generate_moderation_report`: Memverifikasi laporan kepatuhan moderasi.
   - `test_export_to_csv_integrity`: Memverifikasi integritas penulisan CSV oleh `CollectionCSVExporter` (SRP).
