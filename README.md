# 🚦 Sistem Cerdas Pemantauan Lalu Lintas & Identifikasi Pengendara Real-Time
### Berbasis Computer Vision (YOLOv8 + ByteTrack), EasyOCR & Integrasi Multimodal Google Gemini AI

Sistem cerdas modular siap pakai (*production-grade*) berprinsip **Object-Oriented Programming (OOP)** untuk pemantauan lalu lintas real-time, penghitungan volume kendaraan (*line crossing*), inspeksi citra statis (ETLE snapshot), pemotongan otomatis area pengendara & plat nomor, ekstraksi nomor polisi via OCR, serta validasi mendalam kepatuhan atribut keselamatan dan keabsahan dokumen berkendara (SIM/STNK) menggunakan **Google Gemini AI Multimodal** dengan **Structured Outputs (Pydantic / JSON)**.

---

## 🌟 Fitur Utama

1. **Modul Computer Vision & Pelacakan (YOLOv8 + ByteTrack)**:
   - **Deteksi Multi-Kelas Real-Time**: Mendukung kendaraan (`car`, `motorcycle`, `bus`, `truck`) dan manusia (`person` / pengendara).
   - **Pelacakan Temporal ByteTrack**: Melacak lintasan pergerakan kendaraan dan merekam jejak posisi (*tracking trails*).
   - **Penghitungan Garis Virtual (*Line Crossing*)**: Menghitung arus kendaraan yang melintasi garis hitung virtual secara presisi.
   - **Inspeksi Citra Statis (*ETLE Snapshot*)**: Mendukung inspeksi foto langsung (unggah foto atau jepret kamera) tanpa memerlukan garis virtual atau pelacakan temporal.
   - **Asosiasi Spasial Otomatis**: Menghubungkan posisi pengendara dengan unit sepeda motornya secara geometris.
   - **Auto-Crop Resolusi Tinggi**: Memotong area tubuh/kepala pengendara dan area plat nomor untuk forensik digital.

2. **Modul Visualisasi Grafis Terdedikasi (`visualizer.py`)**:
   - Arsitektur terpisah (*Single Responsibility Principle*) untuk seluruh rendering grafis OpenCV.
   - Indikator kepatuhan berkode warna: 🟩 **Hijau** (Patuh Aturan) dan 🟥 **Merah** (Pelanggaran / Tilang).
   - Head-Up Display (HUD) transparan dinamis menampilkan metrik volume kendaraan real-time.

3. **Modul OCR & Ekstraksi Plat Nomor (EasyOCR)**:
   - **Pra-pemrosesan Citra Digital Berlapis**: Konversi Grayscale, upscaling bicubic, CLAHE (*Contrast Limited Adaptive Histogram Equalization*), Bilateral Filtering, dan Otsu Adaptive Thresholding.
   - **Pembersihan Regex & Normalisasi Karakter Ambigu**: Regex cerdas nomor polisi Indonesia (`[A-Z]{1,2} [0-9]{1,4} [A-Z]{1,3}`) yang mentoleransi dan memperbaiki kesalahan umum OCR (misal: 'O'/'D' ↔ '0', 'I'/'L' ↔ '1', 'B' ↔ '8', 'S' ↔ '5').

4. **Integrasi Google Gemini AI (SDK Resmi `google-genai`)**:
   - Menggunakan model multimodal generasi terbaru (`gemini-2.5-flash` / `gemini-2.5-pro`).
   - **Strict System Instruction**: Menjamin peran petugas kepolisian forensik yang objektif dan menolak gambar tidak valid (meme, kartun, citra non-dokumen).
   - **Validasi Keselamatan Pengendara**: Mengevaluasi keberadaan helm, ketepatan pemasangan tali pengikat helm, jenis helm (Full-face, Half-face, Catok), jaket pelindung, tingkat keparahan (*severity*), dan rekomendasi tilang elektronik (*ETLE*).
   - **Validasi Dokumen Berkendara (SIM/STNK)**: Ekstraksi nomor dokumen, nama pemilik, jenis kendaraan, masa berlaku (deteksi expired), serta pencocokan otomatis (*cross-check*) antara nomor plat di dokumen dengan hasil OCR kamera.
   - **Structured Outputs**: Memanfaatkan skema terstruktur `Pydantic` (`RiderSafetyValidation` & `DocumentValidation`) via `response_schema` dan `application/json`.
   - **Graceful Fallback**: Dilengkapi mode simulasi cerdas otomatis (*mock mode*) saat API key belum disetel sehingga sistem tetap dapat didemonstrasikan secara offline tanpa crash.

5. **Manajemen Database & Ekspor Laporan (SQLite + Thread-Safe Concurrent Writes)**:
   - Penyimpanan lokal terstruktur (`data/traffic_system.db`) dengan proteksi write-lock threading.
   - Tabel: `traffic_logs`, `violations`, dan `document_validations`.
   - Ekspor laporan satu-klik ke format **CSV** dan **Microsoft Excel (.xlsx)** dengan multi-sheet.

6. **Antarmuka Pengguna Interaktif (Streamlit Dashboard) & Standalone CLI**:
   - **Streamlit Web UI (`app.py`)**: Dashboard dengan 4 tab komprehensif:
     - 📹 **Pemantauan & ETLE**: Demo video sampel, unggah video, webcam live, **unggah foto (JPG/PNG)**, dan **jepret foto langsung via kamera**.
     - 📋 **Log Pelanggaran & Bukti**: Tinjau galeri foto bukti (*rider crop* & *plate crop*) beserta hasil telaah forensik Gemini AI. Unduh laporan langsung ke format Excel/CSV.
     - 📄 **Validasi Dokumen (SIM/STNK)**: Unggah dokumen untuk verifikasi keaslian, masa berlaku, dan pencocokan nopol.
     - 📊 **Statistik Lalu Lintas**: Metrik kepatuhan helm dan diagram batang analitik volume kendaraan.
   - **CLI Runner (`main.py`)**: Runner mandiri untuk CCTV/Webcam dengan jendela visual OpenCV, penghitungan FPS, dan auto-logging.

7. **Rangkaian Pengujian Unit Otomatis (`tests/`)**:
   - Rangkaian pengujian berbasis `unittest` mencakup unit database (concurrency lock), model domain, pembersihan OCR, validasi skema Gemini, rendering grafis, hingga alur pipeline inspeksi statis.

---

## 📂 Struktur Direktori Proyek

```text
Projek OOP/
│
├── config.py                 # Pengaturan sentral (path direktori, confidence thresholds, nama model)
├── domain_models.py          # Enum tipe-aman domain (VehicleType, ViolationType, HelmetStatus, dll.)
├── database.py               # Pengelola database SQLite thread-safe dan ekspor CSV/Excel
├── detector.py               # Deteksi YOLOv8, ByteTrack, & cropping pengendara/plat
├── visualizer.py             # Rendering grafis anotasi bounding box, HUD, dan label kepatuhan
├── ocr_engine.py             # Preprocessing citra digital & pembacaan EasyOCR plat nomor
├── gemini_client.py          # Klien resmi google-genai dengan System Instruction & Skema Pydantic
├── pipeline.py               # Orkestrator alur kerja terintegrasi & ThreadPool background worker
├── app.py                    # Dashboard visual interaktif multi-tab berbasis Streamlit
├── main.py                   # Runner mandiri CLI / OpenCV untuk kamera & CCTV
├── generate_samples.py       # Generator data uji sintetis (plat nomor, STNK, foto pengendara, video)
├── requirements.txt          # Daftar dependensi pustaka Python
├── .env.example              # Template variabel lingkungan untuk Gemini API Key
├── README.md                 # Dokumentasi teknis proyek
│
└── tests/                    # Rangkaian Unit Test Otomatis
    ├── __init__.py
    ├── test_database.py          # Pengujian thread concurrency & operasi SQLite
    ├── test_domain_models.py     # Pengujian integritas Enum domain
    ├── test_gemini_client.py     # Pengujian validasi skema Pydantic & pencocokan plat
    ├── test_ocr_engine.py        # Pengujian preprocessing, regex, & normalisasi nopol
    ├── test_visualizer.py        # Pengujian rendering grafis video & gambar statis
    └── test_static_inspection.py # Pengujian deteksi statis & pipeline inspeksi ETLE
```

---

## 🚀 Panduan Instalasi & Persiapan

### 1. Prasyarat Lingkungan
Pastikan Anda telah menginstal **Python versi 3.10** atau yang lebih baru. Disarankan menggunakan pengelola paket modern [`uv`](https://github.com/astral-sh/uv) atau `pip` standar.

### 2. Instalasi Dependensi
Buka terminal pada direktori proyek, lalu jalankan salah satu perintah:

**Menggunakan `uv` (Sangat Cepat & Direkomendasikan):**
```bash
uv pip install -r requirements.txt
```

**Menggunakan `pip` standar:**
```bash
pip install -r requirements.txt
```

### 3. Konfigurasi Google Gemini API Key
Dapatkan API Key gratis di [Google AI Studio](https://aistudio.google.com/).

Salin berkas `.env.example` menjadi `.env`:
```bash
cp .env.example .env
```
*(Di Windows PowerShell: `Copy-Item .env.example .env`)*

Buka berkas `.env` dan masukkan API Key Anda:
```env
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-2.5-flash
```
*(Catatan: Jika API Key tidak diisi, sistem otomatis mengaktifkan mode simulasi cerdas / mock mode agar Anda tetap dapat menguji fungsionalitas sistem secara offline).*

### 4. Membuat Data Sampel Uji Coba
Untuk menguji coba sistem tanpa memerlukan kamera fisik atau rekaman video eksternal, buat data sintetis bawaan:
```bash
uv run python generate_samples.py
# atau: python generate_samples.py
```
Perintah ini akan membuat citra plat standar, citra STNK resmi & kedaluwarsa, citra pengendara, serta video demo `sample_traffic.mp4` di dalam folder `data/samples/`.

---

## 💻 Cara Menjalankan Program

### Opsi A: Menjalankan Dashboard Visual Streamlit (Sangat Direkomendasikan)
Jalankan perintah:
```bash
uv run streamlit run app.py
# atau: streamlit run app.py
```
Buka peramban di alamat `http://localhost:8501`. Fitur yang tersedia:
1. **Pemantauan & ETLE (Tab 1)**:
   - **Mode Video**: Pilih *Demo Video Sampel*, *Unggah Berkas Video*, atau *Webcam / Kamera (Live)*.
   - **Mode Foto Statis**: Pilih *Unggah Foto (Gambar)* atau *Ambil Foto Langsung (Kamera)*, lalu klik tombol **🔍 Proses Inspeksi ETLE (Deteksi & AI)** untuk menganalisis pelanggaran seluruh kendaraan dalam foto sekaligus.
2. **Log Pelanggaran & Bukti (Tab 2)**: Tinjau galeri foto bukti (*rider crop* & *plate crop*) beserta catatan resmi kepolisian dari Gemini AI. Unduh rekapitulasi data ke format Excel/CSV.
3. **Validasi Dokumen SIM/STNK (Tab 3)**: Unggah dokumen SIM/STNK untuk memvalidasi nomor polisi, masa berlaku, dan nama pemilik.
4. **Statistik Lalu Lintas (Tab 4)**: Pantau grafik kepatuhan pengendara dan volume jenis kendaraan.

---

### Opsi B: Menjalankan via Baris Perintah / CCTV Stream (CLI OpenCV)

1. **Menjalankan pada Video Sampel Demo**:
   ```bash
   uv run python main.py --demo
   ```

2. **Menjalankan pada Webcam / Kamera Laptop**:
   ```bash
   uv run python main.py --source 0
   ```

3. **Menjalankan pada Berkas Video Tertentu**:
   ```bash
   uv run python main.py --source "jalur_lalu_lintas.mp4"
   ```

4. **Mode Headless (Untuk Server / Edge Device tanpa Layar Monitor)**:
   ```bash
   uv run python main.py --demo --headless --max-frames 150
   ```

*Tekan tombol **`q`** pada jendela video OpenCV untuk menghentikan program dan mengekspor laporan akhir.*

---

## 🧪 Menjalankan Rangkaian Pengujian Unit (Unit Tests)

Proyek ini dilengkapi pengujian unit otomatis untuk memvalidasi integritas kode:

```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
# atau: python -m unittest discover -s tests -p "test_*.py" -v
```

Hasil pengujian mencakup 17 pengujian terisolasi:
- Penulisan konkuren multithreading pada database SQLite
- Konsistensi dan validasi nilai Enum domain
- Parsing dan skema Pydantic Gemini AI
- Ketahanan preprocessing dan regex pembersihan EasyOCR
- Integritas output rendering citra visualizer
- Deteksi statis YOLO dan pipeline orkestrasi ETLE

---

## 🔍 Skema Respon Terstruktur Gemini AI

### 1. Skema Keselamatan Pengendara (`RiderSafetyValidation`)
```json
{
  "is_helmet_present": false,
  "helmet_worn_properly": false,
  "helmet_type": "TIDAK_ADA",
  "jacket_present": false,
  "other_attributes": ["celana panjang", "sandal"],
  "violation_detected": true,
  "violation_reasons": [
    "Pengendara terdeteksi tidak menggunakan helm pelindung kepala standar.",
    "Tidak mengenakan jaket pelindung berkendara."
  ],
  "severity_level": "HIGH",
  "confidence_score": 0.95,
  "officer_notes": "Pelanggaran Pasal 106 ayat (8) jo Pasal 291 ayat (1) UU No. 22 Tahun 2009 tentang LLAJ.",
  "recommended_action": "TILANG_ELEKTRONIK"
}
```

### 2. Skema Validasi Dokumen STNK/SIM (`DocumentValidation`)
```json
{
  "document_type": "STNK",
  "document_number": "STNK-JKT-2024-88910",
  "holder_name": "BUDI SANTOSO",
  "registered_plate_number": "B 1234 ABC",
  "vehicle_type_stated": "HONDA VARIO 160 CC",
  "issue_date": "2023-05-12",
  "expiry_date": "2028-05-12",
  "is_document_valid": true,
  "is_expired": false,
  "confidence_score": 0.96,
  "notes": "Data registrasi STNK terverifikasi aktif pada pangkalan data Korlantas Polri."
}
```

---

## 🛡️ Keandalan Sistem & Arsitektur Perangkat Lunak

- **Clean OOP & Separation of Concerns**: Pemisahan jelas antara layer deteksi (`detector.py`), layer visualisasi (`visualizer.py`), layer orkestrasi (`pipeline.py`), layer domain (`domain_models.py`), layer penyimpanan data (`database.py`), dan layer antarmuka (`app.py`, `main.py`).
- **Asynchronous AI Workers**: Eksekusi API Gemini dan OCR dijalankan melalui `ThreadPoolExecutor` di background thread sehingga frame rate (FPS) pemutaran video tidak terblokir.
- **Thread-Safe SQLite Transactions**: Memanfaatkan context manager khusus dengan proteksi threading lock (`threading.Lock`) dan batas timeout koneksi untuk mencegah isu *database locked* pada sistem multithread.
- **Robust Temporary File Handling**: Mengatasi kendala Windows file locking saat pengunggahan berkas video dan eksekusi pengujian dengan penggunaan memory buffer dan temporary context.
