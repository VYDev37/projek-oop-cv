"""
Module: config.py
Deskripsi: Pengaturan konfigurasi sentral sistem pemantauan lalu lintas cerdas,
           termasuk path direktori, parameter computer vision, dan API credentials.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Muat variabel lingkungan dari file .env jika ada
load_dotenv()

# ==============================================================================
# 1. PATH DIREKTORI & PENYIMPANAN
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EVIDENCE_DIR = DATA_DIR / "evidence"
RIDER_EVIDENCE_DIR = EVIDENCE_DIR / "riders"
PLATE_EVIDENCE_DIR = EVIDENCE_DIR / "plates"
DOCUMENT_DIR = EVIDENCE_DIR / "documents"
EXPORTS_DIR = DATA_DIR / "exports"
MODELS_DIR = DATA_DIR / "models"
SAMPLES_DIR = DATA_DIR / "samples"

DB_PATH = DATA_DIR / "traffic_system.db"


def ensure_directories() -> None:
    """Membuat direktori kerja yang dibutuhkan jika belum tersedia."""
    directories = [
        DATA_DIR,
        EVIDENCE_DIR,
        RIDER_EVIDENCE_DIR,
        PLATE_EVIDENCE_DIR,
        DOCUMENT_DIR,
        EXPORTS_DIR,
        MODELS_DIR,
        SAMPLES_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Pastikan direktori dibuat saat modul diimpor
ensure_directories()

# ==============================================================================
# 2. KONFIGURASI GEMINI AI (google-genai SDK)
# ==============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Model default multimodal untuk analisis cepat dan terstruktur
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# Mode Simulasi / Mock jika API Key belum diatur (mencegah crash saat pengujian)
MOCK_GEMINI_IF_NO_KEY = True

# ==============================================================================
# 3. KONFIGURASI COMPUTER VISION (YOLO & TRACKING)
# ==============================================================================
# Model YOLOv8 (akan diunduh otomatis oleh ultralytics saat pertama kali dijalankan)
DEFAULT_YOLO_MODEL = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")

# Tracker configuration (ByteTrack standar)
DEFAULT_TRACKER = os.getenv("TRACKER_CONFIG", "bytetrack.yaml")

# Threshold Deteksi
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

# Pemetaan ID Kelas COCO dataset untuk deteksi lalu lintas
COCO_CLASS_MAP = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Target kelas kendaraan yang dihitung dan dipantau
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

# Warna visualisasi bounding box (Format BGR OpenCV)
COLOR_PALETTE = {
    "car": (0, 200, 255),         # Kuning cerah
    "motorcycle": (255, 128, 0),  # Biru laut
    "bus": (255, 0, 255),         # Magenta
    "truck": (0, 165, 255),       # Oranye
    "person": (255, 255, 0),      # Cyan
    "violation": (0, 0, 255),     # Merah (Pelanggaran)
    "compliant": (0, 255, 0),     # Hijau (Patuh)
    "counting_line": (255, 0, 0), # Biru murni
}

# Posisi garis hitung virtual (rasio tinggi frame 0.0 - 1.0)
COUNTING_LINE_Y_RATIO = 0.65

# Batas jeda waktu sebelum memicu deteksi ulang pada track_id yang sama (detik)
VIOLATION_COOLDOWN_SECONDS = 5.0

# ==============================================================================
# 4. KONFIGURASI OCR PLAT NOMOR
# ==============================================================================
OCR_LANGUAGES = ["en"]
OCR_MIN_CONFIDENCE = 0.25

# Format Regex plat nomor kendaraan Indonesia standar (e.g. 'B 1234 ABC', 'D 999 XY')
INDONESIA_PLATE_REGEX = r"([A-Z]{1,2})\s*([0-9]{1,4})\s*([A-Z]{1,3})"
