"""
Module: config.py
Deskripsi: Konfigurasi terpusat untuk Sistem AI Image Analysis & Tagging Otomatis [I2].
           Mengelola pemuatan environment variable, path direktori penyimpanan,
           serta kredensial API provider Vision AI (Gemini & OpenAI).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
EXPORTS_DIR = DATA_DIR / "exports"
UPLOADS_DIR = DATA_DIR / "uploads"
EVIDENCE_DIR = DATA_DIR / "evidence"
RIDER_EVIDENCE_DIR = EVIDENCE_DIR / "riders"
PLATE_EVIDENCE_DIR = EVIDENCE_DIR / "plates"
ANNOTATED_DIR = DATA_DIR / "annotated"
DB_PATH = DATA_DIR / "traffic_monitoring.db"

for directory in [
    DATA_DIR,
    SAMPLES_DIR,
    EXPORTS_DIR,
    UPLOADS_DIR,
    EVIDENCE_DIR,
    RIDER_EVIDENCE_DIR,
    PLATE_EVIDENCE_DIR,
    ANNOTATED_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Vision Model & API Configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DEFAULT_GPT4_MODEL = os.getenv("GPT4_MODEL", "gpt-4o").strip()

ALLOW_MOCK_FALLBACK = True
MOCK_GEMINI_IF_NO_KEY = True

# Computer Vision & YOLO Configurations
DEFAULT_YOLO_MODEL = BASE_DIR / "yolov8n.pt"
DEFAULT_TRACKER = "bytetrack.yaml"
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
COUNTING_LINE_Y_RATIO = 0.65

COCO_CLASS_MAP = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}

COLOR_PALETTE = {
    "person": (255, 128, 0),
    "motorcycle": (0, 165, 255),
    "car": (0, 255, 0),
    "bus": (255, 0, 255),
    "truck": (255, 255, 0),
    "bicycle": (0, 255, 255),
    "helmet_safe": (0, 255, 0),
    "helmet_unsafe": (0, 0, 255),
    "counting_line": (255, 0, 0),
}

# OCR Configurations
OCR_LANGUAGES = ["en"]
OCR_MIN_CONFIDENCE = 0.40
INDONESIA_PLATE_REGEX = r"[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{0,3}"
