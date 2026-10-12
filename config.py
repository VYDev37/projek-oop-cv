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

for directory in [DATA_DIR, SAMPLES_DIR, EXPORTS_DIR, UPLOADS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DEFAULT_GPT4_MODEL = os.getenv("GPT4_MODEL", "gpt-4o").strip()

ALLOW_MOCK_FALLBACK = True
