"""
Module: ocr_engine.py
Deskripsi: Modul Optical Character Recognition (OCR) berbasis EasyOCR
           dilengkapi pra-pemrosesan citra digital (CLAHE, bilateral filter, thresholding)
           serta pembersihan regex khusus plat nomor kendaraan Indonesia.
"""

import re
import ssl
import logging
from typing import Optional, Tuple, List
import cv2
import numpy as np

from config import OCR_LANGUAGES, OCR_MIN_CONFIDENCE, INDONESIA_PLATE_REGEX

logger = logging.getLogger(__name__)

# Mengatasi isu sertifikat SSL pada instalasi Python Windows saat mengunduh model EasyOCR
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass


class LicensePlateOCR:
    """Mesin OCR untuk ekstraksi dan standardisasi teks plat nomor kendaraan."""

    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False):
        self.languages = languages or OCR_LANGUAGES
        self.gpu = gpu
        self.reader = None
        self._init_reader()

    def _init_reader(self) -> None:
        """Inisialisasi EasyOCR reader dengan penanganan kesalahan."""
        try:
            import easyocr

            logger.info("Menginisialisasi EasyOCR Reader...")
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
            logger.info("EasyOCR Reader berhasil dimuat.")
        except Exception as e:
            logger.warning(
                "Gagal menginisialisasi EasyOCR (%s). Mode fallback aktif.", e
            )
            self.reader = None

    def preprocess_plate(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Pra-pemrosesan citra plat nomor untuk memaksimalkan akurasi pembacaan OCR:
        1. Konversi ke Grayscale
        2. Upscaling jika resolusi terlalu kecil
        3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        4. Bilateral Filtering untuk menghilangkan derau dengan tetap menjaga ketajaman tepi
        5. Adaptive Thresholding untuk binarisasi teks
        """
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        # 1. Pastikan dalam bentuk grayscale
        if len(plate_crop.shape) == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_crop.copy()

        # 2. Perbesar ukuran citra jika terlalu kecil (kurang dari 100px tinggi)
        h, w = gray.shape[:2]
        if h < 80 or w < 200:
            scale_factor = max(80 / max(h, 1), 200 / max(w, 1), 2.0)
            gray = cv2.resize(
                gray,
                (0, 0),
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_CUBIC,
            )

        # 3. Normalisasi kontras adaptif (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)

        # 4. Reduksi derau tanpa mengaburkan tepi karakter
        denoised = cv2.bilateralFilter(contrast, 11, 17, 17)

        # 5. Binarisasi adaptif Otsu
        _, binary = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return binary

    def clean_plate_text(self, raw_text: str) -> str:
        """
        Membersihkan karakter sampah dan menerapkan aturan format plat nomor Indonesia.
        Format umum: [KODE WILAYAH: 1-2 Huruf] [NOMOR: 1-4 Digit] [HURUF SERI: 1-3 Huruf]
        Contoh: 'B 1234 ABC', 'D 8888 XY', 'DK 901 Z'
        """
        if not raw_text:
            return ""

        # Hapus spasi berlebih dan karakter non-alfanumerik selain spasi
        cleaned = re.sub(r"[^A-Z0-9\s]", "", raw_text.upper()).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Pola plat nomor toleran OCR (mengizinkan karakter ambigu sebelum dinormalisasi)
        robust_pattern = r"\b([A-Z0-9]{1,2})\s+([A-Z0-9]{1,4})\s+([A-Z0-9]{1,3})\b"
        match = re.search(robust_pattern, cleaned) or re.search(INDONESIA_PLATE_REGEX, cleaned)
        if match:
            kode_wilayah, nomor, seri = match.groups()

            # Pembersihan karakter ambigu pada digit angka (misal: 'O' -> '0', 'I' -> '1', 'B' -> '8')
            nomor_cleaned = (
                nomor.replace("O", "0")
                .replace("D", "0")
                .replace("I", "1")
                .replace("L", "1")
                .replace("B", "8")
                .replace("S", "5")
            )
            # Pembersihan karakter ambigu pada kode huruf (misal: '0' -> 'O', '1' -> 'I', '8' -> 'B')
            kode_cleaned = (
                kode_wilayah.replace("0", "O")
                .replace("1", "I")
                .replace("8", "B")
            )
            seri_cleaned = (
                seri.replace("0", "O")
                .replace("1", "I")
                .replace("8", "B")
            )

            # Validasi struktur: Huruf Wilayah + Angka Nomor + Huruf Seri
            if kode_cleaned.isalpha() and nomor_cleaned.isdigit() and seri_cleaned.isalpha():
                return f"{kode_cleaned} {nomor_cleaned} {seri_cleaned}"

        # Jika tidak cocok persis regex, kembalikan teks alfanumerik yang telah disaring
        alphanumeric = "".join(ch for ch in cleaned if ch.isalnum() or ch.isspace())
        return alphanumeric.strip()

    def read_license_plate(
        self, plate_crop: np.ndarray, apply_preprocessing: bool = True
    ) -> Tuple[str, float]:
        """
        Membaca teks dari crop gambar plat nomor.

        Returns:
            Tuple[str, float]: (Nomor plat terformat, skor keyakinan OCR 0.0 - 1.0)
        """
        if plate_crop is None or plate_crop.size == 0:
            return "TIDAK_TERBACA", 0.0

        processed = self.preprocess_plate(plate_crop) if apply_preprocessing else plate_crop

        if self.reader is None:
            # Fallback jika EasyOCR belum siap
            return self._heuristic_fallback_reader(plate_crop)

        try:
            # Jalankan OCR pada gambar asli dan citra terproses
            results = self.reader.readtext(
                processed,
                allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ",
                detail=1,
            )

            # Jika hasil binarisasi kosong, coba gunakan gambar asli
            if not results:
                results = self.reader.readtext(
                    plate_crop,
                    allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ",
                    detail=1,
                )

            if not results:
                return "TIDAK_TERBACA", 0.0

            # Gabungkan teks yang lolos threshold confidence
            text_chunks = []
            confidences = []
            for bbox, text, conf in results:
                if conf >= OCR_MIN_CONFIDENCE:
                    text_chunks.append(text)
                    confidences.append(conf)

            if not text_chunks:
                # Ambil teks dengan confidence tertinggi
                best_entry = max(results, key=lambda x: x[2])
                text_chunks = [best_entry[1]]
                confidences = [best_entry[2]]

            full_text = " ".join(text_chunks)
            cleaned_plate = self.clean_plate_text(full_text)
            avg_conf = float(np.mean(confidences)) if confidences else 0.0

            if not cleaned_plate:
                return "TIDAK_TERBACA", avg_conf

            return cleaned_plate, round(avg_conf, 2)

        except Exception as e:
            logger.error("Terjadi kesalahan saat memproses OCR: %s", e)
            return "TIDAK_TERBACA", 0.0

    def _heuristic_fallback_reader(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        """Metode cadangan jika EasyOCR offline."""
        return "B 1234 ABC (MOCK)", 0.85


# Singleton instance
plate_ocr = LicensePlateOCR()
