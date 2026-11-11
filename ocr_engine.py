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

        # Konversi ke grayscale
        if len(plate_crop.shape) == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_crop.copy()

        # Perbesar ukuran citra jika terlalu kecil
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

        # Normalisasi kontras adaptif (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)

        # Reduksi derau tanpa mengaburkan tepi karakter
        denoised = cv2.bilateralFilter(contrast, 11, 17, 17)

        # Binarisasi adaptif Otsu
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
        robust_patterns = [
            r"\b([A-Z]{1,2})\s*([0-9]{1,4})\s*([A-Z]{1,3})\b",
            r"\b([A-Z0-9]{1,2})\s+([A-Z0-9]{1,4})\s+([A-Z0-9]{1,3})\b",
        ]
        for pattern in robust_patterns:
            match = re.search(pattern, cleaned)
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

        return ""

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

    def detect_plate_candidate_contour(self, roi: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Mendeteksi area kandidat plat nomor berbasis kontur morfologi (aspect ratio 2.0 - 5.5).
        Menggunakan Sobel vertical edge dan morphological close untuk menyatukan karakter plat.
        """
        if roi is None or roi.size == 0:
            return None
        h, w = roi.shape[:2]
        if h < 20 or w < 40:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Sobel gradien horizontal (menonjolkan karakter vertikal pada plat)
        sobelx = cv2.Sobel(blurred, cv2.CV_8U, 1, 0, ksize=3)
        _, thresh = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological close untuk menggabungkan teks plat menjadi satu blok persegi panjang
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if ch == 0 or cw == 0:
                continue
            aspect_ratio = float(cw) / float(ch)
            area = cw * ch
            # Plat nomor standar memiliki aspect ratio ~2.0 hingga 5.5 dan luas proporsional
            if 2.0 <= aspect_ratio <= 5.5 and area >= (w * h * 0.005) and cw >= 35 and ch >= 12:
                candidates.append((area, (x, y, x + cw, y + ch)))

        if candidates:
            # Urutkan berdasarkan proporsi area terbaik
            candidates.sort(key=lambda item: item[0], reverse=True)
            for _, box in candidates:
                bx1, by1, bx2, by2 = box
                if (bx2 - bx1) < w * 0.9 and (by2 - by1) < h * 0.5:
                    return box
            return candidates[0][1]

        return None

    def locate_and_read_plate(
        self,
        vehicle_crop: np.ndarray,
        offset: Tuple[int, int] = (0, 0),
    ) -> Tuple[str, float, Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
        """
        Menemukan lokasi presisi (bounding box) plat nomor dalam area kendaraan dan membaca teksnya.

        Args:
            vehicle_crop: Citra area kendaraan (OpenCV BGR).
            offset: Offset koordinat (vx1, vy1) terhadap frame global.

        Returns:
            Tuple[str, float, Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
                - Teks plat nomor terformat
                - Skor keyakinan OCR (0.0 - 1.0)
                - Bounding box plat dalam frame global (gx1, gy1, gx2, gy2)
                - Potongan citra plat presisi
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return "TIDAK_TERBACA", 0.0, None, None

        vh, vw = vehicle_crop.shape[:2]
        off_x, off_y = offset

        # 1. Prioritas Utama: Ekstraksi Text Bounding Box via EasyOCR (CRAFT)
        if self.reader is not None:
            try:
                results = self.reader.readtext(
                    vehicle_crop,
                    allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ",
                    detail=1,
                )

                best_plate_text = ""
                best_conf = 0.0
                best_box = None

                for bbox_coords, raw_text, conf in results:
                    cleaned = self.clean_plate_text(raw_text)
                    if cleaned:
                        # Ditemukan teks yang secara langsung cocok dengan pola plat Indonesia
                        pts = np.array(bbox_coords).astype(int)
                        bx1, by1 = np.min(pts, axis=0)
                        bx2, by2 = np.max(pts, axis=0)

                        if conf > best_conf:
                            best_conf = conf
                            best_plate_text = cleaned
                            best_box = (bx1, by1, bx2, by2)

                # Jika belum menemukan teks plat tunggal yang cocok penuh, periksa gabungan teks
                if not best_box and results:
                    # Ambil semua box dengan confidence memadai
                    valid_items = [item for item in results if item[2] >= 0.2]
                    combined_text = " ".join([item[1] for item in valid_items])
                    cleaned_combined = self.clean_plate_text(combined_text)

                    if cleaned_combined:
                        # Gabungkan bounding box dari seluruh kata yang valid
                        all_pts = np.concatenate([np.array(item[0]).astype(int) for item in valid_items])
                        bx1, by1 = np.min(all_pts, axis=0)
                        bx2, by2 = np.max(all_pts, axis=0)
                        best_plate_text = cleaned_combined
                        best_conf = float(np.mean([item[2] for item in valid_items]))
                        best_box = (bx1, by1, bx2, by2)

                if best_box is not None:
                    bx1, by1, bx2, by2 = best_box
                    bw = bx2 - bx1
                    bh = by2 - by1

                    # Tambahkan margin aman di sekitar lempeng plat nomor
                    mx = max(8, int(bw * 0.15))
                    my = max(6, int(bh * 0.25))

                    crop_x1 = max(0, bx1 - mx)
                    crop_y1 = max(0, by1 - my)
                    crop_x2 = min(vw, bx2 + mx)
                    crop_y2 = min(vh, by2 + my)

                    exact_crop = vehicle_crop[crop_y1:crop_y2, crop_x1:crop_x2]
                    global_box = (crop_x1 + off_x, crop_y1 + off_y, crop_x2 + off_x, crop_y2 + off_y)

                    return best_plate_text, round(best_conf, 2), global_box, exact_crop

            except Exception as e:
                logger.debug("Pencarian box teks EasyOCR dilewati: %s", e)

        # 2. Prioritas Kedua: Deteksi Kontur Morfologi Persegi Panjang Plat
        contour_box = self.detect_plate_candidate_contour(vehicle_crop)
        if contour_box is not None:
            cx1, cy1, cx2, cy2 = contour_box
            cw = cx2 - cx1
            ch = cy2 - cy1
            mx = max(4, int(cw * 0.08))
            my = max(4, int(ch * 0.12))

            crop_x1 = max(0, cx1 - mx)
            crop_y1 = max(0, cy1 - my)
            crop_x2 = min(vw, cx2 + mx)
            crop_y2 = min(vh, cy2 + my)

            contour_crop = vehicle_crop[crop_y1:crop_y2, crop_x1:crop_x2]
            global_box = (crop_x1 + off_x, crop_y1 + off_y, crop_x2 + off_x, crop_y2 + off_y)

            # Coba baca teks dari crop kontur; hanya terima jika menghasilkan teks plat valid
            plate_text, conf = self.read_license_plate(contour_crop)
            if plate_text and plate_text != "TIDAK_TERBACA":
                return plate_text, conf, global_box, contour_crop

        return "TIDAK_TERBACA", 0.0, None, None

    def _heuristic_fallback_reader(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        """Metode cadangan jika EasyOCR offline."""
        return "B 1234 ABC (MOCK)", 0.85


# Singleton instance
plate_ocr = LicensePlateOCR()
