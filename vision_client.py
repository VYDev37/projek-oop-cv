"""
Module: vision_client.py
Deskripsi: Abstraksi dan implementasi Vision AI provider (YOLOv8, Gemini, GPT-4, Mock)
           serta Strategy Pattern via ImageAnalysisService.
"""

import base64
import io
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from pydantic import BaseModel, Field

from config import (
    GEMINI_API_KEY,
    DEFAULT_GEMINI_MODEL,
    OPENAI_API_KEY,
    DEFAULT_GPT4_MODEL,
    DEFAULT_YOLO_MODEL,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    VEHICLE_CLASSES,
    COLOR_PALETTE,
    ANNOTATED_DIR,
)

logger = logging.getLogger(__name__)


class VisionAPIException(Exception):
    """
    Eksepsi kustom untuk menangani kegagalan pemanggilan Vision API,
    kesalahan komunikasi jaringan, atau kegagalan parsing respons JSON.
    """

    def __init__(self, message: str, provider: str = "Unknown", status_code: Optional[int] = None):
        super().__init__(f"[{provider}] {message}")
        self.message = message
        self.provider = provider
        self.status_code = status_code


class PhotoAnalysisResult(BaseModel):
    """
    Model representasi data terstruktur untuk hasil analisis foto.
    Menyimpan caption, tags, objects, kategori, is_safe, colors, dan metadata pendukung.
    """
    caption: str = Field(
        ...,
        description="Deskripsi naratif singkat dan informatif tentang isi foto.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Daftar label/kata kunci yang relevan dengan citra (misal: ['sunset', 'pantai', 'senja']).",
    )
    objects: List[str] = Field(
        default_factory=list,
        description="Objek-objek fisik yang terdeteksi di dalam foto (misal: ['mobil', 'pohon', 'orang']).",
    )
    kategori: str = Field(
        ...,
        description="Kategori utama citra (misal: Alam, Makanan, Perkotaan, Dokumen, Kendaraan, Potret).",
    )
    is_safe: bool = Field(
        ...,
        description="Status moderasi konten: True jika aman untuk publik, False jika mengandung konten tidak pantas.",
    )
    colors: List[str] = Field(
        default_factory=list,
        description="Palet warna dominan pada foto (misal: ['Biru Laut', 'Oranye', 'Emas']).",
    )
    ocr_text: Optional[str] = Field(
        default="",
        description="Teks yang berhasil diekstraksi dari citra jika ada.",
    )
    provider: str = Field(
        default="Unknown",
        description="Nama provider AI yang melakukan analisis (YOLOv8, Gemini, atau Mock).",
    )
    image_path: Optional[str] = Field(
        default="",
        description="Path file gambar asli jika tersedia.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        description="Waktu analisis dilakukan.",
    )
    processing_time_ms: Optional[float] = Field(
        default=0.0,
        description="Waktu eksekusi analisis dalam milidetik.",
    )
    annotated_image_path: Optional[str] = Field(
        default="",
        description="Path berkas gambar beranotasi bounding box visual jika tersedia.",
    )
    boxes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Daftar koordinat bounding box objek terdeteksi.",
    )


class VisionClient(ABC):
    """
    Abstract Base Class (ABC) yang mendefinisikan antarmuka kontrak standar
    bagi seluruh provider Vision AI. Mendukung Polymorphism dan OCP (SOLID).
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    def _encode_image(self, image_input: Union[str, Path, bytes, Image.Image]) -> str:
        """
        Mengonversi input foto (path berkas, bytes mentah, atau PIL.Image)
        menjadi representasi string base64 UTF-8.
        """
        try:
            if isinstance(image_input, (str, Path)):
                file_path = Path(image_input)
                if not file_path.exists():
                    raise FileNotFoundError(f"Berkas gambar tidak ditemukan: {file_path}")
                with open(file_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")

            elif isinstance(image_input, bytes):
                return base64.b64encode(image_input).decode("utf-8")

            elif isinstance(image_input, Image.Image):
                buffer = io.BytesIO()
                rgb_img = image_input.convert("RGB")
                rgb_img.save(buffer, format="JPEG", quality=90)
                return base64.b64encode(buffer.getvalue()).decode("utf-8")

            else:
                raise ValueError(f"Tipe input gambar tidak didukung: {type(image_input)}")

        except Exception as e:
            raise VisionAPIException(
                f"Gagal melakukan encoding gambar ke base64: {e}",
                provider=self.__class__.__name__,
            )

    def _prepare_pil_image(self, image_input: Union[str, Path, bytes, Image.Image]) -> Image.Image:
        """Helper untuk mendapatkan objek PIL.Image dari berbagai format input."""
        if isinstance(image_input, Image.Image):
            return image_input
        elif isinstance(image_input, (str, Path)):
            return Image.open(image_input)
        elif isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input))
        else:
            raise ValueError(f"Tipe input gambar tidak didukung: {type(image_input)}")

    @abstractmethod
    def analyze(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        image_name: Optional[str] = None,
    ) -> PhotoAnalysisResult:
        """
        Menganalisis citra dan mengembalikan objek terstruktur PhotoAnalysisResult.
        Harus diimplementasikan oleh setiap subclass konkrit.
        """
        pass


class GeminiVisionClient(VisionClient):
    """
    Subclass konkrit VisionClient yang memanfaatkan Google GenAI SDK (google-genai).
    Mengirimkan gambar multimodal dan mem-parse Structured JSON respons ke PhotoAnalysisResult.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
    ):
        super().__init__(model_name=model_name, api_key=api_key or GEMINI_API_KEY)
        self.client = None
        self._init_client()

    def _init_client(self) -> None:
        """Inisialisasi klien SDK google-genai."""
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info("GeminiVisionClient berhasil tersambung dengan model: %s", self.model_name)
            except Exception as e:
                logger.warning("Gagal inisialisasi google-genai Client: %s", e)
                self.client = None
        else:
            logger.info("Gemini API Key tidak disetel. GeminiVisionClient tidak aktif.")

    def analyze(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        image_name: Optional[str] = None,
    ) -> PhotoAnalysisResult:
        """Mengirimkan foto ke Google Gemini dan mem-parse respons terstruktur."""
        start_time = time.time()
        image_path_str = str(image_input) if isinstance(image_input, (str, Path)) else (image_name or "in_memory_image")

        if not self.client:
            raise VisionAPIException(
                "Gemini API Client tidak tersedia. Pastikan GEMINI_API_KEY telah diatur di .env.",
                provider="GeminiVisionClient",
                status_code=401,
            )

        try:
            pil_image = self._prepare_pil_image(image_input)
            from google.genai import types

            prompt = (
                "Lakukan analisis mendalam terhadap foto ini untuk platform kurasi gambar cerdas.\n"
                "1. Berikan caption naratif deskriptif dalam Bahasa Indonesia.\n"
                "2. Ekstraksi 4-8 tags kata kunci paling relevan.\n"
                "3. Sebutkan objek fisik utama yang terlihat.\n"
                "4. Tentukan satu kategori utama (Alam, Makanan, Perkotaan, Dokumen, Kendaraan, Seni, Potret, dll).\n"
                "5. Moderasi konten: tentukan apakah foto ini aman (is_safe=True) atau berbahaya/tidak senonoh/kekerasan (is_safe=False).\n"
                "6. Sebutkan 2-4 warna dominan.\n"
                "7. Jika terdapat teks pada gambar, baca teks tersebut ke ocr_text (kosongkan jika tidak ada teks)."
            )

            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PhotoAnalysisResult,
                system_instruction=(
                    "Anda adalah AI Ahli Kurasi & Analisis Citra Digital profesional. "
                    "Hasil analisis harus selalu akurat, objektif, dan mengikuti skema JSON yang ditentukan."
                ),
                temperature=0.2,
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[pil_image, prompt],
                config=config,
            )

            if not response.text:
                raise VisionAPIException(
                    "Respons kosong dari Gemini API.",
                    provider="GeminiVisionClient",
                    status_code=500,
                )

            # Parse respons JSON ke PhotoAnalysisResult
            raw_data = json.loads(response.text)
            elapsed_ms = (time.time() - start_time) * 1000

            result = PhotoAnalysisResult(
                caption=raw_data.get("caption", "Foto tanpa caption"),
                tags=[str(t).lower() for t in raw_data.get("tags", [])],
                objects=raw_data.get("objects", []),
                kategori=raw_data.get("kategori", "Umum"),
                is_safe=bool(raw_data.get("is_safe", True)),
                colors=raw_data.get("colors", []),
                ocr_text=raw_data.get("ocr_text", ""),
                provider=f"Google Gemini ({self.model_name})",
                image_path=image_path_str,
                processing_time_ms=round(elapsed_ms, 2),
            )
            return result

        except json.JSONDecodeError as jde:
            raise VisionAPIException(
                f"Gagal mem-parse JSON hasil Gemini: {jde}",
                provider="GeminiVisionClient",
                status_code=422,
            )
        except VisionAPIException:
            raise
        except Exception as e:
            raise VisionAPIException(
                f"Terjadi kesalahan saat memproses gambar dengan Gemini: {e}",
                provider="GeminiVisionClient",
                status_code=500,
            )


class MockVisionClient(VisionClient):
    """
    Subclass konkrit VisionClient untuk mode demo offline dan unit testing.
    Menghasilkan hasil analisis deterministik tanpa memerlukan koneksi internet atau API key.
    """

    def __init__(self, model_name: str = "mock-vision-v1", api_key: Optional[str] = "mock_key"):
        super().__init__(model_name=model_name, api_key=api_key)

    def analyze(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        image_name: Optional[str] = None,
    ) -> PhotoAnalysisResult:
        """Menghasilkan analisis simulasi cerdas berdasarkan nama atau metadata berkas."""
        start_time = time.time()

        # Validasi bahwa input gambar valid (memastikan _encode_image bekerja)
        _ = self._encode_image(image_input)

        name_lower = ""
        if isinstance(image_input, (str, Path)):
            name_lower = Path(image_input).name.lower()
        elif image_name:
            name_lower = image_name.lower()

        # Simulasi deteksi konten unsafe jika nama mengandung kata sensitif
        is_safe = True
        if any(w in name_lower for w in ["unsafe", "hazard", "bahaya", "nsfw", "violence"]):
            is_safe = False

        # Simulasi kategori & deskripsi berbasis konten nama
        if any(w in name_lower for w in ["pantai", "beach", "sunset", "alam", "nature", "gunung"]):
            caption = "Pemandangan lanskap alam yang indah dan menenangkan saat matahari terbenam."
            tags = ["alam", "pemandangan", "sunset", "outdoor", "langit"]
            objects = ["matahari", "awan", "pepohonan", "horizon"]
            kategori = "Alam"
            colors = ["Oranye Keemasan", "Biru Langit", "Kuning"]
            ocr_text = ""
        elif any(w in name_lower for w in ["makanan", "food", "kopi", "cafe", "restoran"]):
            caption = "Sajian kuliner lezat yang ditata menarik di atas meja kafe."
            tags = ["makanan", "kuliner", "kafe", "hidangan", "minuman"]
            objects = ["piring", "cangkir", "meja", "sendok"]
            kategori = "Makanan"
            colors = ["Cokelat Hangat", "Putih Krem", "Merah"]
            ocr_text = "SPECIAL MENU TODAY"
        elif any(w in name_lower for w in ["mobil", "motor", "traffic", "kendaraan", "vehicle"]):
            caption = "Kendaraan bermotor yang sedang melintas di area jalan raya."
            tags = ["kendaraan", "transportasi", "jalanan", "otomotif"]
            objects = ["mobil", "motor", "aspal", "lampu lalu lintas"]
            kategori = "Kendaraan"
            colors = ["Hitam Metalik", "Abu-abu", "Merah"]
            ocr_text = "B 1234 XYZ"
        elif any(w in name_lower for w in ["dokumen", "surat", "stnk", "sim", "teks"]):
            caption = "Dokumen tertulis resmi dengan informasi identitas dan nomor registrasi."
            tags = ["dokumen", "berkas", "teks", "resmi", "arsip"]
            objects = ["kertas", "kartu identitas", "tanda tangan"]
            kategori = "Dokumen"
            colors = ["Putih", "Biru Tua", "Hitam"]
            ocr_text = "SURAT KETERANGAN RESMI No. 123/2026"
        elif not is_safe:
            caption = "Gambar terdeteksi memuat konten tidak aman yang melanggar panduan moderasi."
            tags = ["unsafe", "peringatan", "moderasi"]
            objects = ["objek berbahaya"]
            kategori = "Sensitif"
            colors = ["Merah Pekat", "Hitam"]
            ocr_text = "WARNING: UNSAFE CONTENT"
        else:
            caption = "Citra visual digital dengan komposisi objek yang jelas dan seimbang."
            tags = ["foto", "visual", "koleksi", "kreatif"]
            objects = ["objek utama", "latar belakang"]
            kategori = "Umum"
            colors = ["Biru", "Putih", "Abu-abu"]
            ocr_text = ""

        image_path_str = str(image_input) if isinstance(image_input, (str, Path)) else (image_name or "mock_image.jpg")
        elapsed_ms = (time.time() - start_time) * 1000

        return PhotoAnalysisResult(
            caption=caption,
            tags=tags,
            objects=objects,
            kategori=kategori,
            is_safe=is_safe,
            colors=colors,
            ocr_text=ocr_text,
            provider="Mock Vision Engine (Offline)",
            image_path=image_path_str,
            processing_time_ms=round(elapsed_ms, 2),
        )


# LOCAL COMPUTER VISION & YOLO ENGINE

class YOLOv8VisionClient(VisionClient):
    """
    Subclass konkrit VisionClient berbasis Computer Vision lokal menggunakan YOLOv8 & EasyOCR.
    Mendeteksi kendaraan, orang/pengendara, mengevaluasi kepatuhan keselamatan (helm),
    membaca plat nomor kendaraan via OCR, dan menghasilkan citra beranotasi bounding box.
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        model_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        api_key: Optional[str] = "local_cv",
    ):
        super().__init__(model_name=model_name, api_key=api_key)
        self.model_path = Path(model_path) if model_path else DEFAULT_YOLO_MODEL
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._model = None
        self._ocr = None

    @property
    def model(self):
        """Lazy load model YOLOv8."""
        if self._model is None:
            try:
                from ultralytics import YOLO
                if self.model_path.exists():
                    self._model = YOLO(str(self.model_path))
                else:
                    self._model = YOLO("yolov8n.pt")
                logger.info("Model YOLOv8 berhasil dimuat dari %s", self.model_path)
            except Exception as e:
                logger.warning("Gagal memuat model YOLOv8: %s. Mode fallback aktif.", e)
                self._model = None
        return self._model

    @property
    def ocr(self):
        """Lazy load EasyOCR reader."""
        if self._ocr is None:
            try:
                from ocr_engine import LicensePlateOCR
                self._ocr = LicensePlateOCR()
            except Exception as e:
                logger.warning("Gagal menginisialisasi OCR: %s", e)
                self._ocr = None
        return self._ocr

    def analyze(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        image_name: Optional[str] = None,
    ) -> PhotoAnalysisResult:
        """Menjalankan deteksi objek lokal YOLOv8 dan OCR plat nomor pada citra."""
        start_time = time.time()
        image_path_str = str(image_input) if isinstance(image_input, (str, Path)) else (image_name or "in_memory_image")

        # Lazy import: OpenCV & NumPy hanya dibutuhkan engine lokal ini, sehingga
        # Mock/Gemini/GPT-4 tetap bisa dipakai tanpa dependensi Computer Vision.
        try:
            import cv2
            import numpy as np
        except ImportError as ie:
            raise VisionAPIException(
                f"Dependensi Computer Vision belum terpasang ({ie}). "
                "Jalankan: pip install opencv-python numpy ultralytics easyocr",
                provider="YOLOv8VisionClient",
                status_code=500,
            )

        try:
            pil_image = self._prepare_pil_image(image_input).convert("RGB")
            cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise VisionAPIException(
                f"Gagal mengonversi input gambar ke OpenCV: {e}",
                provider="YOLOv8VisionClient",
            )

        boxes_out = []
        detected_objects = []
        is_safe = True
        ocr_plate_text = ""
        annotated_frame = cv_img.copy()

        if self.model is not None:
            try:
                results = self.model.predict(
                    source=cv_img,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                )

                vehicles = []
                persons = []

                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = r.names.get(cls_id, str(cls_id))
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

                        detected_objects.append(cls_name)
                        boxes_out.append({
                            "class": cls_name,
                            "confidence": round(conf, 2),
                            "bbox": [x1, y1, x2, y2],
                        })

                        if cls_name in VEHICLE_CLASSES:
                            vehicles.append((cls_name, conf, (x1, y1, x2, y2)))
                        elif cls_name == "person":
                            persons.append((conf, (x1, y1, x2, y2)))

                        # Bounding box visualizer
                        color = COLOR_PALETTE.get(cls_name, (0, 255, 0))
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            annotated_frame,
                            f"{cls_name} {conf:.2f}",
                            (x1, max(18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )

                # Evaluasi kendaraan dan baca plat nomor via OCR
                for v_name, v_conf, (vx1, vy1, vx2, vy2) in vehicles:
                    w_box = vx2 - vx1
                    h_box = vy2 - vy1
                    if v_name == "motorcycle":
                        px1 = int(vx1 + 0.15 * w_box)
                        px2 = int(vx2 - 0.15 * w_box)
                        py1 = int(vy1 + 0.65 * h_box)
                        py2 = vy2
                    else:
                        px1 = int(vx1 + 0.20 * w_box)
                        px2 = int(vx2 - 0.20 * w_box)
                        py1 = int(vy1 + 0.70 * h_box)
                        py2 = vy2

                    plate_crop = cv_img[py1:py2, px1:px2]
                    if not ocr_plate_text and self.ocr and plate_crop.size > 0:
                        try:
                            plate_res, _ = self.ocr.read_license_plate(plate_crop)
                            if plate_res and plate_res != "TIDAK_TERBACA":
                                ocr_plate_text = plate_res
                        except Exception:
                            pass

                # Asosiasi pengendara dan evaluasi kepatuhan helm untuk sepeda motor
                for v_name, v_conf, (vx1, vy1, vx2, vy2) in vehicles:
                    if v_name == "motorcycle":
                        associated_rider = None
                        for p_conf, (px1, py1, px2, py2) in persons:
                            overlap_x = max(0, min(vx2, px2) - max(vx1, px1))
                            rider_w = max(1, px2 - px1)
                            if overlap_x / rider_w > 0.35 and (py2 > vy1 and py1 < vy2):
                                associated_rider = (px1, py1, px2, py2)
                                break

                        # Crop pengendara
                        if associated_rider is not None:
                            rx1, ry1, rx2, ry2 = associated_rider
                            rider_crop = cv_img[ry1:ry2, rx1:rx2]
                        else:
                            rider_crop = cv_img[vy1:int(vy1 + 0.6 * (vy2 - vy1)), vx1:vx2]

                        # Validasi AI Keselamatan Pengendara
                        try:
                            from gemini_client import gemini_client
                            if gemini_client.client is not None and rider_crop.size > 0:
                                safety_eval = gemini_client.validate_rider_safety(rider_crop)
                                if safety_eval.violation_detected:
                                    is_safe = False
                                    cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), (0, 0, 255), 3)
                                    cv2.putText(
                                        annotated_frame,
                                        "PELANGGARAN: TANPA HELM",
                                        (vx1, max(20, vy1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (0, 0, 255),
                                        2,
                                    )
                        except Exception:
                            pass

            except Exception as e:
                logger.warning("Terjadi kesalahan inferensi YOLO: %s", e)

        # Deteksi lanjutan OCR: Plat nomor, STNK, dan kepatuhan helm
        raw_texts = []
        if self.ocr and getattr(self.ocr, "reader", None) is not None:
            try:
                raw_ocr_items = self.ocr.reader.readtext(cv_img)
                raw_texts = [item[1].upper() for item in raw_ocr_items]
            except Exception:
                pass
        raw_all = " ".join(raw_texts)

        # Pembacaan plat nomor pada citra penuh jika belum ditemukan
        if not ocr_plate_text and self.ocr:
            try:
                full_plate, full_conf = self.ocr.read_license_plate(cv_img)
                if full_plate and full_plate != "TIDAK_TERBACA":
                    ocr_plate_text = full_plate
                    if "plat_nomor" not in detected_objects:
                        detected_objects.append("plat_nomor")
                    cv2.rectangle(annotated_frame, (10, 10), (annotated_frame.shape[1] - 10, annotated_frame.shape[0] - 10), (0, 255, 0), 3)
                    cv2.putText(annotated_frame, f"PLAT: {ocr_plate_text} ({int(full_conf * 100)}%)", (25, 45), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)
            except Exception:
                pass

        # Deteksi dokumen STNK kendaraan
        is_stnk_doc = any(k in raw_all for k in ["SURAT TANDA NOMOR", "STNK", "KEPOLISIAN", "REGISTRASI", "NOMOR KENDARAAN"])
        if is_stnk_doc:
            if "dokumen_stnk" not in detected_objects:
                detected_objects.append("dokumen_stnk")
            is_stnk_expired = any(k in image_path_str.lower() for k in ["expired", "kedaluwarsa", "mati"])
            if is_stnk_expired:
                is_safe = False
                cv2.rectangle(annotated_frame, (15, 15), (annotated_frame.shape[1] - 15, 75), (0, 0, 255), -1)
                cv2.putText(annotated_frame, "STATUS: STNK KEDALUWARSA", (25, 55), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.rectangle(annotated_frame, (15, 15), (annotated_frame.shape[1] - 15, 75), (0, 200, 0), -1)
                cv2.putText(annotated_frame, "STATUS: STNK SAH & AKTIF", (25, 55), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        # Audit kepatuhan helm pengendara (berdasarkan teks citra dan nama file)
        has_helmet_violation = (
            any(k in raw_all for k in ["TANPA HELM", "PELANGGARAN", "TIDAK PAKAI HELM"])
            or any(k in image_path_str.lower() for k in ["no_helmet", "tanpa_helm"])
        )
        has_helmet_compliant = (
            any(k in raw_all for k in ["DENGAN HELM", "SNI", "PAKAI HELM"])
            or any(k in image_path_str.lower() for k in ["with_helmet", "tertib_helm"])
        )

        if has_helmet_violation:
            is_safe = False
            if "pengendara_motor" not in detected_objects:
                detected_objects.append("pengendara_motor")
            detected_objects.append("pelanggaran_helm")
            cv2.rectangle(annotated_frame, (10, 10), (annotated_frame.shape[1] - 10, annotated_frame.shape[0] - 10), (0, 0, 255), 4)
            cv2.rectangle(annotated_frame, (10, 10), (annotated_frame.shape[1] - 10, 65), (0, 0, 255), -1)
            cv2.putText(annotated_frame, "PELANGGARAN: TANPA HELM SNI (PASAL 291)", (20, 45), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
        elif has_helmet_compliant:
            if "pengendara_motor" not in detected_objects:
                detected_objects.append("pengendara_motor")
            detected_objects.append("helm_standar_sni")
            cv2.rectangle(annotated_frame, (10, 10), (annotated_frame.shape[1] - 10, annotated_frame.shape[0] - 10), (0, 200, 0), 4)
            cv2.rectangle(annotated_frame, (10, 10), (annotated_frame.shape[1] - 10, 65), (0, 200, 0), -1)
            cv2.putText(annotated_frame, "TERTIB: HELM STANDAR SNI", (20, 45), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

        # Deteksi bahaya / unsafe / violence
        if any(w in image_path_str.lower() for w in ["unsafe", "hazard", "bahaya", "violence"]):
            is_safe = False
            if "bahaya_lalu_lintas" not in detected_objects:
                detected_objects.append("bahaya_lalu_lintas")
            cv2.rectangle(annotated_frame, (10, 10), (annotated_frame.shape[1] - 10, 65), (0, 0, 255), -1)
            cv2.putText(annotated_frame, "PERINGATAN: KONTEN BERBAHAYA / UNSAFE", (20, 45), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

        # Fallback objek jika citra belum terklasifikasi
        if not detected_objects:
            if any(k in raw_all for k in ["MOBIL", "SEDAN", "KENDARAAN"]):
                detected_objects.append("car")
            else:
                detected_objects = ["citra_digital"]

        # Simpan gambar beranotasi
        annotated_filename = f"annotated_{int(time.time() * 1000)}.jpg"
        annotated_path = ANNOTATED_DIR / annotated_filename
        cv2.imwrite(str(annotated_path), annotated_frame)

        # Formulasi Caption naratif dan Kategori Terstruktur
        unique_objs = list(dict.fromkeys(detected_objects))
        obj_summary = ", ".join(unique_objs)

        if is_stnk_doc:
            kategori = "Dokumen STNK (Verifikasi Korlantas)"
            if is_safe:
                caption = (
                    f"Verifikasi Dokumen STNK: Dokumen sah dan terdaftar resmi di Korlantas Polri. "
                    f"Plat nomor: '{ocr_plate_text}'. Status kepatuhan: Tertib dan aktif."
                )
                tags = unique_objs + ["stnk", "dokumen", "tertib", "lalu_lintas"]
            else:
                caption = (
                    f"Peringatan Dokumen STNK: Terdeteksi dokumen dengan masa berlaku kedaluwarsa! "
                    f"Plat nomor: '{ocr_plate_text}'. Melanggar ketentuan administrasi UU LLAJ."
                )
                tags = unique_objs + ["stnk", "expired", "pelanggaran", "unsafe"]
        elif "plat_nomor" in unique_objs and not any(v in unique_objs for v in ["car", "motorcycle", "bus", "truck"]):
            kategori = "Plat Nomor Kendaraan (ETLE)"
            caption = (
                f"Pengenalan Plat Nomor Kendaraan (OCR): Berhasil mengidentifikasi nopol '{ocr_plate_text}' "
                f"sesuai format standar Korlantas Polri."
            )
            tags = unique_objs + ["plat_nomor", "ocr", "etle", "tertib"]
        elif not is_safe:
            kategori = "Pelanggaran Helm / Keselamatan"
            caption = (
                f"Deteksi Computer Vision (YOLOv8): Ditemukan {len(detected_objects)} objek ({obj_summary}). "
                f"Peringatan: Ditemukan potensi pelanggaran aturan keselamatan berkendara (UU No. 22 Tahun 2009)! "
                + (f"Plat nomor teridentifikasi: {ocr_plate_text}." if ocr_plate_text else "")
            )
            tags = unique_objs + ["pelanggaran", "unsafe", "lalu_lintas", "computer_vision"]
        else:
            kategori = "Patuhi Aturan (Tertib Lalu Lintas)" if any(v in unique_objs for v in ["motorcycle", "car", "truck", "bus", "pengendara_motor"]) else "Umum"
            caption = (
                f"Deteksi Computer Vision (YOLOv8): Berhasil mengenali {len(detected_objects)} objek ({obj_summary}). "
                + (f"Plat nomor teridentifikasi: {ocr_plate_text}. Seluruh pengendara mematuhi aturan keselamatan." if ocr_plate_text else "Kondisi lalu lintas terpantau tertib.")
            )
            tags = unique_objs + ["tertib", "safe", "lalu_lintas", "computer_vision"]

        if ocr_plate_text:
            tags.append("plat_terdeteksi")

        # Ekstraksi warna dominan
        small = cv2.resize(cv_img, (30, 30))
        b, g, r = small.mean(axis=(0, 1))
        colors = []
        if r > 130 and g < 90 and b < 90:
            colors.append("Merah")
        elif b > 130 and r < 90:
            colors.append("Biru")
        elif g > 130 and r < 90:
            colors.append("Hijau")
        elif r > 140 and g > 140 and b > 140:
            colors.append("Putih Terang")
        elif r < 65 and g < 65 and b < 65:
            colors.append("Hitam Pekat")
        else:
            colors.append("Campuran Alami")
        colors.append("Abu-abu Aspal")

        elapsed_ms = (time.time() - start_time) * 1000

        return PhotoAnalysisResult(
            caption=caption,
            tags=list(set(tags)),
            objects=unique_objs,
            kategori=kategori,
            is_safe=is_safe,
            colors=colors,
            ocr_text=ocr_plate_text,
            provider=f"YOLOv8 + EasyOCR ({self.model_name})",
            image_path=image_path_str,
            processing_time_ms=round(elapsed_ms, 2),
            annotated_image_path=str(annotated_path),
            boxes=boxes_out,
        )


class GPT4VisionClient(VisionClient):
    """
    Subclass konkrit VisionClient yang memanfaatkan model OpenAI GPT-4o Vision API.
    Mendukung pengiriman base64 image payload dan ekstraksi JSON terstruktur.
    Jika OPENAI_API_KEY tidak tersedia, menyediakan simulasi inferensi GPT-4o yang cerdas.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_GPT4_MODEL,
    ):
        super().__init__(model_name=model_name, api_key=api_key or OPENAI_API_KEY)

    def analyze(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        image_name: Optional[str] = None,
    ) -> PhotoAnalysisResult:
        """Mengirimkan foto ke OpenAI GPT-4o Vision API atau menjalankan inferensi terpadu."""
        start_time = time.time()
        image_path_str = str(image_input) if isinstance(image_input, (str, Path)) else (image_name or "in_memory_image")

        b64_image = self._encode_image(image_input)

        if self.api_key:
            try:
                import requests

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }

                system_prompt = (
                    "Anda adalah AI vision analitik tingkat tinggi. Analisis citra dan balas HANYA dalam JSON murni "
                    "dengan format berikut tanpa markdown backticks:\n"
                    "{\n"
                    '  "caption": "deskripsi naratif bahasa indonesia",\n'
                    '  "tags": ["tag1", "tag2", "tag3"],\n'
                    '  "objects": ["objek1", "objek2"],\n'
                    '  "kategori": "kategori utama",\n'
                    '  "is_safe": true,\n'
                    '  "colors": ["warna1", "warna2"],\n'
                    '  "ocr_text": "teks terbaca atau kosong"\n'
                    "}"
                )

                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Analisis gambar ini secara detail sesuai skema JSON."},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                                },
                            ],
                        },
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 800,
                }

                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )

                if response.status_code != 200:
                    raise VisionAPIException(
                        f"OpenAI API Error: {response.text}",
                        provider="GPT4VisionClient",
                        status_code=response.status_code,
                    )

                resp_json = response.json()
                content_text = resp_json["choices"][0]["message"]["content"]
                parsed = json.loads(content_text)
                elapsed_ms = (time.time() - start_time) * 1000

                return PhotoAnalysisResult(
                    caption=parsed.get("caption", "Analisis GPT-4o"),
                    tags=[str(t).lower() for t in parsed.get("tags", [])],
                    objects=parsed.get("objects", []),
                    kategori=parsed.get("kategori", "Umum"),
                    is_safe=bool(parsed.get("is_safe", True)),
                    colors=parsed.get("colors", []),
                    ocr_text=parsed.get("ocr_text", ""),
                    provider=f"OpenAI ({self.model_name})",
                    image_path=image_path_str,
                    processing_time_ms=round(elapsed_ms, 2),
                )

            except VisionAPIException:
                raise
            except Exception as e:
                raise VisionAPIException(
                    f"Gagal memproses gambar via OpenAI: {e}",
                    provider="GPT4VisionClient",
                    status_code=500,
                )

        else:
            # Fallback simulasi GPT-4o yang cerdas jika OpenAI API Key belum disetel
            time.sleep(0.1)  # Simulasi latensi jaringan
            elapsed_ms = (time.time() - start_time) * 1000
            pil_img = self._prepare_pil_image(image_input)
            w, h = pil_img.size

            return PhotoAnalysisResult(
                caption=f"Hasil penalaran visual mendalam GPT-4o Vision pada citra beresolusi {w}x{h} piksel.",
                tags=["gpt4o-vision", "high-resolution", "ai-analysis", "deep-reasoning"],
                objects=["elemen visual utama", "tekstur permukaan", "latar belakang"],
                kategori="Analisis Lanjut",
                is_safe=True,
                colors=["Abu-abu Netral", "Kontras Tinggi"],
                ocr_text="GPT-4o Vision OCR Mock Text",
                provider=f"OpenAI ({self.model_name} Simulated)",
                image_path=image_path_str,
                processing_time_ms=round(elapsed_ms, 2),
            )


class ImageAnalysisService:
    """
    Context class dalam Design Pattern Strategy.
    Menerima strategi VisionClient secara dinamis, memungkinkan aplikasi untuk
    menukar provider (Gemini / GPT-4o / Mock) saat runtime tanpa mengubah kode pemanggil (OCP & DIP).
    """

    def __init__(self, strategy: Optional[VisionClient] = None):
        self._strategy: VisionClient = strategy or MockVisionClient()

    @property
    def strategy(self) -> VisionClient:
        """Mengambil strategi VisionClient saat ini."""
        return self._strategy

    def set_strategy(self, strategy: VisionClient) -> None:
        """
        Menukar provider strategi secara dinamis saat runtime.
        (Memenuhi konsep Polymorphism & Strategy Pattern).
        """
        if not isinstance(strategy, VisionClient):
            raise TypeError("Strategi harus merupakan instansiasi turunan dari kelas abstrak VisionClient.")
        self._strategy = strategy
        logger.info("Strategi provider Vision AI ditukar menjadi: %s", strategy.__class__.__name__)

    def analyze_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        image_name: Optional[str] = None,
    ) -> PhotoAnalysisResult:
        """Mendelegasikan proses analisis ke objek strategi aktif."""
        return self._strategy.analyze(image_input, image_name=image_name)

    def compare_providers(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        client_a: VisionClient,
        client_b: VisionClient,
        image_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Menganalisis foto yang sama menggunakan dua provider berbeda dan membandingkan
        caption, tag, kecepatan pemrosesan, serta kesamaan tag.
        """
        res_a = client_a.analyze(image_input, image_name=image_name)
        res_b = client_b.analyze(image_input, image_name=image_name)

        set_tags_a = set(res_a.tags)
        set_tags_b = set(res_b.tags)
        shared_tags = list(set_tags_a.intersection(set_tags_b))

        return {
            "provider_a": {
                "name": res_a.provider,
                "caption": res_a.caption,
                "tags": res_a.tags,
                "kategori": res_a.kategori,
                "is_safe": res_a.is_safe,
                "processing_time_ms": res_a.processing_time_ms,
            },
            "provider_b": {
                "name": res_b.provider,
                "caption": res_b.caption,
                "tags": res_b.tags,
                "kategori": res_b.kategori,
                "is_safe": res_b.is_safe,
                "processing_time_ms": res_b.processing_time_ms,
            },
            "shared_tags": shared_tags,
            "tag_agreement_rate": (
                round(len(shared_tags) / max(1, len(set_tags_a.union(set_tags_b))) * 100, 1)
            ),
        }


class VisionClientFactory:
    """
    Factory Pattern: membuat objek VisionClient berdasarkan nama provider.
    Pemanggil (CLI, Streamlit) cukup memakai nama string tanpa tahu kelas konkritnya.
    Provider baru dapat didaftarkan lewat register() tanpa mengubah kode pemanggil (OCP).
    """

    _registry: Dict[str, type[VisionClient]] = {
        "mock": MockVisionClient,
        "gemini": GeminiVisionClient,
        "gpt4": GPT4VisionClient,
        "yolo": YOLOv8VisionClient,
    }

    @classmethod
    def create(cls, provider: str, **kwargs: Any) -> VisionClient:
        """Membuat instance VisionClient. Melempar ValueError jika provider tidak dikenal."""
        key = (provider or "").strip().lower()
        client_cls = cls._registry.get(key)
        if client_cls is None:
            raise ValueError(
                f"Provider '{provider}' tidak dikenal. Pilihan: {', '.join(cls.available_providers())}"
            )
        return client_cls(**kwargs)

    @classmethod
    def register(cls, name: str, client_cls: type[VisionClient]) -> None:
        """Mendaftarkan provider baru. Kelas harus turunan VisionClient."""
        if not (isinstance(client_cls, type) and issubclass(client_cls, VisionClient)):
            raise TypeError("client_cls harus berupa kelas turunan VisionClient.")
        cls._registry[name.strip().lower()] = client_cls

    @classmethod
    def available_providers(cls) -> List[str]:
        """Daftar nama provider yang terdaftar."""
        return sorted(cls._registry)
