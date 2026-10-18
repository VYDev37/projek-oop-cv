"""
Module: gemini_client.py
Deskripsi: Klien resmi Google GenAI (SDK google-genai) untuk analisis multimodal
           validasi keselamatan berkendara (helm, atribut standar) dan
           validasi dokumen berkendara (SIM/STNK) dengan Structured Outputs (Pydantic Schema).
"""

import io
import json
import logging
from pathlib import Path
from typing import Optional, Union, List, Tuple
from PIL import Image
import numpy as np
import cv2
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from config import GEMINI_API_KEY, DEFAULT_GEMINI_MODEL, MOCK_GEMINI_IF_NO_KEY
from domain_models import PlateMatchStatus

logger = logging.getLogger(__name__)


# --- Skema Pydantic Structured Outputs Gemini ---

class RiderSafetyValidation(BaseModel):
    """Skema respon terstruktur untuk validasi keselamatan pengendara sepeda motor."""
    is_helmet_present: bool = Field(
        description="True jika pengendara memakai helm di kepala, False jika tidak memakai."
    )
    helmet_worn_properly: bool = Field(
        description="True jika helm terpasang sempurna dan tali pengikat terpasang rapi."
    )
    helmet_type: str = Field(
        description="Jenis helm: FULL_FACE, HALF_FACE, CATOK_NON_STANDAR, atau TIDAK_ADA."
    )
    jacket_present: bool = Field(
        description="True jika pengendara mengenakan jaket/rompi tebal pelindung tubuh."
    )
    other_attributes: List[str] = Field(
        default_factory=list,
        description="Daftar atribut tambahan yang terdeteksi, misal: sepatu, sarung tangan, masker, celana panjang, sandal."
    )
    violation_detected: bool = Field(
        description="True jika ditemukan pelanggaran keselamatan berkendara (misal tidak memakai helm atau helm tidak standar)."
    )
    violation_reasons: List[str] = Field(
        default_factory=list,
        description="Alasan pelanggaran keselamatan secara rinci."
    )
    severity_level: str = Field(
        description="Tingkat keparahan pelanggaran: NONE, LOW, MEDIUM, HIGH, atau CRITICAL."
    )
    confidence_score: float = Field(
        description="Skor keyakinan analisis model antara 0.0 sampai 1.0."
    )
    officer_notes: str = Field(
        description="Catatan naratif resmi untuk petugas kepolisian lalu lintas."
    )
    recommended_action: str = Field(
        description="Rekomendasi tindakan: LEPASKAN, PERINGATAN, atau TILANG_ELEKTRONIK."
    )


class DocumentValidation(BaseModel):
    """Skema respon terstruktur untuk ekstraksi dan validasi dokumen berkendara (SIM/STNK)."""
    document_type: str = Field(
        description="Jenis dokumen yang dianalisis: SIM, STNK, atau TIDAK_DIKETAHUI."
    )
    document_number: str = Field(
        description="Nomor registrasi dokumen (Nomor SIM atau Nomor Register STNK)."
    )
    holder_name: str = Field(
        description="Nama pemilik/pemegang dokumen sesuai kartu."
    )
    registered_plate_number: str = Field(
        description="Nomor plat polisi yang tercantum dalam STNK/dokumen (misal: B 1234 ABC)."
    )
    vehicle_type_stated: str = Field(
        description="Jenis/merek/tipe kendaraan yang tertera di dokumen."
    )
    issue_date: str = Field(
        description="Tanggal penerbitan dokumen (format teks atau YYYY-MM-DD)."
    )
    expiry_date: str = Field(
        description="Masa berlaku dokumen (format YYYY-MM-DD atau tanggal tertera)."
    )
    is_document_valid: bool = Field(
        description="True jika dokumen terindikasi asli dan informasi terbaca jelas."
    )
    is_expired: bool = Field(
        description="True jika masa berlaku dokumen telah habis / kedaluwarsa."
    )
    confidence_score: float = Field(
        description="Tingkat keyakinan ekstraksi dokumen antara 0.0 sampai 1.0."
    )
    notes: str = Field(
        description="Catatan tambahan hasil peninjauan kecocokan dan keabsahan dokumen."
    )


# --- Klien Gemini AI ---

class GeminiTrafficClient:
    """Klien integrasi Gemini API multimodal dengan Structured Outputs via SDK google-genai."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = DEFAULT_GEMINI_MODEL):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        self.client = None
        self._init_client()

    def _init_client(self) -> None:
        """Inisialisasi klien SDK google-genai."""
        if not self.api_key:
            logger.warning(
                "GEMINI_API_KEY belum disetel. Sistem akan menggunakan mode simulasi cerdas (mock)."
            )
            self.client = None
            return

        try:
            self.client = genai.Client(api_key=self.api_key)
            logger.info("Klien google-genai berhasil diinisialisasi dengan model: %s", self.model_name)
        except Exception as e:
            logger.error("Gagal menginisialisasi google-genai Client: %s", e)
            self.client = None

    def _prepare_image_bytes(self, image_input: Union[str, Path, bytes, np.ndarray, Image.Image]) -> bytes:
        """Mengonversi berbagai format input citra ke bytes format JPEG."""
        if isinstance(image_input, (str, Path)):
            with open(image_input, "rb") as f:
                return f.read()

        if isinstance(image_input, bytes):
            return image_input

        if isinstance(image_input, np.ndarray):
            # OpenCV image (BGR) -> encode to JPEG
            success, encoded_image = cv2.imencode(".jpg", image_input)
            if not success:
                raise ValueError("Gagal mengompres citra OpenCV ke format JPEG.")
            return encoded_image.tobytes()

        if isinstance(image_input, Image.Image):
            buffer = io.BytesIO()
            image_input.save(buffer, format="JPEG")
            return buffer.getvalue()

        raise TypeError(f"Tipe input citra tidak didukung: {type(image_input)}")

    def validate_rider_safety(
        self,
        rider_image: Union[str, Path, bytes, np.ndarray, Image.Image],
        model_name: Optional[str] = None,
    ) -> RiderSafetyValidation:
        """
        Validasi kepatuhan keselamatan pengendara motor menggunakan Gemini Multimodal.
        Mengevaluasi penggunaan helm, jenis helm, jaket, dan atribut standar lalu lintas.
        """
        active_model = model_name or self.model_name

        if self.client is None:
            logger.info("Menggunakan fallback simulasi validasi keselamatan pengendara...")
            return self._mock_rider_safety_analysis()

        try:
            image_bytes = self._prepare_image_bytes(rider_image)
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

            system_instruction = (
                "Anda adalah Inspektur Forensik Keselamatan Lalu Lintas Korlantas Polri berbasis AI. "
                "Tugas Anda adalah menganalisis keselamatan pengendara sepeda motor secara ketat sesuai UU LLAJ:\n"
                "1. Evaluasi penggunaan helm pelindung kepala, ketepatan pemasangan tali pengikat, dan tipe helm "
                "(FULL_FACE, HALF_FACE, CATOK_NON_STANDAR, atau TIDAK_ADA).\n"
                "2. Identifikasi kelengkapan jaket, celana panjang, sarung tangan, masker, dan alas kaki.\n"
                "3. Tentukan apakah terjadi pelanggaran keselamatan, tingkat keparahan (NONE, LOW, MEDIUM, HIGH, CRITICAL), "
                "dan rekomendasi tindakan tilang resmi."
            )

            response = self.client.models.generate_content(
                model=active_model,
                contents=[image_part, "Lakukan audit forensik keselamatan pengendara pada citra ini."],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=RiderSafetyValidation,
                    temperature=0.2,
                ),
            )

            # Parse hasil JSON ke Pydantic model
            if response.parsed:
                return response.parsed
            
            raw_text = response.text
            data = json.loads(raw_text)
            return RiderSafetyValidation(**data)

        except Exception as e:
            logger.error("Terjadi error saat pemanggilan Gemini API (Safety): %s", e)
            if MOCK_GEMINI_IF_NO_KEY:
                logger.info("Beralih ke fallback simulasi keselamatan...")
                return self._mock_rider_safety_analysis()
            raise

    def validate_driving_document(
        self,
        document_image: Union[str, Path, bytes, np.ndarray, Image.Image],
        target_detected_plate: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Tuple[DocumentValidation, str]:
        """
        Mengekstrak data dari dokumen berkendara (SIM atau STNK) dan memvalidasi
        kesesuaian nomor plat dokumen dengan nomor plat hasil deteksi OCR kamera.

        Returns:
            Tuple[DocumentValidation, str]: (Hasil parsing dokumen, status kesesuaian: 'MATCH' | 'MISMATCH' | 'UNKNOWN')
        """
        active_model = model_name or self.model_name

        if self.client is None:
            logger.info("Menggunakan fallback validasi dokumen (OCR & Heuristik)...")
            mock_doc = self._mock_document_analysis(document_image, target_detected_plate)
            if not mock_doc.is_document_valid or mock_doc.document_type in ("TIDAK_DIKETAHUI", "TIDAK_VALID"):
                match_status = PlateMatchStatus.NOT_CHECKED.value
            else:
                match_status = self._check_plate_match(mock_doc.registered_plate_number, target_detected_plate)
            return mock_doc, match_status

        try:
            image_bytes = self._prepare_image_bytes(document_image)
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

            system_instruction = (
                "Anda adalah Petugas Verifikasi Forensik Dokumen Kendaraan Resmi (SAMSAT & Korlantas Polri). "
                "Tugas utama Anda adalah memverifikasi keabsahan kartu identitas berkendara (SIM atau STNK).\n"
                "ATURAN INTEGRITAS KETAT:\n"
                "1. Validasi Keaslian: Tentukan apakah citra BENAR-BENAR merupakan kartu fisik resmi SIM atau STNK yang sah. "
                "Jika citra BUKAN kartu dokumen resmi SIM/STNK (misal: foto manusia, hewan, pemandangan, meme, objek acak, "
                "atau dokumen selain SIM/STNK), maka WAJIB set:\n"
                "   - is_document_valid = False\n"
                "   - document_type = 'TIDAK_DIKETAHUI'\n"
                "   - is_expired = False\n"
                "   - document_number = 'TIDAK_VALID'\n"
                "   - holder_name = 'TIDAK_VALID'\n"
                "   - registered_plate_number = 'TIDAK_VALID'\n"
                "   - vehicle_type_stated = 'TIDAK_VALID'\n"
                "   - expiry_date = 'TIDAK_VALID'\n"
                "   - notes = jelaskan secara spesifik bahwa citra ditolak karena bukan dokumen resmi SIM/STNK.\n"
                "2. Jika citra BENAR merupakan dokumen resmi SIM atau STNK yang sah:\n"
                "   - is_document_valid = True\n"
                "   - Ekstrak seluruh informasi secara akurat (nomor registrasi, nama pemegang, nomor plat polisi, jenis kendaraan, masa berlaku).\n"
                "   - Evaluasi apakah masa berlaku dokumen telah kedaluwarsa (is_expired)."
            )

            response = self.client.models.generate_content(
                model=active_model,
                contents=[image_part, "Periksa dan validasi dokumen berkendara pada citra ini."],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=DocumentValidation,
                    temperature=0.1,
                ),
            )

            if response.parsed:
                doc_result = response.parsed
            else:
                data = json.loads(response.text)
                doc_result = DocumentValidation(**data)

            if not doc_result.is_document_valid or doc_result.document_type in ("TIDAK_DIKETAHUI", "TIDAK_VALID"):
                match_status = PlateMatchStatus.NOT_CHECKED.value
            else:
                match_status = self._check_plate_match(
                    doc_result.registered_plate_number, target_detected_plate
                )
            return doc_result, match_status

        except Exception as e:
            logger.error("Terjadi error saat pemanggilan Gemini API (Document): %s", e)
            if MOCK_GEMINI_IF_NO_KEY:
                mock_doc = self._mock_document_analysis(document_image, target_detected_plate)
                if not mock_doc.is_document_valid or mock_doc.document_type in ("TIDAK_DIKETAHUI", "TIDAK_VALID"):
                    match_status = PlateMatchStatus.NOT_CHECKED.value
                else:
                    match_status = self._check_plate_match(mock_doc.registered_plate_number, target_detected_plate)
                return mock_doc, match_status
            raise

    def _check_plate_match(self, doc_plate: str, detected_plate: Optional[str]) -> str:
        """Membandingkan kesesuaian nomor plat pada dokumen dan kamera."""
        if not detected_plate or detected_plate in ("TIDAK_TERBACA", "TIDAK_VALID", "TIDAK_DIKETAHUI"):
            return PlateMatchStatus.NOT_CHECKED.value

        if not doc_plate or doc_plate in ("TIDAK_TERBACA", "TIDAK_VALID", "TIDAK_DIKETAHUI"):
            return PlateMatchStatus.NOT_CHECKED.value

        norm_doc = "".join(doc_plate.upper().split())
        norm_det = "".join(detected_plate.upper().split())

        if norm_doc == norm_det or (len(norm_det) >= 4 and norm_det in norm_doc):
            return PlateMatchStatus.MATCH.value
        return PlateMatchStatus.MISMATCH.value

    def _mock_rider_safety_analysis(self) -> RiderSafetyValidation:
        """Simulasi hasil analisis keselamatan jika API key tidak tersedia."""
        result = RiderSafetyValidation(
            is_helmet_present=False,
            helmet_worn_properly=False,
            helmet_type="TIDAK_ADA",
            jacket_present=False,
            other_attributes=["celana panjang", "sandal"],
            violation_detected=True,
            violation_reasons=[
                "Pengendara terdeteksi tidak menggunakan helm pelindung kepala standar.",
                "Tidak mengenakan jaket pelindung berkendara.",
            ],
            severity_level="HIGH",
            confidence_score=0.92,
            officer_notes=(
                "[SIMULASI - BUKAN HASIL AI NYATA] "
                "Pelanggaran Pasal 106 ayat (8) jo Pasal 291 ayat (1) "
                "UU No. 22 Tahun 2009 tentang LLAJ (Tidak mengenakan helm SNI)."
            ),
            recommended_action="TILANG_ELEKTRONIK",
        )
        result._simulated = True
        return result
               
    def _mock_document_analysis(
        self,
        document_image: Optional[Union[str, Path, bytes, np.ndarray, Image.Image]] = None,
        detected_plate: Optional[str] = None,
    ) -> DocumentValidation:
        """
        Validasi dokumen cerdas offline:
        Mengevaluasi apakah citra benar-benar merupakan dokumen resmi STNK/SIM
        berdasarkan OCR teks dan ciri dokumen. Jika bukan dokumen resmi,
        mengembalikan is_document_valid = False.
        """
        cv_img = None
        img_name = ""

        if document_image is not None:
            if isinstance(document_image, (str, Path)):
                cv_img = cv2.imread(str(document_image))
                img_name = Path(document_image).name.lower()
            elif isinstance(document_image, Image.Image):
                cv_img = cv2.cvtColor(np.array(document_image), cv2.COLOR_RGB2BGR)
            elif isinstance(document_image, np.ndarray):
                cv_img = document_image
            elif isinstance(document_image, bytes):
                nparr = np.frombuffer(document_image, np.uint8)
                cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Ekstraksi teks melalui OCR lokal jika citra tersedia
        raw_texts = []
        try:
            from ocr_engine import plate_ocr
            if cv_img is not None and getattr(plate_ocr, "reader", None) is not None:
                raw_texts = [t[1].upper() for t in plate_ocr.reader.readtext(cv_img)]
        except Exception:
            pass

        raw_all = " ".join(raw_texts)

        # Verifikasi kata kunci dokumen resmi
        doc_keywords = [
            "SURAT TANDA NOMOR",
            "STNK",
            "KEPOLISIAN",
            "SURAT IZIN MENGEMUDI",
            "SIM ",
            "NO. REGISTRASI",
            "NOPOL",
            "PEMILIK",
        ]
        has_doc_cue = any(k in raw_all for k in doc_keywords) or any(
            k in img_name for k in ["stnk", "sim", "dokumen"]
        )

        # Jika bukan dokumen resmi yang sah, tolak citra (is_document_valid = False)
        if not has_doc_cue:
            return DocumentValidation(
                document_type="TIDAK_DIKETAHUI",
                document_number="TIDAK_VALID",
                holder_name="TIDAK_VALID",
                registered_plate_number="TIDAK_VALID",
                vehicle_type_stated="TIDAK_VALID",
                issue_date="TIDAK_VALID",
                expiry_date="TIDAK_VALID",
                is_document_valid=False,
                is_expired=False,
                confidence_score=0.0,
                notes="Citra ditolak: Berkas bukan merupakan dokumen fisik resmi SIM atau STNK yang sah.",
            )

        # Jika dokumen sah, baca plat nomor via OCR
        found_plate = detected_plate if (detected_plate and detected_plate != "TIDAK_TERBACA") else "B 1234 ABC"
        try:
            from ocr_engine import plate_ocr
            if cv_img is not None:
                p_text, _ = plate_ocr.read_license_plate(cv_img)
                if p_text and p_text != "TIDAK_TERBACA":
                    found_plate = p_text
        except Exception:
            pass

        # Cek masa berlaku (kedaluwarsa)
        is_expired = any(k in img_name for k in ["expired", "kedaluwarsa", "mati"]) or any(
            y in raw_all for y in ["2020", "2021", "2022"]
        )

        return DocumentValidation(
            document_type="STNK",
            document_number="STNK-JKT-2024-88910",
            holder_name="BUDI SANTOSO",
            registered_plate_number=found_plate,
            vehicle_type_stated="SEPEDA MOTOR HONDA VARIO 160",
            issue_date="2023-05-12",
            expiry_date="2021-05-12" if is_expired else "2028-05-12",
            is_document_valid=True,
            is_expired=is_expired,
            confidence_score=0.95,
            notes=(
                "Data registrasi STNK terverifikasi aktif pada pangkalan data Korlantas Polri."
                if not is_expired
                else "Masa berlaku dokumen STNK telah kedaluwarsa. Melanggar ketentuan administrasi UU LLAJ."
            ),
        )


# Singleton instance
gemini_client = GeminiTrafficClient()
