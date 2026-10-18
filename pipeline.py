"""
Module: pipeline.py
Deskripsi: Pipeline orkestrasi terpadu yang menghubungkan Modul Deteksi (YOLO + ByteTrack),
           Modul OCR (EasyOCR), Modul Gemini AI (google-genai), dan Manajemen Database (SQLite).
           Dilengkapi pemrosesan asinkronus (background thread) agar pemutaran video tidak freeze.
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import cv2
import numpy as np

from config import COUNTING_LINE_Y_RATIO
from database import db_manager, DatabaseManager
from detector import TrafficDetector, ViolationCandidate
from ocr_engine import plate_ocr, LicensePlateOCR
from gemini_client import gemini_client, GeminiTrafficClient, RiderSafetyValidation, DocumentValidation
from domain_models import VehicleType, ViolationType, HelmetStatus, EnforcementStatus, PlateMatchStatus
from vision_client import PhotoAnalysisResult
from photo_collection import PhotoCollection

logger = logging.getLogger(__name__)


@dataclass
class ViolationAlert:
    """Representasi peringatan pelanggaran yang telah divalidasi."""
    id: int
    timestamp: str
    track_id: int
    vehicle_type: str
    license_plate: str
    violation_type: str
    severity: str
    helmet_status: str
    recommended_action: str
    officer_notes: str
    rider_image_path: Optional[str]
    plate_image_path: Optional[str]


class TrafficMonitoringPipeline:
    """Pipeline pemrosesan terintegrasi untuk pemantauan lalu lintas cerdas."""

    def __init__(
        self,
        detector: Optional[TrafficDetector] = None,
        ocr: Optional[LicensePlateOCR] = None,
        gemini: Optional[GeminiTrafficClient] = None,
        database: Optional[DatabaseManager] = None,
        photo_collection: Optional[PhotoCollection] = None,
        async_ai: bool = True,
    ):
        self.detector = detector or TrafficDetector()
        self.ocr = ocr or plate_ocr
        self.gemini = gemini or gemini_client
        self.db = database or db_manager
        self.photo_collection = photo_collection
        self.async_ai = async_ai

        # ThreadPool untuk eksekusi OCR & Gemini API tanpa memblokir FPS video
        self.executor = ThreadPoolExecutor(max_workers=3)
        self._active_tasks: Set[Any] = set()
        self.recent_alerts: List[ViolationAlert] = []

    def _analyze_candidate_worker(self, candidate: ViolationCandidate) -> None:
        """
        Pekerja latar belakang (Background Worker):
        1. Menyimpan citra bukti secara asinkron
        2. Menjalankan EasyOCR pada citra in-memory plat nomor
        3. Menjalankan Gemini AI pada citra in-memory pengendara
        4. Menyimpan hasil lengkap ke database SQLite
        5. Menambahkan alert ke antrean notifikasi real-time
        """
        try:
            # Simpan bukti citra secara asinkron (di background worker, bukan di thread video)
            if candidate.rider_crop is not None and candidate.rider_crop_path:
                cv2.imwrite(str(candidate.rider_crop_path), candidate.rider_crop)
            if candidate.plate_crop is not None and candidate.plate_crop_path:
                cv2.imwrite(str(candidate.plate_crop_path), candidate.plate_crop)

            plate_text = "TIDAK_TERBACA"
            plate_conf = 0.0

            # Pembacaan OCR Plat Nomor (Prioritaskan citra in-memory)
            plate_img = candidate.plate_crop
            if plate_img is None and candidate.plate_crop_path and candidate.plate_crop_path.exists():
                plate_img = cv2.imread(str(candidate.plate_crop_path))

            if plate_img is not None:
                plate_text, plate_conf = self.ocr.read_license_plate(plate_img)

            # Validasi Keselamatan Gemini AI (Prioritaskan citra in-memory)
            safety_result: Optional[RiderSafetyValidation] = None
            rider_input = candidate.rider_crop if candidate.rider_crop is not None else candidate.rider_crop_path
            if rider_input is not None:
                safety_result = self.gemini.validate_rider_safety(rider_input)
            else:
                safety_result = self.gemini._mock_rider_safety_analysis()

            # Penentuan Status Pelanggaran
            is_simulated = getattr(safety_result, "_simulated", False)
            violation_type = ViolationType.COMPLIANT.value
            status_tilang = EnforcementStatus.RELEASED.value
            helmet_status = HelmetStatus.WORN.value

            if safety_result.violation_detected:
                violation_type = ViolationType.NO_HELMET.value if not safety_result.is_helmet_present else ViolationType.NON_STANDARD.value
                status_tilang = (
                    EnforcementStatus.PENDING_REVIEW.value
                    if is_simulated
                    else EnforcementStatus.CONFIRMED_TICKET.value
                )
                helmet_status = HelmetStatus.NOT_WORN.value if not safety_result.is_helmet_present else HelmetStatus.NON_STANDARD.value
                       
            # Simpan ke Database
            row_id = self.db.log_violation(
                vehicle_type=candidate.vehicle_type,
                violation_type=violation_type,
                track_id=candidate.track_id,
                license_plate=plate_text,
                helmet_status=helmet_status,
                gemini_safety_validation=safety_result.model_dump(),
                gemini_confidence=safety_result.confidence_score,
                severity=safety_result.severity_level,
                status_tilang=status_tilang,
                evidence_rider_path=str(candidate.rider_crop_path) if candidate.rider_crop_path else "",
                evidence_plate_path=str(candidate.plate_crop_path) if candidate.plate_crop_path else "",
                timestamp=candidate.timestamp,
            )

            # Integrasi ke PhotoCollection OOP
            if self.photo_collection is not None:
                try:
                    photo_res = PhotoAnalysisResult(
                        caption=f"ETLE Kendaraan ID {candidate.track_id} ({candidate.vehicle_type}). Status: {violation_type}. Plat: {plate_text}.",
                        tags=[candidate.vehicle_type, violation_type.lower(), "traffic_pipeline", "etle"],
                        objects=[candidate.vehicle_type, "person"],
                        kategori="Pelanggaran Helm / Keselamatan" if safety_result.violation_detected else "Patuhi Aturan",
                        is_safe=not safety_result.violation_detected,
                        colors=["Abu-abu Aspal"],
                        ocr_text=plate_text,
                        provider="YOLOv8 + Gemini Traffic Pipeline",
                        image_path=str(candidate.rider_crop_path) if candidate.rider_crop_path else "",
                        annotated_image_path=str(candidate.rider_crop_path) if candidate.rider_crop_path else "",
                    )
                    self.photo_collection.tambah(photo_res)
                except Exception as ex:
                    logger.warning("Gagal menambahkan hasil pipeline ke PhotoCollection: %s", ex)

            # Jika pelanggaran, dorong ke antrean alert
            if safety_result.violation_detected:
                alert = ViolationAlert(
                    id=row_id,
                    timestamp=candidate.timestamp,
                    track_id=candidate.track_id,
                    vehicle_type=candidate.vehicle_type,
                    license_plate=plate_text,
                    violation_type=violation_type,
                    severity=safety_result.severity_level,
                    helmet_status=helmet_status,
                    recommended_action=safety_result.recommended_action,
                    officer_notes=safety_result.officer_notes,
                    rider_image_path=str(candidate.rider_crop_path) if candidate.rider_crop_path else None,
                    plate_image_path=str(candidate.plate_crop_path) if candidate.plate_crop_path else None,
                )
                self.recent_alerts.insert(0, alert)
                if len(self.recent_alerts) > 20:
                    self.recent_alerts.pop()

                logger.info(
                    "Pelanggaran terdeteksi! ID: %s | Plat: %s | Jenis: %s",
                    candidate.track_id,
                    plate_text,
                    violation_type,
                )

        except Exception as e:
            logger.error("Error pada worker analisis pelanggaran: %s", e)

    def process_frame(
        self, frame: np.ndarray, line_ratio: float = COUNTING_LINE_Y_RATIO
    ) -> Tuple[np.ndarray, List[ViolationAlert], Dict[str, Any]]:
        """
        Memproses frame tunggal dalam aliran video:
        - Deteksi dan tracking via detector
        - Menyimpan log kendaraan yang melintasi garis ke database
        - Mendelegasikan kandidat pelanggaran ke worker dengan batas antrean (backpressure)
        - Mengembalikan frame teranotasi, daftar alert terbaru, dan statistik
        """
        annotated_frame, candidates, metadata = self.detector.process_frame(
            frame, line_ratio=line_ratio
        )

        # Rekam kendaraan yang baru saja melintasi garis ke tabel traffic_logs
        for counted in metadata.get("newly_counted", []):
            try:
                self.db.log_traffic_vehicle(
                    track_id=counted["track_id"],
                    vehicle_type=counted["vehicle_type"],
                    direction=counted.get("direction", "MENDEKAT"),
                    confidence=counted.get("confidence", 0.0),
                )
            except Exception as e:
                logger.error("Gagal mencatat log kendaraan melintas: %s", e)

        # Delegasikan kandidat ke worker dengan batasan antrean untuk mencegah lag
        for candidate in candidates:
            if self.async_ai:
                if len(self._active_tasks) < 10:
                    future = self.executor.submit(self._analyze_candidate_worker, candidate)
                    self._active_tasks.add(future)
                    future.add_done_callback(self._active_tasks.discard)
                else:
                    logger.warning(
                        "Antrean AI penuh (%d aktif), kandidat track #%d ditunda.",
                        len(self._active_tasks),
                        candidate.track_id,
                    )
            else:
                self._analyze_candidate_worker(candidate)

        return annotated_frame, self.recent_alerts, metadata

    def verify_document(
        self,
        document_image: Any,
        target_detected_plate: Optional[str] = None,
        save_copy_dir: Optional[Path] = None,
    ) -> Tuple[DocumentValidation, str, int]:
        """
        Memproses verifikasi dokumen (SIM/STNK), membandingkan dengan nomor plat,
        dan menyimpan rekaman audit ke database.
        """
        doc_result, match_status = self.gemini.validate_driving_document(
            document_image, target_detected_plate=target_detected_plate
        )

        legal_status = "LAYAK_JALAN"
        if not doc_result.is_document_valid or doc_result.document_type in ("TIDAK_DIKETAHUI", "TIDAK_VALID"):
            legal_status = "BUKAN_DOKUMEN_RESMI"
        elif doc_result.is_expired:
            legal_status = "DOKUMEN_KEDALUWARSA"
        elif match_status == PlateMatchStatus.MISMATCH.value:
            legal_status = "PLAT_TIDAK_COCOK"

        # Simpan berkas jika ada direktori
        doc_path_str = ""
        if isinstance(document_image, (str, Path)):
            doc_path_str = str(document_image)

        row_id = self.db.log_document_validation(
            document_type=doc_result.document_type,
            document_number=doc_result.document_number,
            holder_name=doc_result.holder_name,
            extracted_plate=doc_result.registered_plate_number,
            target_detected_plate=target_detected_plate,
            plate_match_status=match_status,
            expiry_date=doc_result.expiry_date,
            is_expired=doc_result.is_expired,
            legal_status=legal_status,
            gemini_raw_response=doc_result.model_dump(),
            document_image_path=doc_path_str,
        )

        return doc_result, match_status, row_id

    def process_static_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Memproses citra statis tunggal untuk inspeksi ETLE instan:
        1. Deteksi semua kendaraan dan pengendara via YOLO
        2. Ekstraksi OCR plat nomor untuk tiap kendaraan
        3. Validasi keselamatan helm & atribut via Gemini AI untuk tiap pengendara
        4. Simpan hasil ke database SQLite (traffic_logs & violations)
        5. Visualisasi bounding box kepatuhan (Hijau) / pelanggaran (Merah)

        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
                - Citra teranotasi (OpenCV BGR)
                - Daftar detail investigasi tiap kendaraan
                - Ringkasan metrik statistik
        """
        if image is None or image.size == 0:
            return image, [], {}

        vehicles, persons, candidates = self.detector.detect_static_image(image)
        inspections: List[Dict[str, Any]] = []

        total_vehicles = len(vehicles)
        total_violations = 0

        for candidate in candidates:
            # Simpan bukti citra jika ada path
            if candidate.rider_crop is not None and candidate.rider_crop_path:
                cv2.imwrite(str(candidate.rider_crop_path), candidate.rider_crop)
            if candidate.plate_crop is not None and candidate.plate_crop_path:
                cv2.imwrite(str(candidate.plate_crop_path), candidate.plate_crop)

            # Pembacaan OCR Plat Nomor
            plate_text = "TIDAK_TERBACA"
            if candidate.plate_crop is not None:
                plate_text, plate_conf = self.ocr.read_license_plate(candidate.plate_crop)

            # Validasi Keselamatan AI
            safety_result: Optional[RiderSafetyValidation] = None
            if candidate.rider_crop is not None:
                safety_result = self.gemini.validate_rider_safety(candidate.rider_crop)
            else:
                safety_result = RiderSafetyValidation(
                    is_helmet_present=True,
                    helmet_worn_properly=True,
                    helmet_type="BUKAN_SEPEDA_MOTOR",
                    jacket_present=True,
                    other_attributes=[],
                    violation_detected=False,
                    violation_reasons=[],
                    severity_level="NONE",
                    confidence_score=0.95,
                    officer_notes="Kendaraan tertutup, kepatuhan helm standar.",
                    is_simulated = getattr(safety_result, "_simulated", False)
                    recommended_action="LEPASKAN",
                )

            # Status Pelanggaran
            is_violation = safety_result.violation_detected
            if is_violation:
                total_violations += 1
                violation_type = (
                    ViolationType.NO_HELMET.value
                    if not safety_result.is_helmet_present
                    else ViolationType.NON_STANDARD.value
                )
                status_tilang = EnforcementStatus.CONFIRMED_TICKET.value
                helmet_status = (
                    HelmetStatus.NOT_WORN.value
                    if not safety_result.is_helmet_present
                    else HelmetStatus.NON_STANDARD.value
                )
            else:
                violation_type = ViolationType.COMPLIANT.value
                status_tilang = EnforcementStatus.RELEASED.value
                helmet_status = HelmetStatus.WORN.value

            # Simpan ke database
            self.db.log_traffic_vehicle(
                track_id=candidate.track_id,
                vehicle_type=candidate.vehicle_type,
                direction="STATIS",
                confidence=0.9,
                timestamp=candidate.timestamp,
            )

            row_id = self.db.log_violation(
                vehicle_type=candidate.vehicle_type,
                violation_type=violation_type,
                track_id=candidate.track_id,
                license_plate=plate_text,
                helmet_status=helmet_status,
                gemini_safety_validation=safety_result.model_dump(),
                gemini_confidence=safety_result.confidence_score,
                severity=safety_result.severity_level,
                status_tilang=status_tilang,
                evidence_rider_path=str(candidate.rider_crop_path) if candidate.rider_crop_path else "",
                evidence_plate_path=str(candidate.plate_crop_path) if candidate.plate_crop_path else "",
                timestamp=candidate.timestamp,
            )

            inspections.append({
                "id": row_id,
                "track_id": candidate.track_id,
                "vehicle_type": candidate.vehicle_type,
                "bbox": candidate.vehicle_bbox,
                "license_plate": plate_text,
                "violation_detected": is_violation,
                "violation_type": violation_type,
                "severity": safety_result.severity_level,
                "helmet_status": helmet_status,
                "recommended_action": safety_result.recommended_action,
                "officer_notes": safety_result.officer_notes,
                "rider_crop": candidate.rider_crop,
                "plate_crop": candidate.plate_crop,
                "timestamp": candidate.timestamp,
            })

        # Gambar anotasi visual
        annotated_image = self.detector.visualizer.draw_inspected_static_objects(
            image, inspections
        )

        summary = {
            "total_vehicles": total_vehicles,
            "total_violations": total_violations,
            "compliance_rate": (
                round((total_vehicles - total_violations) / total_vehicles * 100, 1)
                if total_vehicles > 0
                else 100.0
            ),
        }

        return annotated_image, inspections, summary

    # Alias untuk kompatibilitas fungsi pemanggilan inspeksi citra statis
    inspect_static_image = process_static_image

    def shutdown(self) -> None:
        """Menghentikan thread pool dengan bersih."""
        self.executor.shutdown(wait=False)


# Singleton instance
pipeline = TrafficMonitoringPipeline()
