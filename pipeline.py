"""
Module: pipeline.py
Deskripsi: Pipeline orkestrasi terpadu yang menghubungkan Modul Deteksi (YOLO + ByteTrack),
           Modul OCR (EasyOCR), Modul Gemini AI (google-genai), dan Manajemen Database (SQLite).
           Dilengkapi pemrosesan asinkronus (background thread) agar pemutaran video tidak freeze.
"""

import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import cv2
import numpy as np

from config import COUNTING_LINE_Y_RATIO, RIDER_EVIDENCE_DIR, PLATE_EVIDENCE_DIR
from database import db_manager, DatabaseManager
from detector import TrafficDetector, ViolationCandidate
from ocr_engine import plate_ocr, LicensePlateOCR
from gemini_client import (
    gemini_client,
    GeminiTrafficClient,
    RiderSafetyValidation,
    DocumentValidation,
    TrafficSceneAnalysis,
    VehicleSceneResult,
)
from domain_models import VehicleType, ViolationType, HelmetStatus, EnforcementStatus, PlateMatchStatus
from vision_client import PhotoAnalysisResult
from photo_collection import PhotoCollection

logger = logging.getLogger(__name__)

_VEHICLE_TYPE_ALIASES: Dict[str, str] = {
    "motorcycle": VehicleType.MOTORCYCLE.value,
    "motorbike": VehicleType.MOTORCYCLE.value,
    "motor": VehicleType.MOTORCYCLE.value,
    "sepeda motor": VehicleType.MOTORCYCLE.value,
    "scooter": VehicleType.MOTORCYCLE.value,
    "car": VehicleType.CAR.value,
    "mobil": VehicleType.CAR.value,
    "bus": VehicleType.BUS.value,
    "truck": VehicleType.TRUCK.value,
    "truk": VehicleType.TRUCK.value,
    "bicycle": VehicleType.BICYCLE.value,
    "sepeda": VehicleType.BICYCLE.value,
}

_WATERMARK_MARKERS = ("WWW.", ".COM", ".ID", ".NET", ".ORG", "HTTP", "ANTARA", "WATERMARK")


def _normalize_vehicle_type(raw_type: Optional[str]) -> str:
    """Menstandarkan label jenis kendaraan ke nilai kanonik VehicleType."""
    cleaned = (raw_type or "").strip().lower()
    return _VEHICLE_TYPE_ALIASES.get(cleaned, VehicleType.MOTORCYCLE.value if "motor" in cleaned else (cleaned or VehicleType.CAR.value))


def _sanitize_plate_text(raw_plate: Optional[str]) -> str:
    """Membersihkan teks plat nomor dan menolak watermark media / string non-plat."""
    if not raw_plate:
        return "TIDAK_TERBACA"
    text = re.sub(r"\s+", " ", raw_plate.upper()).strip()
    if text in ("TIDAK_TERBACA", "TIDAK_VALID", "TIDAK_DIKETAHUI", "NONE", "N/A", "-", ""):
        return "TIDAK_TERBACA"
    if any(marker in text for marker in _WATERMARK_MARKERS):
        return "TIDAK_TERBACA"
    alnum_only = re.sub(r"[^A-Z0-9]", "", text)
    if len(alnum_only) < 3 or not any(ch.isdigit() for ch in alnum_only):
        return "TIDAK_TERBACA"
    return text


def _bbox_1000_to_pixels(
    bbox_1000: Optional[List[int]], width: int, height: int
) -> Optional[Tuple[int, int, int, int]]:
    """Mengonversi koordinat [ymin, xmin, ymax, xmax] skala 0-1000 ke piksel (x1, y1, x2, y2)."""
    if not bbox_1000 or len(bbox_1000) != 4 or width <= 0 or height <= 0:
        return None
    try:
        ymin, xmin, ymax, xmax = [float(v) for v in bbox_1000]
    except (TypeError, ValueError):
        return None

    y1 = int(round(max(0.0, min(1000.0, ymin)) / 1000.0 * height))
    x1 = int(round(max(0.0, min(1000.0, xmin)) / 1000.0 * width))
    y2 = int(round(max(0.0, min(1000.0, ymax)) / 1000.0 * height))
    x2 = int(round(max(0.0, min(1000.0, xmax)) / 1000.0 * width))

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None
    return (x1, y1, x2, y2)


def _compute_box_metrics(
    box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]
) -> Tuple[float, float]:
    """Menghitung IoU dan Overlap-over-Min-Area (IoM) antara dua bounding box piksel."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0, 0.0

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = max(1, area_a + area_b - inter)
    return inter / union, inter / min(area_a, area_b)


def _is_rider_box_on_vehicle(
    rider_box: Tuple[int, int, int, int], vehicle_box: Tuple[int, int, int, int]
) -> bool:
    """Memastikan koordinat pengendara dari Gemini masih relevan dengan unit motor terkait."""
    vx1, vy1, vx2, vy2 = vehicle_box
    vw, vh = max(1, vx2 - vx1), max(1, vy2 - vy1)
    zone = (
        int(vx1 - 0.35 * vw),
        int(vy1 - 0.85 * vh),
        int(vx2 + 0.35 * vw),
        int(vy2 + 0.10 * vh),
    )
    _, iom = _compute_box_metrics(rider_box, zone)
    return iom >= 0.40


def _is_box_inside_expanded(
    inner_box: Tuple[int, int, int, int], outer_box: Tuple[int, int, int, int], expand_ratio: float = 0.20
) -> bool:
    """Memastikan bounding box (misal plat) berada di dalam area kendaraan yang diperluas."""
    ox1, oy1, ox2, oy2 = outer_box
    ow, oh = max(1, ox2 - ox1), max(1, oy2 - oy1)
    expanded = (
        int(ox1 - expand_ratio * ow),
        int(oy1 - expand_ratio * oh),
        int(ox2 + expand_ratio * ow),
        int(oy2 + expand_ratio * oh),
    )
    _, iom = _compute_box_metrics(inner_box, expanded)
    return iom >= 0.50


def _derive_enforcement_fields(
    violation_detected: bool, is_helmet_present: bool, is_simulated: bool
) -> Tuple[str, str, str]:
    """Menurunkan tuple (violation_type, status_tilang, helmet_status) secara seragam."""
    if not violation_detected:
        return (
            ViolationType.COMPLIANT.value,
            EnforcementStatus.RELEASED.value,
            HelmetStatus.WORN.value,
        )
    violation_type = (
        ViolationType.NO_HELMET.value if not is_helmet_present else ViolationType.NON_STANDARD.value
    )
    status_tilang = (
        EnforcementStatus.PENDING_REVIEW.value
        if is_simulated
        else EnforcementStatus.CONFIRMED_TICKET.value
    )
    helmet_status = (
        HelmetStatus.NOT_WORN.value if not is_helmet_present else HelmetStatus.NON_STANDARD.value
    )
    return violation_type, status_tilang, helmet_status


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
        self.__active_tasks: Set[Any] = set()
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

            plate_text = getattr(candidate, "license_plate", "TIDAK_TERBACA")
            plate_conf = getattr(candidate, "plate_confidence", 0.0)

            # Pembacaan OCR Plat Nomor jika belum terbaca
            if plate_text == "TIDAK_TERBACA":
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
            violation_type, status_tilang, helmet_status = _derive_enforcement_fields(
                safety_result.violation_detected, safety_result.is_helmet_present, is_simulated
            )
                       
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
                if len(self.__active_tasks) < 10:
                    future = self.executor.submit(self._analyze_candidate_worker, candidate)
                    self.__active_tasks.add(future)
                    future.add_done_callback(self.__active_tasks.discard)
                else:
                    logger.warning(
                        "Antrean AI penuh (%d aktif), kandidat track #%d ditunda.",
                        len(self.__active_tasks),
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

    def _match_scene_vehicles(
        self,
        candidates: List[ViolationCandidate],
        scene_vehicles: List[VehicleSceneResult],
        width: int,
        height: int,
    ) -> Tuple[Dict[int, VehicleSceneResult], List[VehicleSceneResult]]:
        """Memasangkan hasil deteksi kendaraan Gemini (full-image) dengan kandidat YOLO 1-ke-1."""
        pairs: List[Tuple[float, int, int]] = []

        for g_idx, g_veh in enumerate(scene_vehicles):
            g_box = _bbox_1000_to_pixels(g_veh.vehicle_bbox_1000, width, height)
            if g_box is None:
                continue
            g_type = _normalize_vehicle_type(g_veh.vehicle_type)

            for c_idx, cand in enumerate(candidates):
                iou, iom = _compute_box_metrics(cand.vehicle_bbox, g_box)
                if iou < 0.12 and iom < 0.35:
                    continue
                same_type = cand.vehicle_type == g_type
                both_two_wheelers = {cand.vehicle_type, g_type} <= {
                    VehicleType.MOTORCYCLE.value,
                    VehicleType.BICYCLE.value,
                }
                if not same_type and not both_two_wheelers and iou < 0.45:
                    continue
                score = iou * 2.0 + iom + (0.30 if same_type else 0.0)
                pairs.append((score, c_idx, g_idx))

        pairs.sort(key=lambda item: item[0], reverse=True)
        matched_by_cand: Dict[int, VehicleSceneResult] = {}
        matched_g_indices: Set[int] = set()

        for _, c_idx, g_idx in pairs:
            if c_idx in matched_by_cand or g_idx in matched_g_indices:
                continue
            matched_by_cand[c_idx] = scene_vehicles[g_idx]
            matched_g_indices.add(g_idx)

        unmatched_scene = [
            g_veh for g_idx, g_veh in enumerate(scene_vehicles) if g_idx not in matched_g_indices
        ]
        return matched_by_cand, unmatched_scene

    def _evaluate_candidate_per_crop(self, candidate: ViolationCandidate) -> RiderSafetyValidation:
        """Evaluasi keselamatan pengendara berbasis crop tunggal sebagai mekanisme fallback."""
        if candidate.rider_crop is not None:
            return self.gemini.validate_rider_safety(candidate.rider_crop)
        return RiderSafetyValidation(
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
            recommended_action="LEPASKAN",
        )

    def _record_inspection(
        self,
        *,
        track_id: int,
        vehicle_type: str,
        bbox: Optional[Tuple[int, int, int, int]],
        rider_bbox: Optional[Tuple[int, int, int, int]],
        plate_bbox: Optional[Tuple[int, int, int, int]],
        rider_crop: Optional[np.ndarray],
        plate_crop: Optional[np.ndarray],
        rider_crop_path: Optional[Path],
        plate_crop_path: Optional[Path],
        license_plate: str,
        violation_detected: bool,
        is_helmet_present: bool,
        is_simulated: bool,
        safety_payload: Dict[str, Any],
        confidence_score: float,
        severity: str,
        recommended_action: str,
        officer_notes: str,
        timestamp: str,
        analysis_source: str,
        rider_role_description: str = "",
        violation_reasons: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Menyimpan bukti crop, mencatat rekaman DB, dan membentuk kamus inspeksi UI."""
        if rider_crop is not None and rider_crop_path is not None:
            cv2.imwrite(str(rider_crop_path), rider_crop)
        if plate_crop is not None and plate_crop_path is not None:
            cv2.imwrite(str(plate_crop_path), plate_crop)

        violation_type, status_tilang, helmet_status = _derive_enforcement_fields(
            violation_detected, is_helmet_present, is_simulated
        )

        self.db.log_traffic_vehicle(
            track_id=track_id,
            vehicle_type=vehicle_type,
            direction="STATIS",
            confidence=confidence_score,
            timestamp=timestamp,
        )

        row_id = self.db.log_violation(
            vehicle_type=vehicle_type,
            violation_type=violation_type,
            track_id=track_id,
            license_plate=license_plate,
            helmet_status=helmet_status,
            gemini_safety_validation=safety_payload,
            gemini_confidence=confidence_score,
            severity=severity,
            status_tilang=status_tilang,
            evidence_rider_path=str(rider_crop_path) if (rider_crop is not None and rider_crop_path) else "",
            evidence_plate_path=str(plate_crop_path) if (plate_crop is not None and plate_crop_path) else "",
            timestamp=timestamp,
        )

        return {
            "id": row_id,
            "track_id": track_id,
            "vehicle_type": vehicle_type,
            "bbox": bbox,
            "plate_bbox": plate_bbox,
            "rider_bbox": rider_bbox,
            "license_plate": license_plate,
            "violation_detected": violation_detected,
            "violation_type": violation_type,
            "severity": severity,
            "helmet_status": helmet_status,
            "recommended_action": recommended_action,
            "officer_notes": officer_notes,
            "rider_crop": rider_crop,
            "plate_crop": plate_crop,
            "timestamp": timestamp,
            "analysis_source": analysis_source,
            "rider_role_description": rider_role_description,
            "violation_reasons": violation_reasons or [],
            "is_simulated": is_simulated,
        }

    def process_static_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Memproses citra statis tunggal untuk inspeksi ETLE instan (Hybrid YOLO + Full-Image Gemini):
        1. Deteksi kendaraan dan lokalisasi awal via YOLO + EasyOCR
        2. Analisis holistik full-image via Gemini AI untuk seluruh adegan jalan raya (Jalur B)
        3. Penggabungan bounding box YOLO dengan keputusan akurat Gemini (termasuk kendaraan luput YOLO)
        4. Fallback otomatis ke mode per-crop jika Gemini API tidak aktif/gagal
        5. Visualisasi bounding box kepatuhan (Hijau) / pelanggaran (Merah)

        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
                - Citra teranotasi (OpenCV BGR)
                - Daftar detail investigasi tiap kendaraan
                - Ringkasan metrik statistik
        """
        if image is None or image.size == 0:
            return image, [], {}

        h, w = image.shape[:2]
        _, _, candidates = self.detector.detect_static_image(image)
        inspections: List[Dict[str, Any]] = []

        scene_analysis: Optional[TrafficSceneAnalysis] = None
        use_holistic = False
        try:
            scene_analysis = self.gemini.analyze_traffic_scene(image)
            if scene_analysis is not None and not getattr(scene_analysis, "_simulated", False):
                use_holistic = True
        except Exception as e:
            logger.warning("Analisis full-image Gemini gagal, beralih ke fallback per-crop: %s", e)

        matched_by_cand: Dict[int, VehicleSceneResult] = {}
        unmatched_scene: List[VehicleSceneResult] = []
        if use_holistic and scene_analysis is not None:
            matched_by_cand, unmatched_scene = self._match_scene_vehicles(
                candidates, scene_analysis.vehicles, w, h
            )

        # 1. Proses seluruh kandidat hasil deteksi YOLO (diperkaya hasil holistik Gemini bila cocok)
        for c_idx, candidate in enumerate(candidates):
            plate_text = _sanitize_plate_text(getattr(candidate, "license_plate", "TIDAK_TERBACA"))
            if plate_text == "TIDAK_TERBACA" and candidate.plate_crop is not None:
                ocr_text, _ = self.ocr.read_license_plate(candidate.plate_crop)
                plate_text = _sanitize_plate_text(ocr_text)

            matched_g = matched_by_cand.get(c_idx)
            if matched_g is not None:
                rider_bbox = candidate.rider_bbox
                rider_crop = candidate.rider_crop
                if candidate.vehicle_type == VehicleType.MOTORCYCLE.value:
                    g_rider_box = _bbox_1000_to_pixels(matched_g.rider_bbox_1000, w, h)
                    if g_rider_box is not None and _is_rider_box_on_vehicle(g_rider_box, candidate.vehicle_bbox):
                        rider_bbox = g_rider_box
                        cropped_rider = self.detector._crop_with_margin(image, g_rider_box, margin_ratio=0.10)
                        if cropped_rider is not None:
                            rider_crop = cropped_rider

                plate_bbox = candidate.plate_bbox if plate_text != "TIDAK_TERBACA" else None
                plate_crop = candidate.plate_crop
                g_plate = _sanitize_plate_text(matched_g.license_plate)
                if g_plate != "TIDAK_TERBACA":
                    plate_text = g_plate
                    g_plate_box = _bbox_1000_to_pixels(matched_g.plate_bbox_1000, w, h)
                    if g_plate_box is not None and _is_box_inside_expanded(g_plate_box, candidate.vehicle_bbox):
                        plate_bbox = g_plate_box
                        cropped_plate = self.detector._crop_with_margin(image, g_plate_box, margin_ratio=0.08)
                        if cropped_plate is not None:
                            plate_crop = cropped_plate

                inspections.append(
                    self._record_inspection(
                        track_id=candidate.track_id,
                        vehicle_type=candidate.vehicle_type,
                        bbox=candidate.vehicle_bbox,
                        rider_bbox=rider_bbox,
                        plate_bbox=plate_bbox,
                        rider_crop=rider_crop,
                        plate_crop=plate_crop,
                        rider_crop_path=candidate.rider_crop_path,
                        plate_crop_path=candidate.plate_crop_path,
                        license_plate=plate_text,
                        violation_detected=matched_g.violation_detected,
                        is_helmet_present=matched_g.is_helmet_present,
                        is_simulated=False,
                        safety_payload=matched_g.model_dump(),
                        confidence_score=matched_g.confidence_score,
                        severity=matched_g.severity_level,
                        recommended_action=matched_g.recommended_action,
                        officer_notes=matched_g.officer_notes,
                        timestamp=candidate.timestamp,
                        analysis_source="YOLO + Gemini Full-Image",
                        rider_role_description=matched_g.rider_role_description,
                        violation_reasons=matched_g.violation_reasons,
                    )
                )
            else:
                safety_result = self._evaluate_candidate_per_crop(candidate)
                is_simulated = getattr(safety_result, "_simulated", False)
                plate_bbox = candidate.plate_bbox if plate_text != "TIDAK_TERBACA" else None
                inspections.append(
                    self._record_inspection(
                        track_id=candidate.track_id,
                        vehicle_type=candidate.vehicle_type,
                        bbox=candidate.vehicle_bbox,
                        rider_bbox=candidate.rider_bbox,
                        plate_bbox=plate_bbox,
                        rider_crop=candidate.rider_crop,
                        plate_crop=candidate.plate_crop,
                        rider_crop_path=candidate.rider_crop_path,
                        plate_crop_path=candidate.plate_crop_path,
                        license_plate=plate_text,
                        violation_detected=safety_result.violation_detected,
                        is_helmet_present=safety_result.is_helmet_present,
                        is_simulated=is_simulated,
                        safety_payload=safety_result.model_dump(),
                        confidence_score=safety_result.confidence_score,
                        severity=safety_result.severity_level,
                        recommended_action=safety_result.recommended_action,
                        officer_notes=safety_result.officer_notes,
                        timestamp=candidate.timestamp,
                        analysis_source="YOLO + Simulasi Lokal" if is_simulated else "YOLO + Gemini Per-Crop",
                        violation_reasons=safety_result.violation_reasons,
                    )
                )

        # 2. Proses kendaraan tambahan yang terlihat di Gemini full-image tetapi terlewat oleh YOLO
        next_track_id = max((c.track_id for c in candidates), default=0) + 1
        ts_file = time.strftime("%Y%m%d_%H%M%S")
        ts_db = time.strftime("%Y-%m-%d %H:%M:%S")

        for g_veh in unmatched_scene:
            v_type = _normalize_vehicle_type(g_veh.vehicle_type)
            v_box = _bbox_1000_to_pixels(g_veh.vehicle_bbox_1000, w, h)
            r_box = _bbox_1000_to_pixels(g_veh.rider_bbox_1000, w, h) if v_type == VehicleType.MOTORCYCLE.value else None
            p_box_raw = _bbox_1000_to_pixels(g_veh.plate_bbox_1000, w, h)

            rider_crop = (
                self.detector._crop_with_margin(image, r_box, margin_ratio=0.10)
                if r_box is not None
                else (
                    self.detector._crop_with_margin(image, v_box, margin_ratio=0.05)
                    if (v_box is not None and v_type == VehicleType.MOTORCYCLE.value)
                    else None
                )
            )
            plate_crop = (
                self.detector._crop_with_margin(image, p_box_raw, margin_ratio=0.08)
                if p_box_raw is not None
                else None
            )

            plate_text = _sanitize_plate_text(g_veh.license_plate)
            if plate_text == "TIDAK_TERBACA" and plate_crop is not None:
                ocr_text, _ = self.ocr.read_license_plate(plate_crop)
                plate_text = _sanitize_plate_text(ocr_text)
            plate_bbox = p_box_raw if (p_box_raw is not None and plate_text != "TIDAK_TERBACA") else None

            rider_path = RIDER_EVIDENCE_DIR / f"static_rider_{ts_file}_id{next_track_id}.jpg" if rider_crop is not None else None
            plate_path = PLATE_EVIDENCE_DIR / f"static_plate_{ts_file}_id{next_track_id}.jpg" if plate_crop is not None else None

            inspections.append(
                self._record_inspection(
                    track_id=next_track_id,
                    vehicle_type=v_type,
                    bbox=v_box,
                    rider_bbox=r_box,
                    plate_bbox=plate_bbox,
                    rider_crop=rider_crop,
                    plate_crop=plate_crop,
                    rider_crop_path=rider_path,
                    plate_crop_path=plate_path,
                    license_plate=plate_text,
                    violation_detected=g_veh.violation_detected,
                    is_helmet_present=g_veh.is_helmet_present,
                    is_simulated=False,
                    safety_payload=g_veh.model_dump(),
                    confidence_score=g_veh.confidence_score,
                    severity=g_veh.severity_level,
                    recommended_action=g_veh.recommended_action,
                    officer_notes=g_veh.officer_notes,
                    timestamp=ts_db,
                    analysis_source="Gemini Full-Image",
                    rider_role_description=g_veh.rider_role_description,
                    violation_reasons=g_veh.violation_reasons,
                )
            )
            next_track_id += 1

        annotated_image = self.detector.visualizer.draw_inspected_static_objects(
            image, inspections
        )

        total_vehicles = len(inspections)
        total_violations = sum(1 for item in inspections if item["violation_detected"])
        analysis_mode = (
            "YOLO + Gemini Full-Image"
            if use_holistic
            else (
                "YOLO + Simulasi Lokal (Tanpa Gemini API Key)"
                if (scene_analysis is not None and getattr(scene_analysis, "_simulated", False))
                else "YOLO + Gemini Per-Crop Fallback"
            )
        )

        summary = {
            "total_vehicles": total_vehicles,
            "total_violations": total_violations,
            "compliance_rate": (
                round((total_vehicles - total_violations) / total_vehicles * 100, 1)
                if total_vehicles > 0
                else 100.0
            ),
            "analysis_mode": analysis_mode,
            "scene_summary": scene_analysis.scene_summary if use_holistic and scene_analysis else "",
        }

        return annotated_image, inspections, summary

    # Alias untuk kompatibilitas fungsi pemanggilan inspeksi citra statis
    inspect_static_image = process_static_image

    def shutdown(self) -> None:
        """Menghentikan thread pool dengan bersih."""
        self.executor.shutdown(wait=False)


# Singleton instance
pipeline = TrafficMonitoringPipeline()
