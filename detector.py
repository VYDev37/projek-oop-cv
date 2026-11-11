"""
Module: detector.py
Deskripsi: Modul Computer Vision terpadu berbasis YOLOv8 dan ByteTrack untuk deteksi,
           pelacakan (tracking) kendaraan, penghitungan arus lalu lintas (virtual line crossing),
           serta pemotongan citra (cropping) otomatis pengendara dan plat nomor.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import Counter
import numpy as np

from ultralytics import YOLO

from config import (
    DEFAULT_YOLO_MODEL,
    DEFAULT_TRACKER,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    COCO_CLASS_MAP,
    VEHICLE_CLASSES,
    COLOR_PALETTE,
    COUNTING_LINE_Y_RATIO,
    RIDER_EVIDENCE_DIR,
    PLATE_EVIDENCE_DIR,
)
from visualizer import TrafficVisualizer

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """Objek representasi kendaraan atau manusia yang sedang dilacak."""
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    center: Tuple[int, int]
    has_crossed: bool = False
    last_seen: float = field(default_factory=time.time)


@dataclass
class ViolationCandidate:
    """Kandidat pelanggaran yang memerlukan verifikasi lanjutan OCR & Gemini AI."""
    track_id: int
    vehicle_type: str
    vehicle_bbox: Tuple[int, int, int, int]
    rider_bbox: Optional[Tuple[int, int, int, int]] = None
    plate_bbox: Optional[Tuple[int, int, int, int]] = None
    rider_crop: Optional[np.ndarray] = None
    plate_crop: Optional[np.ndarray] = None
    rider_crop_path: Optional[Path] = None
    plate_crop_path: Optional[Path] = None
    timestamp: str = ""
    license_plate: str = "TIDAK_TERBACA"
    plate_confidence: float = 0.0


class TrafficDetector:
    """Detektor dan Pelacak Lalu Lintas Cerdas menggunakan YOLOv8 dan ByteTrack."""

    def __init__(
        self,
        model_path: str = DEFAULT_YOLO_MODEL,
        tracker_config: str = DEFAULT_TRACKER,
        conf_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        visualizer: Optional[TrafficVisualizer] = None,
    ):
        self.model_path = model_path
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.visualizer = visualizer or TrafficVisualizer()
        self.model = None

        # Pelacakan riwayat koordinat pusat untuk kalkulasi garis hitung
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        self.track_class_history: Dict[int, List[str]] = {}
        self.counted_ids: Set[int] = set()
        self.processed_violation_ids: Set[int] = set()

        # Statistik akumulasi
        self.vehicle_counts: Dict[str, int] = {v: 0 for v in VEHICLE_CLASSES}
        self.total_counted: int = 0

        self._load_model()

    def _load_model(self) -> None:
        """Memuat bobot model YOLOv8."""
        try:
            logger.info("Memuat model YOLO: %s...", self.model_path)
            self.model = YOLO(self.model_path)
            logger.info("Model YOLO berhasil dimuat.")
        except Exception as e:
            logger.error("Gagal memuat model YOLO (%s): %s", self.model_path, e)
            raise

    def reset_counters(self) -> None:
        """Mereset penghitung lalu lintas dan riwayat pelacakan."""
        self.track_history.clear()
        self.track_class_history.clear()
        self.counted_ids.clear()
        self.processed_violation_ids.clear()
        self.vehicle_counts = {v: 0 for v in VEHICLE_CLASSES}
        self.total_counted = 0

    def _crop_with_margin(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int], margin_ratio: float = 0.1
    ) -> Optional[np.ndarray]:
        """Memotong area citra sesuai bounding box dengan penambahan margin aman."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        mx = int(bw * margin_ratio)
        my = int(bh * margin_ratio)

        nx1 = max(0, x1 - mx)
        ny1 = max(0, y1 - my)
        nx2 = min(w, x2 + mx)
        ny2 = min(h, y2 + my)

        if nx2 <= nx1 or ny2 <= ny1:
            return None

        crop = frame[ny1:ny2, nx1:nx2]
        return crop if crop.size > 0 else None

    def _detect_or_estimate_plate(
        self, frame: np.ndarray, vehicle_bbox: Tuple[int, int, int, int], vehicle_type: str
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], str, float]:
        """
        Mendeteksi letak plat nomor secara presisi menggunakan pendekatan multi-tier:
        1. EasyOCR CRAFT Text Localization di area kendaraan.
        2. Filter Kontur Morfologi Persegi Panjang Plat.
        3. Fallback Heuristik Geometris jika detektor teks belum menemukan posisi plat.

        Returns:
            Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], str, float]:
                - Potongan citra plat (crop)
                - Bounding box plat dalam koordinat frame (px1, py1, px2, py2)
                - Teks plat hasil OCR jika langsung terdeteksi
                - Skor keyakinan OCR
        """
        x1, y1, x2, y2 = vehicle_bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        # Fokuskan pencarian pada area 70% bagian bawah kendaraan untuk efisiensi
        search_y1 = max(0, y1 + int(0.30 * h))
        search_roi = frame[search_y1:y2, x1:x2]

        try:
            from ocr_engine import plate_ocr
            if search_roi.size > 0:
                plate_text, conf, global_box, plate_crop = plate_ocr.locate_and_read_plate(
                    search_roi, offset=(x1, search_y1)
                )
                if plate_crop is not None and global_box is not None:
                    return plate_crop, global_box, plate_text, conf
        except Exception as e:
            logger.debug("Lokalisasi cerdas plat dilewati: %s", e)

        # Fallback ke estimasi geometris adaptif
        if vehicle_type == "motorcycle":
            px1 = int(x1 + 0.12 * w)
            px2 = int(x2 - 0.12 * w)
            py1 = int(y1 + 0.60 * h)
            py2 = y2
        else:
            px1 = int(x1 + 0.15 * w)
            px2 = int(x2 - 0.15 * w)
            py1 = int(y1 + 0.65 * h)
            py2 = y2

        plate_bbox = (max(0, px1), max(0, py1), min(frame.shape[1], px2), min(frame.shape[0], py2))
        crop = self._crop_with_margin(frame, plate_bbox, margin_ratio=0.05)
        # Jika teks plat tidak terkonfirmasi, jangan kembalikan bounding box plat semu (hindari salah sorot)
        return crop, None, "TIDAK_TERBACA", 0.0

    def _find_associated_rider(
        self, vehicle_bbox: Tuple[int, int, int, int], persons: List[TrackedObject]
    ) -> Optional[TrackedObject]:
        """
        Mencari objek person yang benar-benar menunggangi sepeda motor.
        Mencegah kesalahan asosiasi terhadap polisi atau pejalan kaki yang berdiri di tepi jalan.
        """
        vx1, vy1, vx2, vy2 = vehicle_bbox
        vw = max(1, vx2 - vx1)
        vh = max(1, vy2 - vy1)
        v_cx = (vx1 + vx2) / 2

        best_candidate: Optional[TrackedObject] = None
        best_score = -1.0

        for person in persons:
            px1, py1, px2, py2 = person.bbox
            pw = max(1, px2 - px1)
            p_cx = (px1 + px2) / 2

            # 1. Titik tengah pengendara harus berada di atas rentang horizontal motor
            if not (vx1 - 0.05 * vw <= p_cx <= vx2 + 0.05 * vw):
                continue

            # 2. Overlap horizontal signifikan
            overlap_x = max(0, min(vx2, px2) - max(vx1, px1))
            overlap_ratio = overlap_x / pw
            if overlap_ratio < 0.35:
                continue

            # 3. Posisi vertikal: Kepala/badan harus berada di paruh atas motor atau di atasnya
            if py1 > vy1 + 0.50 * vh:
                continue
            # Kaki tidak boleh berdiri jauh di bawah motor (orang di trotoar/tepi jalan)
            if py2 > vy2 + 0.15 * vh:
                continue

            # Skor gabungan: semakin mendekati garis tengah motor, semakin tinggi prioritasnya
            norm_dist_x = abs(p_cx - v_cx) / vw
            score = overlap_ratio * 1.5 - norm_dist_x

            if score > best_score:
                best_score = score
                best_candidate = person

        return best_candidate

    def _estimate_plate_region(
        self, frame: np.ndarray, vehicle_bbox: Tuple[int, int, int, int], vehicle_type: str
    ) -> Optional[np.ndarray]:
        """
        Wrapper kompatibilitas untuk mengestimasi dan memotong area plat nomor.
        """
        crop, _, _, _ = self._detect_or_estimate_plate(frame, vehicle_bbox, vehicle_type)
        return crop

    def process_frame(
        self, frame: np.ndarray, line_ratio: float = COUNTING_LINE_Y_RATIO
    ) -> Tuple[np.ndarray, List[ViolationCandidate], Dict[str, Any]]:
        """
        Memproses satu frame video:
        1. Menjalankan deteksi dan pelacakan ByteTrack
        2. Mengelompokkan kendaraan dan pengendara (person)
        3. Menghitung kendaraan yang melintasi garis hitung virtual
        4. Mengidentifikasi dan memotong kandidat pelanggaran
        5. Menggambar anotasi visual ke frame

        Returns:
            Tuple[np.ndarray, List[ViolationCandidate], Dict[str, Any]]:
                - Frame teranotasi (OpenCV BGR)
                - Daftar kandidat pelanggaran baru yang perlu diuji
                - Metadata statistik deteksi
        """
        if frame is None or frame.size == 0:
            return frame, [], {}

        h, w = frame.shape[:2]
        line_y = int(h * line_ratio)
        new_violation_candidates: List[ViolationCandidate] = []
        newly_counted_vehicles: List[Dict[str, Any]] = []

        # Jalankan YOLO + ByteTrack
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        tracked_vehicles: List[TrackedObject] = []
        detected_persons: List[TrackedObject] = []

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            for bbox, tid, cid, conf in zip(boxes, track_ids, class_ids, confs):
                cls_name = COCO_CLASS_MAP.get(cid, None)
                if cls_name is None:
                    continue

                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                tid_int = int(tid)

                # Stabilisasi kelas kendaraan melalui temporal majority voting
                if cls_name in VEHICLE_CLASSES:
                    if tid_int not in self.track_class_history:
                        self.track_class_history[tid_int] = []
                    self.track_class_history[tid_int].append(cls_name)
                    if len(self.track_class_history[tid_int]) > 30:
                        self.track_class_history[tid_int].pop(0)

                    consensus_cls = Counter(self.track_class_history[tid_int]).most_common(1)[0][0]
                    cls_name = consensus_cls

                obj = TrackedObject(
                    track_id=tid_int,
                    class_name=cls_name,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    center=(cx, cy),
                )

                if cls_name in VEHICLE_CLASSES:
                    tracked_vehicles.append(obj)
                elif cls_name == "person":
                    detected_persons.append(obj)

        # ----------------------------------------------------------------------
        # A. LOGIKA PENGHITUNGAN GARIS (LINE CROSSING)
        # ----------------------------------------------------------------------
        for vehicle in tracked_vehicles:
            tid = vehicle.track_id
            cx, cy = vehicle.center

            if tid not in self.track_history:
                self.track_history[tid] = []
            self.track_history[tid].append((cx, cy))

            # Batasi panjang history pelacakan
            if len(self.track_history[tid]) > 30:
                self.track_history[tid].pop(0)

            # Cek lintasan melewati garis
            if tid not in self.counted_ids and len(self.track_history[tid]) >= 2:
                prev_y = self.track_history[tid][-2][1]
                curr_y = cy

                # Mendeteksi penyeberangan dari atas ke bawah atau sebaliknya
                if (prev_y < line_y <= curr_y) or (prev_y > line_y >= curr_y):
                    self.counted_ids.add(tid)
                    vehicle.has_crossed = True
                    self.total_counted += 1
                    self.vehicle_counts[vehicle.class_name] = (
                        self.vehicle_counts.get(vehicle.class_name, 0) + 1
                    )
                    direction = "MENDEKAT" if curr_y > prev_y else "MENJAUH"
                    newly_counted_vehicles.append({
                        "track_id": tid,
                        "vehicle_type": vehicle.class_name,
                        "direction": direction,
                        "confidence": vehicle.confidence,
                    })

        # ----------------------------------------------------------------------
        # B. ASOSIASI PENGENDARA DENGAN KENDARAAN & DETEKSI PELANGGARAN
        # ----------------------------------------------------------------------
        for vehicle in tracked_vehicles:
            tid = vehicle.track_id
            vx1, vy1, vx2, vy2 = vehicle.bbox
            v_type = vehicle.class_name

            # Deteksi khusus Sepeda Motor: Asosiasikan dengan Pengendara
            associated_rider: Optional[TrackedObject] = None
            rider_box: Optional[Tuple[int, int, int, int]] = None
            if v_type == "motorcycle":
                associated_rider = self._find_associated_rider(vehicle.bbox, detected_persons)

                # Pemicu evaluasi pelanggaran (Crop Pengendara & Plat)
                # Evaluasi dipicu jika kendaraan mendekati/melewati area pantau dan belum pernah diproses
                if tid not in self.processed_violation_ids and vehicle.center[1] >= (line_y - 80):
                    self.processed_violation_ids.add(tid)
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")

                    # Crop pengendara
                    rider_crop = None
                    rider_crop_path = None
                    if associated_rider is not None:
                        rider_crop = self._crop_with_margin(frame, associated_rider.bbox, margin_ratio=0.15)
                        rider_box = associated_rider.bbox
                    else:
                        # Jika objek person tidak terpisah, crop 55% area atas sepeda motor
                        upper_motor_bbox = (vx1, vy1, vx2, int(vy1 + 0.55 * (vy2 - vy1)))
                        rider_crop = self._crop_with_margin(frame, upper_motor_bbox, margin_ratio=0.1)
                        rider_box = upper_motor_bbox

                    if rider_crop is not None:
                        rider_filename = f"rider_{timestamp_str}_id{tid}.jpg"
                        rider_crop_path = RIDER_EVIDENCE_DIR / rider_filename

                    # Deteksi / crop plat nomor secara presisi (CRAFT + Kontur + Geometri)
                    plate_crop, plate_bbox, detected_plate_text, detected_plate_conf = (
                        self._detect_or_estimate_plate(frame, vehicle.bbox, v_type)
                    )
                    plate_crop_path = None
                    if plate_crop is not None:
                        plate_filename = f"plate_{timestamp_str}_id{tid}.jpg"
                        plate_crop_path = PLATE_EVIDENCE_DIR / plate_filename

                    candidate = ViolationCandidate(
                        track_id=tid,
                        vehicle_type=v_type,
                        vehicle_bbox=vehicle.bbox,
                        rider_bbox=rider_box,
                        plate_bbox=plate_bbox,
                        rider_crop=rider_crop,
                        plate_crop=plate_crop,
                        rider_crop_path=rider_crop_path,
                        plate_crop_path=plate_crop_path,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        license_plate=detected_plate_text,
                        plate_confidence=detected_plate_conf,
                    )
                    new_violation_candidates.append(candidate)

        # ----------------------------------------------------------------------
        # C. DELEGASIKAN RENDERING ANOTASI KE TRAFFIC VISUALIZER
        # ----------------------------------------------------------------------
        annotated_frame = self.visualizer.annotate_frame(
            frame=frame,
            line_y=line_y,
            tracked_vehicles=tracked_vehicles,
            detected_persons=detected_persons,
            track_history=self.track_history,
            total_counted=self.total_counted,
            vehicle_counts=self.vehicle_counts,
        )

        metadata = {
            "total_counted": self.total_counted,
            "vehicle_counts": self.vehicle_counts,
            "active_tracks": len(tracked_vehicles),
            "newly_counted": newly_counted_vehicles,
        }

        return annotated_frame, new_violation_candidates, metadata

    def detect_static_image(
        self, image: np.ndarray
    ) -> Tuple[List[TrackedObject], List[TrackedObject], List[ViolationCandidate]]:
        """
        Mendeteksi seluruh kendaraan dan pejalan kaki/pengendara pada citra statis
        tanpa bergantung pada pelacakan temporal (ByteTrack).
        Menghasilkan kandidat evaluasi keselamatan dan plat nomor untuk setiap kendaraan.
        """
        if image is None or image.size == 0:
            return [], [], []

        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detected_vehicles: List[TrackedObject] = []
        detected_persons: List[TrackedObject] = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            track_counter = 1
            for bbox, cid, conf in zip(boxes, class_ids, confs):
                cls_name = COCO_CLASS_MAP.get(cid, None)
                if cls_name is None:
                    continue

                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                obj = TrackedObject(
                    track_id=track_counter,
                    class_name=cls_name,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    center=(cx, cy),
                )
                track_counter += 1

                if cls_name in VEHICLE_CLASSES:
                    detected_vehicles.append(obj)
                elif cls_name == "person":
                    detected_persons.append(obj)

        # Asosiasi Pengendara & Buat Kandidat Inspeksi
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        candidates: List[ViolationCandidate] = []

        for vehicle in detected_vehicles:
            tid = vehicle.track_id
            vx1, vy1, vx2, vy2 = vehicle.bbox
            v_type = vehicle.class_name

            associated_rider: Optional[TrackedObject] = None
            rider_crop = None
            rider_crop_path = None
            rider_box: Optional[Tuple[int, int, int, int]] = None

            if v_type == "motorcycle":
                associated_rider = self._find_associated_rider(vehicle.bbox, detected_persons)

                if associated_rider is not None:
                    rider_crop = self._crop_with_margin(image, associated_rider.bbox, margin_ratio=0.15)
                    rider_box = associated_rider.bbox
                else:
                    # Ambil 55% area atas sepeda motor (kepala/torso pengendara)
                    upper_motor_bbox = (vx1, vy1, vx2, int(vy1 + 0.55 * (vy2 - vy1)))
                    rider_crop = self._crop_with_margin(image, upper_motor_bbox, margin_ratio=0.1)
                    rider_box = upper_motor_bbox

                if rider_crop is not None:
                    rider_crop_path = RIDER_EVIDENCE_DIR / f"static_rider_{timestamp_str}_id{tid}.jpg"

            # Deteksi / crop plat nomor secara presisi (CRAFT + Kontur + Geometri)
            plate_crop, plate_bbox, detected_plate_text, detected_plate_conf = (
                self._detect_or_estimate_plate(image, vehicle.bbox, v_type)
            )
            plate_crop_path = None
            if plate_crop is not None:
                plate_crop_path = PLATE_EVIDENCE_DIR / f"static_plate_{timestamp_str}_id{tid}.jpg"

            candidates.append(
                ViolationCandidate(
                    track_id=tid,
                    vehicle_type=v_type,
                    vehicle_bbox=vehicle.bbox,
                    rider_bbox=rider_box,
                    plate_bbox=plate_bbox,
                    rider_crop=rider_crop,
                    plate_crop=plate_crop,
                    rider_crop_path=rider_crop_path,
                    plate_crop_path=plate_crop_path,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    license_plate=detected_plate_text,
                    plate_confidence=detected_plate_conf,
                )
            )

        return detected_vehicles, detected_persons, candidates
