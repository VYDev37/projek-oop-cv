"""
Unit Test: tests/test_jalur_b_scene_pipeline.py
Deskripsi: Pengujian Jalur B (Hybrid YOLO + Full-Image Gemini Scene Analysis):
           1. Fallback simulasi analyze_traffic_scene & flag _simulated.
           2. Parsing respons terstruktur TrafficSceneAnalysis dari Gemini SDK.
           3. Penggabungan bounding box YOLO + keputusan holistik Gemini serta pemulihan kendaraan luput YOLO.
           4. Sanitasi watermark media agar tidak dianggap sebagai plat nomor.
"""

import numpy as np

from database import DatabaseManager
from detector import ViolationCandidate
from gemini_client import GeminiTrafficClient, TrafficSceneAnalysis, VehicleSceneResult
from pipeline import TrafficMonitoringPipeline, _sanitize_plate_text, _bbox_1000_to_pixels
from visualizer import TrafficVisualizer


class _StubDetector:
    """Detector stub deterministik tanpa menjalankan inferensi berat YOLOv8."""

    def __init__(self, candidates):
        self._candidates = candidates
        self.visualizer = TrafficVisualizer()

    def _crop_with_margin(self, frame, bbox, margin_ratio=0.1):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        nx1 = max(0, x1)
        ny1 = max(0, y1)
        nx2 = min(w, x2)
        ny2 = min(h, y2)
        if nx2 <= nx1 or ny2 <= ny1:
            return None
        return frame[ny1:ny2, nx1:nx2]

    def detect_static_image(self, image):
        return [], [], list(self._candidates)


class _StubOCR:
    """OCR stub offline yang tidak membaca plat."""

    def read_license_plate(self, plate_crop):
        return "TIDAK_TERBACA", 0.0


def test_analyze_traffic_scene_mock_fallback():
    client = GeminiTrafficClient(api_key="")
    dummy = np.zeros((240, 320, 3), dtype=np.uint8)
    result = client.analyze_traffic_scene(dummy)

    assert isinstance(result, TrafficSceneAnalysis)
    assert getattr(result, "_simulated", False) is True
    assert "SIMULASI" in result.scene_summary
    assert len(result.vehicles) >= 1
    assert "_simulated" not in result.model_dump()


def test_analyze_traffic_scene_json_parsing():
    client = GeminiTrafficClient(api_key="dummy_key")

    payload = (
        '{'
        '  "scene_summary": "Dua sepeda motor dan satu polisi berdiri di pinggir jalan.",'
        '  "overall_confidence": 0.94,'
        '  "vehicles": ['
        '    {'
        '      "vehicle_type": "motorcycle",'
        '      "vehicle_bbox_1000": [400, 150, 900, 480],'
        '      "rider_bbox_1000": [220, 200, 700, 440],'
        '      "plate_bbox_1000": [760, 250, 840, 370],'
        '      "license_plate": "BM 6116 OPP",'
        '      "rider_role_description": "Motor tengah berboncengan tanpa helm",'
        '      "is_helmet_present": false,'
        '      "helmet_worn_properly": false,'
        '      "helmet_type": "TIDAK_ADA",'
        '      "jacket_present": false,'
        '      "other_attributes": ["boncengan tanpa helm"],'
        '      "violation_detected": true,'
        '      "violation_reasons": ["Pengemudi dan pengendara tidak memakai helm"],'
        '      "severity_level": "HIGH",'
        '      "confidence_score": 0.95,'
        '      "officer_notes": "Pelanggaran Pasal 291 UU LLAJ.",'
        '      "recommended_action": "TILANG_ELEKTRONIK"'
        '    }'
        '  ]'
        '}'
    )

    class _MockResp:
        parsed = None
        text = payload

    class _MockModels:
        def generate_content(self, *args, **kwargs):
            return _MockResp()

    class _MockClient:
        models = _MockModels()

    client.client = _MockClient()
    dummy = np.zeros((240, 320, 3), dtype=np.uint8)
    result = client.analyze_traffic_scene(dummy)

    assert getattr(result, "_simulated", False) is False
    assert len(result.vehicles) == 1
    assert result.vehicles[0].license_plate == "BM 6116 OPP"
    assert result.vehicles[0].violation_detected is True


def test_pipeline_jalur_b_merges_yolo_and_recovers_missed_vehicle(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "jalur_b.db")
    img = np.zeros((500, 800, 3), dtype=np.uint8)

    # YOLO hanya mendeteksi motor 1 dan keliru mengambil bbox polisi di sebelah kiri
    cand1 = ViolationCandidate(
        track_id=1,
        vehicle_type="motorcycle",
        vehicle_bbox=(120, 200, 380, 450),
        rider_bbox=(10, 50, 100, 350),  # polisi salah sorot di sebelah kiri
        plate_bbox=None,
        rider_crop=img[50:350, 10:100],
        plate_crop=img[350:440, 160:320],
        rider_crop_path=tmp_path / "r1.jpg",
        plate_crop_path=tmp_path / "p1.jpg",
        timestamp="2026-11-11 10:00:00",
        license_plate="TIDAK_TERBACA",
    )

    scene = TrafficSceneAnalysis(
        scene_summary="Polisi berdiri di tepi kiri; dua motor siswa terdeteksi.",
        overall_confidence=0.93,
        vehicles=[
            # Cocok dengan kandidat YOLO #1
            VehicleSceneResult(
                vehicle_type="motorcycle",
                vehicle_bbox_1000=[400, 150, 900, 475],   # y:200..450, x:120..380
                rider_bbox_1000=[220, 190, 720, 430],     # pengendara sebenarnya di atas motor
                plate_bbox_1000=[760, 220, 860, 340],     # plat berada di dalam vehicle_bbox
                license_plate="BM 6116 OPP",
                rider_role_description="Motor tengah, 2 siswa tanpa helm",
                is_helmet_present=False,
                helmet_worn_properly=False,
                helmet_type="TIDAK_ADA",
                jacket_present=False,
                violation_detected=True,
                violation_reasons=["Pengendara dan pengendara tidak memakai helm"],
                severity_level="HIGH",
                confidence_score=0.95,
                officer_notes="Kedua pelajar di motor tengah tidak mengenakan helm.",
                recommended_action="TILANG_ELEKTRONIK",
            ),
            # Motor #2 yang luput dari YOLO (depan kanan)
            VehicleSceneResult(
                vehicle_type="motorcycle",
                vehicle_bbox_1000=[350, 550, 950, 925],   # y:175..475, x:440..740
                rider_bbox_1000=[160, 625, 700, 880],
                plate_bbox_1000=[780, 650, 860, 750],
                license_plate="B 9999 XYZ",
                rider_role_description="Motor kanan depan berhelm hitam",
                is_helmet_present=True,
                helmet_worn_properly=True,
                helmet_type="HALF_FACE",
                jacket_present=False,
                violation_detected=False,
                violation_reasons=[],
                severity_level="NONE",
                confidence_score=0.94,
                officer_notes="Pengendara mengenakan helm SNI dengan baik.",
                recommended_action="LEPASKAN",
            ),
        ],
    )

    gemini = GeminiTrafficClient(api_key="")
    gemini.analyze_traffic_scene = lambda _img: scene

    pipe = TrafficMonitoringPipeline(
        detector=_StubDetector([cand1]),
        ocr=_StubOCR(),
        gemini=gemini,
        database=db,
        async_ai=False,
    )

    annotated, inspections, summary = pipe.process_static_image(img)

    assert annotated is not None
    assert summary["total_vehicles"] == 2
    assert summary["total_violations"] == 1
    assert summary["compliance_rate"] == 50.0
    assert summary["analysis_mode"] == "YOLO + Gemini Full-Image"
    assert "Polisi" in summary["scene_summary"]

    assert len(inspections) == 2
    first, second = inspections[0], inspections[1]

    # Kandidat #1 memakai kotak YOLO tapi keputusan & rider_bbox diperbaiki Gemini
    assert first["bbox"] == (120, 200, 380, 450)
    assert first["rider_bbox"] == _bbox_1000_to_pixels([220, 190, 720, 430], 800, 500)
    assert first["plate_bbox"] == _bbox_1000_to_pixels([760, 220, 860, 340], 800, 500)
    assert first["plate_crop"] is not None
    assert first["violation_detected"] is True
    assert first["license_plate"] == "BM 6116 OPP"
    assert first["analysis_source"] == "YOLO + Gemini Full-Image"

    # Motor #2 dipulihkan murni dari analisis full-image Gemini
    assert second["track_id"] == 2
    assert second["bbox"] == _bbox_1000_to_pixels([350, 550, 950, 925], 800, 500)
    assert second["violation_detected"] is False
    assert second["license_plate"] == "B 9999 XYZ"
    assert second["analysis_source"] == "Gemini Full-Image"


def test_watermark_sanitization_and_offline_empty_yolo_behavior(tmp_path):
    assert _sanitize_plate_text("www.antaranews.com") == "TIDAK_TERBACA"
    assert _sanitize_plate_text("ANTARA FOTO") == "TIDAK_TERBACA"
    assert _sanitize_plate_text("BM 6116 OPP") == "BM 6116 OPP"

    db = DatabaseManager(db_path=tmp_path / "offline.db")
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # Saat Gemini offline (_simulated=True) dan YOLO menemukan 0 kendaraan, jangan ciptakan kendaraan palsu
    pipe = TrafficMonitoringPipeline(
        detector=_StubDetector([]),
        ocr=_StubOCR(),
        gemini=GeminiTrafficClient(api_key=""),
        database=db,
        async_ai=False,
    )
    _, inspections, summary = pipe.process_static_image(img)
    assert inspections == []
    assert summary["total_vehicles"] == 0


def test_pipeline_rejects_watermark_returned_as_plate_by_gemini(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "wm.db")
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    gemini = GeminiTrafficClient(api_key="")
    gemini.analyze_traffic_scene = lambda _img: TrafficSceneAnalysis(
        scene_summary="Satu motor terlihat dengan watermark berita.",
        vehicles=[
            VehicleSceneResult(
                vehicle_type="motorcycle",
                vehicle_bbox_1000=[300, 200, 900, 700],
                rider_bbox_1000=[150, 260, 600, 620],
                plate_bbox_1000=[750, 550, 880, 900],
                license_plate="www.antaranews.com",
                is_helmet_present=True,
                helmet_worn_properly=True,
                helmet_type="HALF_FACE",
                jacket_present=True,
                violation_detected=False,
                severity_level="NONE",
                confidence_score=0.88,
                officer_notes="Tertib.",
                recommended_action="LEPASKAN",
            )
        ],
    )

    pipe = TrafficMonitoringPipeline(
        detector=_StubDetector([]),
        ocr=_StubOCR(),
        gemini=gemini,
        database=db,
        async_ai=False,
    )
    _, inspections, _ = pipe.process_static_image(img)
    assert len(inspections) == 1
    assert inspections[0]["license_plate"] == "TIDAK_TERBACA"
    assert inspections[0]["plate_bbox"] is None
