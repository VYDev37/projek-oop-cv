import unittest
from unittest.mock import MagicMock
import numpy as np
import tempfile
from pathlib import Path

from detector import TrafficDetector, TrackedObject, ViolationCandidate
from pipeline import TrafficMonitoringPipeline
from database import DatabaseManager
from gemini_client import RiderSafetyValidation


class TestStaticInspection(unittest.TestCase):
    """Pengujian alur deteksi citra statis dan pipeline inspeksi ETLE."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_traffic.db"
        self.db_manager = DatabaseManager(db_path=str(self.db_path))

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_detect_static_image_with_mocked_yolo(self):
        """Memverifikasi ekstraksi kendaraan, person, dan kandidat inspeksi statis."""
        detector = TrafficDetector()

        # Mock predict method
        mock_box = MagicMock()
        mock_box.xyxy.cpu().numpy.return_value = np.array([
            [50, 100, 250, 400],  # motorcycle
            [80, 50, 200, 250],   # person (rider)
            [300, 200, 600, 450], # car
        ])
        mock_box.cls.cpu().numpy.return_value = np.array([3, 0, 2])  # 3: motorcycle, 0: person, 2: car
        mock_box.conf.cpu().numpy.return_value = np.array([0.9, 0.85, 0.95])

        mock_result = MagicMock()
        mock_result.boxes = mock_box
        detector.model.predict = MagicMock(return_value=[mock_result])

        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        vehicles, persons, candidates = detector.detect_static_image(dummy_image)

        self.assertEqual(len(vehicles), 2)
        self.assertEqual(len(persons), 1)
        self.assertEqual(len(candidates), 2)

        # Cek tipe kendaraan kandidat
        types = [c.vehicle_type for c in candidates]
        self.assertIn("motorcycle", types)
        self.assertIn("car", types)

    def test_process_static_image_pipeline(self):
        """Memverifikasi pemrosesan gambar menyeluruh pada pipeline dengan Dependency Injection."""
        mock_ocr = MagicMock()
        mock_ocr.read_license_plate.return_value = ("B 1234 ABC", 0.95)

        mock_gemini = MagicMock()
        mock_safety_motor = RiderSafetyValidation(
            is_helmet_present=False,
            helmet_worn_properly=False,
            helmet_type="TIDAK_ADA",
            jacket_present=True,
            other_attributes=[],
            violation_detected=True,
            violation_reasons=["Pengendara tidak memakai helm"],
            severity_level="HIGH",
            confidence_score=0.98,
            officer_notes="Pelanggaran helm terkonfirmasi.",
            recommended_action="TILANG",
        )
        mock_gemini.validate_rider_safety.return_value = mock_safety_motor

        mock_detector = MagicMock()
        dummy_candidate = ViolationCandidate(
            track_id=1,
            vehicle_type="motorcycle",
            vehicle_bbox=(50, 100, 250, 400),
            rider_bbox=(80, 50, 200, 250),
            rider_crop=np.zeros((100, 100, 3), dtype=np.uint8),
            plate_crop=np.zeros((40, 100, 3), dtype=np.uint8),
            rider_crop_path=None,
            plate_crop_path=None,
            timestamp="2026-10-11 20:00:00",
        )
        mock_detector.detect_static_image.return_value = (
            [MagicMock()],
            [MagicMock()],
            [dummy_candidate],
        )
        mock_detector.visualizer.draw_inspected_static_objects.return_value = np.zeros(
            (480, 640, 3), dtype=np.uint8
        )

        test_pipeline = TrafficMonitoringPipeline(
            detector=mock_detector,
            ocr=mock_ocr,
            gemini=mock_gemini,
            database=self.db_manager,
        )

        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated_img, inspections, summary = test_pipeline.process_static_image(dummy_image)

        self.assertIsInstance(annotated_img, np.ndarray)
        self.assertEqual(len(inspections), 1)
        self.assertEqual(summary["total_vehicles"], 1)
        self.assertEqual(summary["total_violations"], 1)
        self.assertEqual(summary["compliance_rate"], 0.0)

        # Verifikasi data tersimpan di database
        df_violations = self.db_manager.get_violations_dataframe(limit=10)
        self.assertEqual(len(df_violations), 1)
        self.assertEqual(df_violations.iloc[0]["license_plate"], "B 1234 ABC")
        self.assertEqual(df_violations.iloc[0]["violation_type"], "TANPA_HELM")

        test_pipeline.shutdown()


if __name__ == "__main__":
    unittest.main()
