"""
Unit Test: tests/test_factory_and_database.py
Deskripsi: Pengujian tambahan:
           1. Factory Pattern: VisionClientFactory (create, register, kasus negatif).
           2. Lazy import: vision_client dapat diimpor tanpa OpenCV/NumPy.
           3. DatabaseManager: koneksi selalu ditutup, commit & rollback berjalan benar.
"""

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from vision_client import (
    VisionClient,
    VisionClientFactory,
    MockVisionClient,
    GeminiVisionClient,
    GPT4VisionClient,
    YOLOv8VisionClient,
    ImageAnalysisService,
    PhotoAnalysisResult,
)
from database import DatabaseManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestVisionClientFactory:
    """Pengujian Factory Pattern pada pembuatan provider Vision AI."""

    @pytest.mark.parametrize(
        "name, expected_cls",
        [
            ("mock", MockVisionClient),
            ("gemini", GeminiVisionClient),
            ("gpt4", GPT4VisionClient),
            ("yolo", YOLOv8VisionClient),
        ],
    )
    def test_create_returns_correct_subclass(self, name, expected_cls):
        client = VisionClientFactory.create(name)
        assert isinstance(client, expected_cls)
        assert isinstance(client, VisionClient)

    def test_create_is_case_insensitive(self):
        assert isinstance(VisionClientFactory.create("  MoCk "), MockVisionClient)

    def test_create_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError):
            VisionClientFactory.create("provider-tidak-ada")

    def test_register_rejects_non_vision_client(self):
        with pytest.raises(TypeError):
            VisionClientFactory.register("palsu", dict)

    def test_register_new_provider_without_changing_factory(self):
        class CustomClient(MockVisionClient):
            pass

        VisionClientFactory.register("custom-test", CustomClient)
        try:
            assert isinstance(VisionClientFactory.create("custom-test"), CustomClient)
            assert "custom-test" in VisionClientFactory.available_providers()
        finally:
            VisionClientFactory.__registry.pop("custom-test", None)

    def test_factory_product_works_with_strategy_service(self):
        service = ImageAnalysisService(strategy=VisionClientFactory.create("mock"))
        result = service.analyze_image(Image.new("RGB", (64, 64)), image_name="kopi.jpg")
        assert isinstance(result, PhotoAnalysisResult)


def test_vision_client_import_does_not_require_cv2_or_numpy():
    """Mock/Gemini/GPT-4 harus bisa dipakai tanpa OpenCV & NumPy (lazy import)."""
    code = (
        "import sys; sys.modules['cv2'] = None; sys.modules['numpy'] = None; "
        "import vision_client; print('OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


class TestDatabaseConnection:
    """Pengujian pengelolaan koneksi SQLite pada DatabaseManager."""

    def test_log_and_statistics(self, tmp_path):
        db = DatabaseManager(db_path=tmp_path / "uji.db")
        row_id = db.log_traffic_vehicle(track_id=1, vehicle_type="car", confidence=0.9)
        assert row_id == 1
        stats = db.get_system_statistics()
        assert stats["total_vehicles"] == 1

    def test_connection_is_closed_after_use(self, tmp_path):
        db = DatabaseManager(db_path=tmp_path / "uji.db")
        with db._connection() as conn:
            pass
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

    def test_rollback_on_error(self, tmp_path):
        db = DatabaseManager(db_path=tmp_path / "uji.db")
        with pytest.raises(RuntimeError):
            with db._connection() as conn:
                conn.execute(
                    "INSERT INTO traffic_logs (timestamp, track_id, vehicle_type) VALUES ('t', 1, 'car')"
                )
                raise RuntimeError("gagal")
        assert db.get_system_statistics()["total_vehicles"] == 0
