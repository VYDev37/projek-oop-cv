"""
Unit Test: tests/test_vision_client.py
Deskripsi: Pengujian komprehensif untuk modul vision_client.py:
           1. Konsep ABC & Inheritance: Memastikan VisionClient tidak dapat diinstansiasi langsung.
           2. Subclass Konkrit: MockVisionClient, YOLOv8VisionClient menghasilkan PhotoAnalysisResult valid.
           3. Custom Exception: VisionAPIException.
           4. Helper Encoding: _encode_image() menghasilkan base64 valid.
           5. Design Pattern: Strategy Pattern (ImageAnalysisService) dan komparasi provider.
"""

import pytest
from pathlib import Path
from PIL import Image

from vision_client import (
    VisionClient,
    MockVisionClient,
    YOLOv8VisionClient,
    GPT4VisionClient,
    PhotoAnalysisResult,
    VisionAPIException,
    ImageAnalysisService,
    GeminiVisionClient
)


class TestVisionClientCore:
    """Pengujian konsep OOP dasar pada hierarki VisionClient."""

    def test_abc_cannot_be_instantiated(self):
        """Memastikan kelas abstrak VisionClient tidak dapat diinstansiasi langsung."""
        with pytest.raises(TypeError):
            _ = VisionClient(model_name="analyze")  # type: ignore[abstract]

    def test_mock_vision_client_inheritance(self):
        """Memastikan MockVisionClient adalah instance dari VisionClient (Polymorphism)."""
        client = MockVisionClient()
        assert isinstance(client, VisionClient)
        assert client.model_name == "mock-vision-v1"

    def test_yolo_vision_client_inheritance(self):
        """Memastikan YOLOv8VisionClient adalah turunan polimorfik VisionClient."""
        client = YOLOv8VisionClient()
        assert isinstance(client, VisionClient)
        assert client.model_name == "yolov8n"

    def test_encode_image_base64_integrity(self, tmp_path):
        """Memverifikasi fungsi _encode_image menghasilkan string base64 yang valid."""
        client = MockVisionClient()

        # Buat gambar uji coba sederhana
        test_img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img_path = tmp_path / "test_red.jpg"
        test_img.save(img_path)

        # Uji dari Path
        b64_from_path = client._encode_image(img_path)
        assert isinstance(b64_from_path, str)
        assert len(b64_from_path) > 0

        # Uji dari PIL Image
        b64_from_pil = client._encode_image(test_img)
        assert isinstance(b64_from_pil, str)
        assert len(b64_from_pil) > 0

    def test_custom_exception_handling(self):
        """Memverifikasi struktur dan pesan pada VisionAPIException."""
        exc = VisionAPIException("Koneksi API gagal", provider="TestProvider", status_code=503)
        assert exc.provider == "TestProvider"
        assert exc.status_code == 503
        assert "[TestProvider] Koneksi API gagal" in str(exc)

    def test_mock_vision_client_analyze_safe(self):
        """Menguji pemanggilan analyze() pada MockVisionClient untuk citra aman."""
        client = MockVisionClient()
        dummy_img = Image.new("RGB", (200, 200), color=(0, 255, 0))

        result = client.analyze(dummy_img, image_name="pantai_liburan.jpg")
        assert isinstance(result, PhotoAnalysisResult)
        assert result.is_safe is True
        assert result.kategori == "Alam"
        assert "sunset" in result.tags or "alam" in result.tags
        assert len(result.colors) > 0

    def test_mock_vision_client_analyze_unsafe(self):
        """Menguji deteksi konten unsafe pada MockVisionClient saat citra mengandung kata sensitif."""
        client = MockVisionClient()
        dummy_img = Image.new("RGB", (200, 200), color=(0, 0, 0))

        result = client.analyze(dummy_img, image_name="hazard_unsafe_weapon.jpg")
        assert isinstance(result, PhotoAnalysisResult)
        assert result.is_safe is False
        assert "unsafe" in result.tags

    def test_yolo_vision_client_analyze(self):
        """Menguji analisis gambar menggunakan YOLOv8VisionClient."""
        client = YOLOv8VisionClient()
        dummy_img = Image.new("RGB", (150, 150), color=(100, 150, 200))

        result = client.analyze(dummy_img, image_name="mobil_uji.jpg")
        assert isinstance(result, PhotoAnalysisResult)
        assert "YOLOv8" in result.provider
        assert result.caption != ""
        assert isinstance(result.tags, list)
        assert isinstance(result.boxes, list)
        assert Path(result.annotated_image_path).exists()


class TestStrategyPattern:
    """Pengujian implementasi Strategy Pattern."""

    def test_strategy_pattern_execution_and_switching(self):
        """Memverifikasi penukaran strategi runtime pada ImageAnalysisService."""
        mock_strategy = MockVisionClient(model_name="mock-primary")
        yolo_strategy = YOLOv8VisionClient(model_name="yolov8n-secondary")

        service = ImageAnalysisService(strategy=mock_strategy)
        assert service.strategy == mock_strategy

        dummy_img = Image.new("RGB", (150, 150), color=(128, 128, 128))

        # Analisis dengan strategi awal
        res1 = service.analyze_image(dummy_img, image_name="kopi_pagi.jpg")
        assert "Mock" in res1.provider
        assert res1.kategori == "Makanan"

        # Tukar strategi secara dinamis saat runtime (OCP / DIP)
        service.set_strategy(yolo_strategy)
        assert service.strategy == yolo_strategy

        # Analisis dengan strategi baru (YOLO)
        res2 = service.analyze_image(dummy_img, image_name="kopi_pagi.jpg")
        assert "YOLOv8" in res2.provider

    def test_strategy_type_enforcement(self):
        """Memastikan set_strategy menolak objek yang bukan turunan VisionClient."""
        service = ImageAnalysisService()
        with pytest.raises(TypeError):
            service.set_strategy("bukan_vision_client")

    def test_compare_providers(self):
        """Pengujian perbandingan 2 provider berdampingan."""
        service = ImageAnalysisService()
        client_a = MockVisionClient()
        client_b = YOLOv8VisionClient()

        dummy_img = Image.new("RGB", (100, 100), color=(50, 50, 200))
        comparison = service.compare_providers(dummy_img, client_a, client_b, image_name="mobil_sport.jpg")

        assert "provider_a" in comparison
        assert "provider_b" in comparison
        assert "shared_tags" in comparison
        assert "tag_agreement_rate" in comparison
        assert isinstance(comparison["tag_agreement_rate"], float)


class TestVisionClientNegativeCases:
    """Pengujian skenario kegagalan (negative test cases) untuk pembuktian Exception Handling."""

    def test_encode_image_nonexistent_file_raises_exception(self):
        """Memverifikasi bahwa _encode_image melempar VisionAPIException saat berkas tidak ada."""
        client = MockVisionClient()
        with pytest.raises(VisionAPIException) as exc_info:
            client._encode_image("berkas_khayalan_yang_pasti_tidak_ada.jpg")
        assert "tidak ditemukan" in str(exc_info.value).lower() or "gagal" in str(exc_info.value).lower()

    def test_encode_image_invalid_type_raises_exception(self):
        """Memverifikasi bahwa _encode_image melempar VisionAPIException saat diberikan tipe data yang tidak valid."""
        client = MockVisionClient()
        with pytest.raises(VisionAPIException) as exc_info:
            client._encode_image(123456)
        assert "tidak didukung" in str(exc_info.value).lower() or "gagal" in str(exc_info.value).lower()

    def test_strategy_pattern_rejects_invalid_type(self):
        """Memverifikasi bahwa set_strategy melempar TypeError jika diberikan objek bukan turunan VisionClient."""
        service = ImageAnalysisService()
        with pytest.raises(TypeError):
            service.set_strategy("bukan_strategi_valid")
        with pytest.raises(TypeError):
            service.set_strategy(None)


class TestGeminiVisionClient:
    """Pengujian untuk GeminiVisionClient, termasuk parsing JSON respons dan exception handling."""

    def test_gemini_vision_client_inheritance(self):
        """Memastikan GeminiVisionClient adalah turunan dari VisionClient."""
        client = GeminiVisionClient(api_key="dummy_key_for_test")
        assert isinstance(client, VisionClient)

    def test_gemini_vision_client_missing_client_raises_exception(self):
        """Memastikan VisionAPIException dilempar jika klien Gemini belum siap/tanpa api_key."""
        client = GeminiVisionClient(api_key=None)
        client.client = None
        dummy_img = Image.new("RGB", (50, 50), color="white")
        with pytest.raises(VisionAPIException) as exc_info:
            client.analyze(dummy_img)
        assert exc_info.value.status_code == 401

    def test_gemini_vision_client_valid_json_response(self):
        """Memverifikasi parsing respons JSON terstruktur yang valid dari Gemini."""
        client = GeminiVisionClient(api_key="mock_key")

        class MockResponse:
            text = (
                '{\n'
                '  "caption": "Pemandangan pegunungan hijau yang asri",\n'
                '  "tags": ["gunung", "alam", "sejuk"],\n'
                '  "objects": ["bukit", "pepohonan"],\n'
                '  "kategori": "Alam",\n'
                '  "is_safe": true,\n'
                '  "colors": ["Hijau", "Biru Langit"],\n'
                '  "ocr_text": ""\n'
                '}'
            )

        class MockModels:
            def generate_content(self, *args, **kwargs):
                return MockResponse()

        class MockGenAIClient:
            models = MockModels()

        client.client = MockGenAIClient()
        dummy_img = Image.new("RGB", (60, 60), color="green")
        result = client.analyze(dummy_img, image_name="gunung.jpg")

        assert isinstance(result, PhotoAnalysisResult)
        assert result.kategori == "Alam"
        assert result.is_safe is True
        assert "gunung" in result.tags
        assert "Google Gemini" in result.provider

    def test_gemini_vision_client_corrupt_json_raises_exception(self):
        """Memverifikasi bahwa respons JSON yang rusak/korup memicu VisionAPIException (Rubrik Item 4)."""
        client = GeminiVisionClient(api_key="mock_key")

        class MockCorruptResponse:
            text = "{ bad_json: ini_bukan_json_valid_dan_pasti_gagal_parse ..."

        class MockModels:
            def generate_content(self, *args, **kwargs):
                return MockCorruptResponse()

        class MockGenAIClient:
            models = MockModels()

        client.client = MockGenAIClient()
        dummy_img = Image.new("RGB", (60, 60), color="red")

        with pytest.raises(VisionAPIException) as exc_info:
            client.analyze(dummy_img, image_name="korup.jpg")

        assert exc_info.value.status_code == 422
        assert "gagal mem-parse json" in str(exc_info.value).lower()

    def test_gemini_vision_client_empty_response_raises_exception(self):
        """Memverifikasi bahwa respons teks kosong memicu VisionAPIException."""
        client = GeminiVisionClient(api_key="mock_key")

        class MockEmptyResponse:
            text = ""

        class MockModels:
            def generate_content(self, *args, **kwargs):
                return MockEmptyResponse()

        class MockGenAIClient:
            models = MockModels()

        client.client = MockGenAIClient()
        dummy_img = Image.new("RGB", (60, 60), color="blue")

        with pytest.raises(VisionAPIException) as exc_info:
            client.analyze(dummy_img)

        assert exc_info.value.status_code == 500
        assert "respons kosong" in str(exc_info.value).lower()


class TestGPT4VisionClient:
    """Pengujian untuk GPT4VisionClient (Rubrik Item 16)."""

    def test_gpt4_vision_client_inheritance(self):
        """Memastikan GPT4VisionClient adalah turunan dari VisionClient."""
        client = GPT4VisionClient(api_key=None)
        assert isinstance(client, VisionClient)

    def test_gpt4_vision_client_simulated_fallback(self):
        """Memverifikasi bahwa saat API key tidak disetel, GPT-4o menggunakan mode simulasi cerdas."""
        client = GPT4VisionClient(api_key=None)
        dummy_img = Image.new("RGB", (80, 80), color="yellow")
        result = client.analyze(dummy_img, image_name="tes_gpt4.jpg")

        assert isinstance(result, PhotoAnalysisResult)
        assert "OpenAI" in result.provider
        assert result.is_safe is True
        assert len(result.tags) > 0

    def test_gpt4_vision_client_api_error_raises_exception(self, monkeypatch):
        """Memverifikasi bahwa error status code dari OpenAI API melempar VisionAPIException."""
        client = GPT4VisionClient(api_key="sk-fake-openai-key")

        class MockFailedResponse:
            status_code = 401
            text = '{"error": {"message": "Incorrect API key provided"}}'

            def json(self):
                return {"error": "unauthorized"}

        import requests
        monkeypatch.setattr(requests, "post", lambda *args, **kwargs: MockFailedResponse())

        dummy_img = Image.new("RGB", (50, 50), color="black")
        with pytest.raises(VisionAPIException) as exc_info:
            client.analyze(dummy_img)

        assert exc_info.value.status_code == 401
        assert "openai api error" in str(exc_info.value).lower()


class TestPlateLocalizationAndVisualizer:
    """Pengujian untuk lokalisasi presisi plat nomor dan render visualizer."""

    def test_locate_and_read_plate_empty_safe(self):
        from ocr_engine import plate_ocr
        text, conf, g_box, crop = plate_ocr.locate_and_read_plate(None)
        assert text == "TIDAK_TERBACA"
        assert conf == 0.0
        assert g_box is None
        assert crop is None

    def test_detect_plate_candidate_contour_synthetic(self):
        import numpy as np
        import cv2
        from ocr_engine import plate_ocr

        # Buat gambar sintetis berlatar belakang gelap dengan lempeng persegi panjang putih (aspect ratio ~3:1)
        canvas = np.zeros((300, 400, 3), dtype=np.uint8)
        # Gambar plat di tengah: x=100..280 (lebar 180), y=150..210 (tinggi 60), aspect ratio = 3.0
        cv2.rectangle(canvas, (100, 150), (280, 210), (255, 255, 255), -1)
        # Tambahkan teks hitam di dalam plat untuk memicu gradien Sobel vertikal
        cv2.putText(canvas, "B 1234 ABC", (110, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        box = plate_ocr.detect_plate_candidate_contour(canvas)
        assert box is not None
        bx1, by1, bx2, by2 = box
        assert bx2 > bx1
        assert by2 > by1
        # Memastikan deteksi kontur berada di sekitar area plat
        assert 80 <= bx1 <= 120
        assert 130 <= by1 <= 170

    def test_visualizer_renders_plate_and_rider_highlights(self):
        import numpy as np
        from visualizer import TrafficVisualizer

        vis = TrafficVisualizer()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        inspections = [{
            "track_id": 1,
            "vehicle_type": "motorcycle",
            "bbox": (50, 100, 250, 400),
            "plate_bbox": (100, 340, 200, 380),
            "rider_bbox": (80, 100, 220, 250),
            "license_plate": "B 1234 ABC",
            "violation_detected": False,
            "helmet_status": "WORN",
        }]

        annotated = vis.draw_inspected_static_objects(dummy_frame, inspections)
        assert annotated is not None
        assert annotated.shape == dummy_frame.shape
        # Pastikan kanvas tidak lagi hitam kosong (terdapat piksel anotasi berwarna)
        assert np.count_nonzero(annotated) > 0
