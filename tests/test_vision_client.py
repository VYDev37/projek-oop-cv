"""
Unit Test: tests/test_vision_client.py
Deskripsi: Pengujian komprehensif untuk modul vision_client.py:
           1. Konsep ABC & Inheritance: Memastikan VisionClient tidak dapat diinstansiasi langsung.
           2. Subclass Konkrit: MockVisionClient menghasilkan PhotoAnalysisResult valid.
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
    GPT4VisionClient,
    PhotoAnalysisResult,
    VisionAPIException,
    ImageAnalysisService,
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


class TestStrategyPattern:
    """Pengujian implementasi Strategy Pattern."""

    def test_strategy_pattern_execution_and_switching(self):
        """Memverifikasi penukaran strategi runtime pada ImageAnalysisService."""
        mock_strategy = MockVisionClient(model_name="mock-primary")
        gpt4_strategy = GPT4VisionClient(model_name="gpt-4o-secondary")

        service = ImageAnalysisService(strategy=mock_strategy)
        assert service.strategy == mock_strategy

        dummy_img = Image.new("RGB", (150, 150), color=(128, 128, 128))

        # Analisis dengan strategi awal
        res1 = service.analyze_image(dummy_img, image_name="kopi_pagi.jpg")
        assert "Mock" in res1.provider
        assert res1.kategori == "Makanan"

        # Tukar strategi secara dinamis saat runtime (OCP / DIP)
        service.set_strategy(gpt4_strategy)
        assert service.strategy == gpt4_strategy

        # Analisis dengan strategi baru
        res2 = service.analyze_image(dummy_img, image_name="kopi_pagi.jpg")
        assert "OpenAI" in res2.provider

    def test_strategy_type_enforcement(self):
        """Memastikan set_strategy menolak objek yang bukan turunan VisionClient."""
        service = ImageAnalysisService()
        with pytest.raises(TypeError):
            service.set_strategy("bukan_vision_client")

    def test_compare_providers(self):
        """Pengujian perbandingan 2 provider berdampingan."""
        service = ImageAnalysisService()
        client_a = MockVisionClient()
        client_b = GPT4VisionClient()

        dummy_img = Image.new("RGB", (100, 100), color=(50, 50, 200))
        comparison = service.compare_providers(dummy_img, client_a, client_b, image_name="mobil_sport.jpg")

        assert "provider_a" in comparison
        assert "provider_b" in comparison
        assert "shared_tags" in comparison
        assert "tag_agreement_rate" in comparison
        assert isinstance(comparison["tag_agreement_rate"], float)
