"""
Unit Test: tests/test_moderator_and_exporter.py
Deskripsi: Pengujian komprehensif untuk ContentModerator dan CollectionCSVExporter (SOLID: SRP):
           1. ContentModerator: cek is_safe, filter unsafe, laporan moderasi
           2. CollectionCSVExporter: ekspor data koleksi ke CSV (SOLID: SRP)
"""

import csv
import pytest
from pathlib import Path

from vision_client import PhotoAnalysisResult
from photo_collection import PhotoCollection, ContentModerator, CollectionCSVExporter


@pytest.fixture
def mixed_collection():
    """Koleksi berisi foto aman dan foto tidak aman (unsafe)."""
    p_safe1 = PhotoAnalysisResult(
        caption="Pemandangan danau tenang",
        tags=["danau", "air", "alam"],
        objects=["pohon"],
        kategori="Alam",
        is_safe=True,
        colors=["Biru", "Hijau"],
    )
    p_safe2 = PhotoAnalysisResult(
        caption="Piring pasta carbonara",
        tags=["pasta", "makanan", "kuliner"],
        objects=["piring", "garpu"],
        kategori="Makanan",
        is_safe=True,
        colors=["Kuning", "Putih"],
    )
    p_unsafe = PhotoAnalysisResult(
        caption="Konten kekerasan senjata terlarang",
        tags=["unsafe", "senjata", "bahaya"],
        objects=["senjata"],
        kategori="Sensitif",
        is_safe=False,
        colors=["Merah", "Hitam"],
    )
    return PhotoCollection([p_safe1, p_safe2, p_unsafe])


class TestContentModerator:
    """Pengujian untuk ContentModerator."""

    def test_filter_safe_and_unsafe_photos(self, mixed_collection):
        """Memverifikasi pemisahan foto aman dan foto tidak aman."""
        moderator = ContentModerator()

        safe_photos = moderator.filter_safe_photos(mixed_collection)
        assert len(safe_photos) == 2
        for p in safe_photos:
            assert p.is_safe is True

        unsafe_photos = moderator.filter_unsafe_photos(mixed_collection)
        assert len(unsafe_photos) == 1
        assert unsafe_photos[0].is_safe is False
        assert "senjata" in unsafe_photos[0].tags

    def test_generate_moderation_report(self, mixed_collection):
        """Memverifikasi statistik pada laporan moderasi."""
        moderator = ContentModerator()

        report = moderator.generate_moderation_report(mixed_collection)
        assert report["total_photos"] == 3
        assert report["safe_count"] == 2
        assert report["unsafe_count"] == 1
        # 2 / 3 * 100 = 66.7%
        assert report["safe_rate_percentage"] == 66.7
        assert "PERHATIAN" in report["status"]


class TestCollectionCSVExporter:
    """Pengujian untuk CollectionCSVExporter (SOLID: SRP)."""

    def test_export_to_csv_integrity(self, mixed_collection, tmp_path):
        """Memverifikasi berkas CSV yang diekspor berisi header dan baris data yang valid."""
        exporter = CollectionCSVExporter()
        out_csv = tmp_path / "test_koleksi_output.csv"

        saved_path = exporter.export_to_csv(mixed_collection, out_csv)
        assert Path(saved_path).exists()

        # Baca kembali dan validasi baris
        with open(saved_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert "caption" in reader.fieldnames
        assert "tags" in reader.fieldnames
        assert "kategori" in reader.fieldnames
        assert "is_safe" in reader.fieldnames

        assert rows[0]["kategori"] == "Alam"
        assert rows[1]["kategori"] == "Makanan"
        assert rows[2]["is_safe"] == "TIDAK (UNSAFE)"
