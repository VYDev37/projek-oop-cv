"""
Unit Test: tests/test_photo_collection.py
Deskripsi: Pengujian komprehensif untuk modul photo_collection.py:
           1. PhotoCollection: tambah(), cari_by_tag(), filter_kategori()
           2. Iterator Pattern: PhotoCollectionIterator (__iter__ & __next__)
           3. Statistik Counter: get_popular_tags()
           4. Pemrosesan Batch: batch_analyze()
"""

import pytest
from PIL import Image

from vision_client import PhotoAnalysisResult, MockVisionClient
from photo_collection import PhotoCollection, PhotoCollectionIterator


@pytest.fixture
def sample_photos():
    """Fixture data pengujian sekumpulan hasil analisis foto."""
    p1 = PhotoAnalysisResult(
        caption="Pemandangan pantai senja yang indah",
        tags=["pantai", "sunset", "senja", "laut"],
        objects=["matahari", "ombak"],
        kategori="Alam",
        is_safe=True,
        colors=["Oranye", "Biru"],
        provider="Mock",
    )
    p2 = PhotoAnalysisResult(
        caption="Secangkir kopi espresso hangat",
        tags=["kopi", "cafe", "minuman", "santai"],
        objects=["cangkir", "meja"],
        kategori="Makanan",
        is_safe=True,
        colors=["Cokelat", "Hitam"],
        provider="Mock",
    )
    p3 = PhotoAnalysisResult(
        caption="Sunset di puncak bukit hijau",
        tags=["sunset", "bukit", "alam", "senja"],
        objects=["gunung", "pohon"],
        kategori="Alam",
        is_safe=True,
        colors=["Emas", "Hijau"],
        provider="Mock",
    )
    return [p1, p2, p3]


class TestPhotoCollection:
    """Pengujian fungsionalitas PhotoCollection dan Iterator Pattern."""

    def test_tambah_photo(self, sample_photos):
        """Memverifikasi penambahan foto ke dalam koleksi."""
        collection = PhotoCollection()
        assert len(collection) == 0

        collection.tambah(sample_photos[0])
        assert len(collection) == 1
        assert collection[0].caption == sample_photos[0].caption

        # Validasi type checking
        with pytest.raises(TypeError):
            collection.tambah("bukan_photo_analysis_result")

    def test_cari_by_tag(self, sample_photos):
        """Memverifikasi pencarian berdasarkan tag (case-insensitive)."""
        collection = PhotoCollection(sample_photos)

        # Cari tag 'sunset' yang ada di 2 foto
        results = collection.cari_by_tag("SUNSET")
        assert len(results) == 2
        for r in results:
            assert "sunset" in [t.lower() for t in r.tags]

        # Cari tag 'kopi' yang hanya ada di 1 foto
        results_kopi = collection.cari_by_tag("kopi")
        assert len(results_kopi) == 1
        assert results_kopi[0].kategori == "Makanan"

        # Tag yang tidak ada
        results_none = collection.cari_by_tag("astronomi")
        assert len(results_none) == 0

    def test_filter_kategori(self, sample_photos):
        """Memverifikasi pemfilteran berdasarkan kategori."""
        collection = PhotoCollection(sample_photos)

        alam_photos = collection.filter_kategori("alam")
        assert len(alam_photos) == 2
        for p in alam_photos:
            assert p.kategori == "Alam"

        makanan_photos = collection.filter_kategori("Makanan")
        assert len(makanan_photos) == 1
        assert makanan_photos[0].kategori == "Makanan"

    def test_iterator_pattern(self, sample_photos):
        """Memverifikasi implementasi murni Iterator Pattern."""
        collection = PhotoCollection(sample_photos)

        iterator = iter(collection)
        assert isinstance(iterator, PhotoCollectionIterator)

        # Verifikasi pengambilan berurutan
        first = next(iterator)
        assert first == sample_photos[0]

        second = next(iterator)
        assert second == sample_photos[1]

        third = next(iterator)
        assert third == sample_photos[2]

        # Memastikan StopIteration dilempar saat habis
        with pytest.raises(StopIteration):
            next(iterator)

        # Memastikan dapat di-loop menggunakan sintaks for in native
        iterated_items = [p for p in collection]
        assert len(iterated_items) == 3
        assert iterated_items == sample_photos

    def test_popular_tags_counter(self, sample_photos):
        """Memverifikasi statistik Counter tag terpopuler."""
        collection = PhotoCollection(sample_photos)

        # Di sample_photos:
        # 'sunset' muncul 2x
        # 'senja' muncul 2x
        # tag lainnya muncul 1x
        top_tags = collection.get_popular_tags(top_n=2)
        assert len(top_tags) == 2

        tag_names = [t[0] for t in top_tags]
        assert "sunset" in tag_names
        assert "senja" in tag_names

        # Frekuensi harus 2
        for tag, count in top_tags:
            assert count == 2

    def test_batch_analyze(self, tmp_path):
        """Memverifikasi pemrosesan batch foto sekaligus."""
        client = MockVisionClient()
        collection = PhotoCollection()

        # Buat 3 berkas citra uji coba
        img1 = Image.new("RGB", (50, 50), color="blue")
        path1 = tmp_path / "pantai_bali.jpg"
        img1.save(path1)

        img2 = Image.new("RGB", (50, 50), color="brown")
        path2 = tmp_path / "kopi_hitam.jpg"
        img2.save(path2)

        results = collection.batch_analyze([path1, path2], client=client)
        assert len(results) == 2
        assert len(collection) == 2
        assert collection[0].kategori == "Alam"
        assert collection[1].kategori == "Makanan"
