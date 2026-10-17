"""
Module: photo_collection.py
Deskripsi: Pengelolaan koleksi foto, iterasi, filter, statistik, moderasi konten, dan ekspor data CSV.
"""

import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from PIL import Image

from vision_client import PhotoAnalysisResult, VisionClient, VisionAPIException

logger = logging.getLogger(__name__)


class PhotoCollectionIterator(Iterator[PhotoAnalysisResult]):
    """
    Iterator konkrit yang mengimplementasikan protokol Iterator Pattern (__iter__ & __next__).
    Memungkinkan iterasi elemen PhotoCollection secara berurutan tanpa mengekspos
    representasi struktur data internalnya.
    """

    def __init__(self, photos: List[PhotoAnalysisResult]):
        self._photos = photos
        self._index = 0

    def __iter__(self) -> "PhotoCollectionIterator":
        return self

    def __next__(self) -> PhotoAnalysisResult:
        if self._index < len(self._photos):
            photo = self._photos[self._index]
            self._index += 1
            return photo
        raise StopIteration


class PhotoCollection:
    """
    Aggregate class yang menyimpan dan mengelola sekumpulan hasil analisis citra PhotoAnalysisResult.
    Mendukung protokol Iterable, pencarian tag, filter kategori, statistik Counter, dan batch analysis.
    """

    def __init__(self, initial_photos: Optional[List[PhotoAnalysisResult]] = None):
        self._photos: List[PhotoAnalysisResult] = initial_photos.copy() if initial_photos else []

    def tambah(self, result: PhotoAnalysisResult) -> None:
        """Menambahkan objek PhotoAnalysisResult ke dalam koleksi."""
        is_valid_result = isinstance(result, PhotoAnalysisResult) or (
            getattr(result.__class__, "__name__", "") == "PhotoAnalysisResult"
        )
        if not is_valid_result:
            raise TypeError("Objek yang ditambahkan harus berupa instansiasi PhotoAnalysisResult.")
        self._photos.append(result)
        logger.info("Foto berhasil ditambahkan ke koleksi. Total koleksi: %d", len(self._photos))

    def cari_by_tag(self, tag: str) -> List[PhotoAnalysisResult]:
        """
        Mencari dan mengembalikan semua foto yang memiliki tag tertentu (case-insensitive).
        """
        clean_tag = tag.strip().lower()
        return [
            p for p in self._photos
            if any(clean_tag == t.lower() for t in p.tags)
        ]

    def filter_kategori(self, kategori: str) -> List[PhotoAnalysisResult]:
        """
        Memfilter dan mengembalikan semua foto berdasarkan nama kategori (case-insensitive).
        """
        clean_kat = kategori.strip().lower()
        return [
            p for p in self._photos
            if p.kategori.strip().lower() == clean_kat
        ]

    def __iter__(self) -> PhotoCollectionIterator:
        """Mengembalikan objek iterator konkrit PhotoCollectionIterator."""
        return PhotoCollectionIterator(self._photos)

    def __len__(self) -> int:
        """Mengembalikan jumlah foto dalam koleksi."""
        return len(self._photos)

    def __getitem__(self, index: int) -> PhotoAnalysisResult:
        """Memungkinkan akses indeks langsung photo_collection[i]."""
        return self._photos[index]

    def get_popular_tags(self, top_n: int = 5) -> List[Tuple[str, int]]:
        """
        Menghitung frekuensi seluruh tag di dalam koleksi menggunakan collections.Counter
        dan mengembalikan daftar tuple (tag, jumlah) untuk top-N tag terbanyak.
        """
        all_tags = [t.lower() for p in self._photos for t in p.tags]
        counter = Counter(all_tags)
        return counter.most_common(top_n)

    def get_kategori_distribution(self) -> Dict[str, int]:
        """Menghitung distribusi jumlah foto per kategori."""
        categories = [p.kategori for p in self._photos]
        return dict(Counter(categories))

    def batch_analyze(
        self,
        image_inputs: List[Union[str, Path, bytes, Image.Image]],
        client: VisionClient,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[PhotoAnalysisResult]:
        """
        Memproses kumpulan foto secara berurutan, menangani error per-foto secara graceful,
        dan otomatis menambahkan foto yang berhasil dianalisis ke dalam koleksi.
        """
        batch_results: List[PhotoAnalysisResult] = []
        total = len(image_inputs)

        for idx, img_input in enumerate(image_inputs, start=1):
            name_hint = f"Foto #{idx}"
            if isinstance(img_input, (str, Path)):
                name_hint = Path(img_input).name

            if progress_callback:
                progress_callback(idx, total, f"Memproses {name_hint}...")

            try:
                result = client.analyze(img_input, image_name=name_hint)
                self.tambah(result)
                batch_results.append(result)
            except VisionAPIException as vae:
                logger.error("Gagal menganalisis %s: %s", name_hint, vae)
            except Exception as e:
                logger.error("Terjadi error tak terduga pada %s: %s", name_hint, e)

        return batch_results

    def get_all_photos(self) -> List[PhotoAnalysisResult]:
        """Mengembalikan salinan list seluruh foto dalam koleksi."""
        return self._photos.copy()

    def clear(self) -> None:
        """Mengosongkan koleksi."""
        self._photos.clear()


class ContentModerator:
    """
    Kelas bertanggung jawab khusus untuk moderasi konten (memenuhi prinsip Single Responsibility).
    Memeriksa kepatuhan keamanan konten foto, memfilter foto unsafe, dan menyusun laporan moderasi.
    """

    def is_safe(self, result: PhotoAnalysisResult) -> bool:
        """Memeriksa apakah satu hasil analisis foto berstatus aman (is_safe == True)."""
        return bool(result.is_safe)

    def filter_safe_photos(self, photos: Union[PhotoCollection, List[PhotoAnalysisResult]]) -> List[PhotoAnalysisResult]:
        """Memfilter dan mengembalikan hanya foto-foto yang aman."""
        return [p for p in photos if self.is_safe(p)]

    def filter_unsafe_photos(self, photos: Union[PhotoCollection, List[PhotoAnalysisResult]]) -> List[PhotoAnalysisResult]:
        """Memfilter dan mengembalikan hanya foto-foto yang tidak aman (unsafe)."""
        return [p for p in photos if not self.is_safe(p)]

    def generate_moderation_report(
        self, photos: Union[PhotoCollection, List[PhotoAnalysisResult]]
    ) -> Dict[str, Any]:
        """
        Menyusun rekapitulasi laporan moderasi mencakup total foto,
        jumlah foto aman, jumlah foto tidak aman, dan persentase kepatuhan keamanan.
        """
        photo_list = list(photos)
        total = len(photo_list)
        safe_photos = self.filter_safe_photos(photo_list)
        unsafe_photos = self.filter_unsafe_photos(photo_list)

        safe_count = len(safe_photos)
        unsafe_count = len(unsafe_photos)
        compliance_rate = round((safe_count / total * 100), 1) if total > 0 else 100.0

        return {
            "total_photos": total,
            "safe_count": safe_count,
            "unsafe_count": unsafe_count,
            "safe_rate_percentage": compliance_rate,
            "status": "BERSIH" if unsafe_count == 0 else f"PERHATIAN: {unsafe_count} foto melanggar panduan",
        }


class CollectionCSVExporter:
    """
    Kelas khusus yang menangani penulisan data koleksi ke format CSV.
    Dibuat terpisah dari PhotoCollection untuk mematuhi Single Responsibility Principle (SRP):
    PhotoCollection hanya mengelola data koleksi, sedangkan CollectionCSVExporter menangani format dan I/O ekspor.
    """

    def export_to_csv(
        self,
        collection: Union[PhotoCollection, List[PhotoAnalysisResult]],
        output_filepath: Union[str, Path],
    ) -> Path:
        """
        Mengekspor seluruh data hasil analisis dalam koleksi ke berkas CSV.
        Mengembalikan objek Path berkas keluaran.
        """
        dest_path = Path(output_filepath)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "timestamp",
            "provider",
            "kategori",
            "is_safe",
            "caption",
            "tags",
            "objects",
            "colors",
            "ocr_text",
            "image_path",
            "annotated_image_path",
            "processing_time_ms",
        ]

        with open(dest_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for p in collection:
                writer.writerow({
                    "timestamp": p.timestamp,
                    "provider": p.provider,
                    "kategori": p.kategori,
                    "is_safe": "YA (SAFE)" if p.is_safe else "TIDAK (UNSAFE)",
                    "caption": p.caption,
                    "tags": ", ".join(p.tags),
                    "objects": ", ".join(p.objects),
                    "colors": ", ".join(p.colors),
                    "ocr_text": p.ocr_text or "-",
                    "image_path": p.image_path or "-",
                    "annotated_image_path": getattr(p, "annotated_image_path", "") or "-",
                    "processing_time_ms": p.processing_time_ms,
                })

        logger.info("Berhasil mengekspor koleksi foto ke CSV: %s", dest_path)
        return dest_path
