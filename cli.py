"""
Module: cli.py
Deskripsi: Antarmuka baris perintah (CLI) untuk Sistem AI Image Analysis & Tagging Otomatis [I2].
           Menampilkan caption & top tags per foto (CLI),
           serta menyediakan demonstrasi interaktif untuk seluruh fitur sistem.
"""

import argparse
import sys
from pathlib import Path

from config import SAMPLES_DIR, EXPORTS_DIR, GEMINI_API_KEY
from vision_client import (
    GeminiVisionClient,
    GPT4VisionClient,
    MockVisionClient,
    PhotoAnalysisResult,
    ImageAnalysisService,
    VisionAPIException,
)
from photo_collection import (
    PhotoCollection,
    ContentModerator,
    CollectionCSVExporter,
)

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def print_banner():
    """Mencetak banner selamat datang di terminal."""
    print("=" * 70)
    print(" [I2] SISTEM AI IMAGE ANALYSIS & TAGGING OTOMATIS")
    print(" Berbasis OOP (ABC, Polymorphism, Strategy & Iterator Patterns, SOLID)")
    print("=" * 70)


def display_photo_result(result: PhotoAnalysisResult):
    """
    Tampilkan caption & top tags per foto di CLI.
    """
    print("\n" + "-" * 60)
    status_icon = "[AMAN / SAFE]" if result.is_safe else "[PERINGATAN: UNSAFE]"
    print(f"Berkas       : {result.image_path}")
    print(f"Provider     : {result.provider}")
    print(f"Waktu Proses : {result.processing_time_ms} ms")
    print(f"Kategori     : {result.kategori}")
    print(f"Moderasi     : {status_icon}")
    print(f"Caption      : {result.caption}")
    print(f"Tags         : {', '.join(result.tags)}")
    print(f"Objek Fisik  : {', '.join(result.objects)}")
    print(f"Palet Warna  : {', '.join(result.colors)}")
    if result.ocr_text:
        print(f"Teks OCR     : {result.ocr_text}")
    print("-" * 60)


def run_demo():
    """Menjalankan demonstrasi alur kerja menyeluruh."""
    print_banner()
    print("\n>>> Memulai Demonstrasi Otomatis Seluruh Fitur [I2]...\n")

    service = ImageAnalysisService()
    print("1. [Strategy Pattern] Strategi awal: MockVisionClient")

    if GEMINI_API_KEY:
        try:
            gemini_client = GeminiVisionClient()
            service.set_strategy(gemini_client)
            print("   -> GEMINI_API_KEY terdeteksi! Strategi aktif dialihkan ke GeminiVisionClient.")
        except Exception:
            print("   -> Menggunakan MockVisionClient untuk demonstrasi stabil.")
    else:
        print("   -> GEMINI_API_KEY belum disetel, menggunakan MockVisionClient.")

    sample_files = list(SAMPLES_DIR.glob("*.jpg"))
    if not sample_files:
        print("[WARNING] File sampel belum ada. Menjalankan generator otomatis...")
        from generate_samples import generate_all_samples
        generate_all_samples()
        sample_files = list(SAMPLES_DIR.glob("*.jpg"))

    collection = PhotoCollection()
    moderator = ContentModerator()

    print(f"\n2. [Batch Analysis] Menganalisis {len(sample_files)} gambar sampel sekaligus:", flush=True)
    for idx, f in enumerate(sample_files, start=1):
        print(f"\n   [{idx}/{len(sample_files)}] Memproses {f.name} via {service.strategy.__class__.__name__}...", flush=True)
        try:
            res = service.analyze_image(f)
            collection.tambah(res)
            display_photo_result(res)
        except VisionAPIException as e:
            print(f"[ERROR] Gagal menganalisis {f.name}: {e}", flush=True)

    print("\n3. [Iterator Pattern] Mengiterasi koleksi secara native:")
    for idx, photo in enumerate(collection, start=1):
        print(f"   [{idx}/{len(collection)}] {photo.kategori} -> {photo.caption[:45]}...")

    query_tag = "sunset"
    print(f"\n4. [Pencarian Tag] Mencari foto dengan tag '{query_tag}':")
    found_by_tag = collection.cari_by_tag(query_tag)
    print(f"   Ditemukan {len(found_by_tag)} foto.")
    for p in found_by_tag:
        print(f"   - {p.caption} (Tags: {p.tags})")

    query_kat = "Makanan"
    print(f"\n5. [Filter Kategori] Memfilter foto kategori '{query_kat}':")
    found_by_kat = collection.filter_kategori(query_kat)
    print(f"   Ditemukan {len(found_by_kat)} foto.")
    for p in found_by_kat:
        print(f"   - {p.caption}")

    print("\n6. [Statistik Counter] 5 Tag Paling Sering Muncul:")
    popular_tags = collection.get_popular_tags(top_n=5)
    for rank, (tag, count) in enumerate(popular_tags, start=1):
        print(f"   #{rank} {tag} : {count} kali")

    print("\n7. [Content Moderator] Laporan Keamanan Koleksi:")
    report = moderator.generate_moderation_report(collection)
    print(f"   Total Foto  : {report['total_photos']}")
    print(f"   Foto Aman   : {report['safe_count']}")
    print(f"   Foto Unsafe : {report['unsafe_count']}")
    print(f"   Status      : {report['status']}")

    print("\n8. [Komparasi 2 Provider] Membandingkan Gemini vs Mock / GPT-4o:")
    mock_client = MockVisionClient()
    gpt4_client = GPT4VisionClient()
    comp_result = service.compare_providers(sample_files[0], mock_client, gpt4_client)
    print(f"   Provider A ({comp_result['provider_a']['name']}) Caption: {comp_result['provider_a']['caption']}")
    print(f"   Provider B ({comp_result['provider_b']['name']}) Caption: {comp_result['provider_b']['caption']}")
    print(f"   Shared Tags: {comp_result['shared_tags']}")
    print(f"   Tingkat Keselarasan Tag: {comp_result['tag_agreement_rate']}%")

    print("\n9. [SOLID: SRP] Mengekspor Koleksi ke CSV:")
    exporter = CollectionCSVExporter()
    out_csv = EXPORTS_DIR / "koleksi_demo.csv"
    saved_path = exporter.export_to_csv(collection, out_csv)
    print(f"   [SUKSES] Berkas CSV berhasil disimpan di: {saved_path}")

    print("\n" + "=" * 70)
    print(" [SELESAI] DEMONSTRASI 20 FITUR SELESAI DENGAN SUKSES!")
    print("=" * 70 + "\n")


def main():
    """Fungsi utama titik masuk CLI."""
    parser = argparse.ArgumentParser(
        description="Sistem AI Image Analysis & Tagging Otomatis [I2] - CLI Runner"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Jalankan demonstrasi otomatis lengkap",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Jalur berkas foto yang ingin dianalisis",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["gemini", "gpt4", "mock"],
        default="mock",
        help="Pilihan provider Vision AI (default: mock)",
    )
    parser.add_argument(
        "--compare",
        type=str,
        help="Bandingkan 2 provider pada satu berkas citra",
    )

    args = parser.parse_args()

    if args.demo or len(sys.argv) == 1:
        run_demo()
        return

    if args.provider == "gemini":
        client = GeminiVisionClient()
    elif args.provider == "gpt4":
        client = GPT4VisionClient()
    else:
        client = MockVisionClient()

    service = ImageAnalysisService(strategy=client)

    if args.compare:
        print_banner()
        target = Path(args.compare)
        if not target.exists():
            print(f"[ERROR] Berkas tidak ditemukan: {target}")
            return
        p1 = GeminiVisionClient() if GEMINI_API_KEY else MockVisionClient()
        p2 = GPT4VisionClient()
        comp = service.compare_providers(target, p1, p2)
        print(f"\nKomparasi Citra: {target.name}")
        print(f"[{comp['provider_a']['name']}]: {comp['provider_a']['caption']}")
        print(f"[{comp['provider_b']['name']}]: {comp['provider_b']['caption']}")
        print(f"Shared Tags: {comp['shared_tags']} (Agreement: {comp['tag_agreement_rate']}%)")
        return

    if args.file:
        target = Path(args.file)
        if not target.exists():
            print(f"[ERROR] Berkas tidak ditemukan: {target}")
            return
        res = service.analyze_image(target)
        display_photo_result(res)


if __name__ == "__main__":
    main()
