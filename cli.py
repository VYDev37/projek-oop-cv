"""
Module: cli.py
Deskripsi: Antarmuka baris perintah (CLI) untuk Sistem AI Image Analysis & Tagging Otomatis [I2].
           Menampilkan caption, top tags, objek, plat nomor per foto (CLI),
           serta menyediakan demonstrasi interaktif untuk seluruh fitur sistem.
"""

import argparse
import sys
from pathlib import Path

from config import SAMPLES_DIR, EXPORTS_DIR, GEMINI_API_KEY
from vision_client import (
    VisionClient,
    YOLOv8VisionClient,
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
    print(" [I2] SISTEM COMPUTER VISION & AI IMAGE ANALYSIS / TRAFFIC MONITORING")
    print(" Berbasis OOP (ABC, Polymorphism, Strategy & Iterator Patterns, SOLID)")
    print("=" * 70)


def display_photo_result(result: PhotoAnalysisResult):
    """
    Tampilkan caption, top tags, objek, dan plat nomor per foto di CLI.
    """
    print("\n" + "-" * 60)
    status_icon = "[AMAN / SAFE]" if result.is_safe else "[PERINGATAN: UNSAFE / PELANGGARAN]"
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
        print(f"Plat OCR     : {result.ocr_text}")
    if result.annotated_image_path:
        print(f"Anotasi CV   : {result.annotated_image_path}")
    print("-" * 60)


def run_demo():
    """Menjalankan demonstrasi alur kerja menyeluruh."""
    print_banner()
    print("\n>>> Memulai Demonstrasi Otomatis Seluruh Fitur [I2]...\n")

    # 1. Strategy Pattern
    yolo_client = YOLOv8VisionClient()
    service = ImageAnalysisService(strategy=yolo_client)
    print(f"1. [Strategy Pattern] Strategi aktif awal: {service.strategy.__class__.__name__}")

    if GEMINI_API_KEY:
        try:
            gemini_client = GeminiVisionClient()
            print("   -> GEMINI_API_KEY terdeteksi dan siap digunakan sebagai strategi alternatif.")
        except Exception:
            pass

    sample_files = list(SAMPLES_DIR.glob("*.jpg"))
    if not sample_files:
        print("[WARNING] File sampel belum ada. Menjalankan generator otomatis...")
        from generate_samples import generate_all_samples
        generate_all_samples()
        sample_files = list(SAMPLES_DIR.glob("*.jpg"))

    collection = PhotoCollection()
    moderator = ContentModerator()

    # 2. Batch Analysis & Polimorfisme
    print(f"\n2. [Batch Analysis] Menganalisis {len(sample_files)} gambar sampel via YOLOv8 & Mock:")
    for idx, f in enumerate(sample_files, start=1):
        print(f"\n   [{idx}/{len(sample_files)}] Memproses {f.name} via {service.strategy.__class__.__name__}...", flush=True)
        try:
            res = service.analyze_image(f)
            collection.tambah(res)
            display_photo_result(res)
        except VisionAPIException as e:
            print(f"[ERROR] Gagal menganalisis {f.name}: {e}", flush=True)

    # 3. Iterator Pattern
    print("\n3. [Iterator Pattern] Mengiterasi koleksi secara native via for-loop:")
    for idx, photo in enumerate(collection, start=1):
        print(f"   [{idx}/{len(collection)}] {photo.kategori} -> {photo.caption[:50]}...")

    # 4. Pencarian Tag
    query_tag = "motorcycle"
    print(f"\n4. [Pencarian Tag] Mencari foto dengan tag '{query_tag}':")
    found_by_tag = collection.cari_by_tag(query_tag)
    if not found_by_tag:
        query_tag = "alam"
        found_by_tag = collection.cari_by_tag(query_tag)
    print(f"   Ditemukan {len(found_by_tag)} foto.")
    for p in found_by_tag:
        print(f"   - {p.caption} (Tags: {p.tags})")

    # 5. Filter Kategori
    query_kat = "Patuhi Aturan"
    print(f"\n5. [Filter Kategori] Memfilter foto kategori '{query_kat}':")
    found_by_kat = collection.filter_kategori(query_kat)
    if not found_by_kat and collection:
        query_kat = collection[0].kategori
        found_by_kat = collection.filter_kategori(query_kat)
    print(f"   Ditemukan {len(found_by_kat)} foto.")
    for p in found_by_kat:
        print(f"   - {p.caption}")

    # 6. Statistik Tag Terpopuler (Counter)
    print("\n6. [Statistik Counter] 5 Tag Paling Sering Muncul:")
    popular_tags = collection.get_popular_tags(top_n=5)
    for rank, (tag, count) in enumerate(popular_tags, start=1):
        print(f"   #{rank} {tag} : {count} kali")

    # 7. Content Moderator
    print("\n7. [Content Moderator] Laporan Kepatuhan / Keselamatan Koleksi:")
    report = moderator.generate_moderation_report(collection)
    print(f"   Total Foto  : {report['total_photos']}")
    print(f"   Foto Tertib : {report['safe_count']}")
    print(f"   Pelanggaran : {report['unsafe_count']}")
    print(f"   Kepatuhan   : {report['safe_rate_percentage']}%")
    print(f"   Status      : {report['status']}")

    # 8. Komparasi 2 Engine Berbeda
    print("\n8. [Komparasi 2 Engine] Membandingkan YOLOv8 vs Mock Vision:")
    mock_client = MockVisionClient()
    comp_result = service.compare_providers(sample_files[0], yolo_client, mock_client)
    print(f"   Engine A ({comp_result['provider_a']['name']}) Caption: {comp_result['provider_a']['caption']}")
    print(f"   Engine B ({comp_result['provider_b']['name']}) Caption: {comp_result['provider_b']['caption']}")
    print(f"   Shared Tags: {comp_result['shared_tags']}")
    print(f"   Tingkat Keselarasan Tag: {comp_result['tag_agreement_rate']}%")

    # 9. SOLID SRP: CSV Exporter
    print("\n9. [SOLID: SRP] Mengekspor Koleksi ke CSV:")
    exporter = CollectionCSVExporter()
    out_csv = EXPORTS_DIR / "koleksi_demo.csv"
    saved_path = exporter.export_to_csv(collection, out_csv)
    print(f"   [SUKSES] Berkas CSV berhasil disimpan di: {saved_path}")

    print("\n" + "=" * 70)
    print(" [SELESAI] DEMONSTRASI SELURUH FITUR SELESAI DENGAN SUKSES!")
    print("=" * 70 + "\n")


def run_video_cli(video_path: str):
    """Menjalankan pemrosesan video di terminal via OpenCV window."""
    import cv2
    from pipeline import TrafficMonitoringPipeline

    print_banner()
    print(f"\n>>> Memulai Pemantauan Video: {video_path}")
    print("Tekan tombol 'q' pada jendela video untuk berhenti.\n")

    pipeline = TrafficMonitoringPipeline()
    cap = cv2.VideoCapture(int(video_path) if video_path.isdigit() else video_path)

    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka video: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (960, 540))
        annotated = pipeline.process_frame(frame_resized)

        cv2.imshow("Traffic Monitoring CLI [I2]", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Pemantauan video selesai.")


def main():
    """Fungsi utama titik masuk CLI."""
    parser = argparse.ArgumentParser(
        description="Sistem Computer Vision & AI Image Analysis / Traffic Monitoring [I2] - CLI Runner"
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
        "--video",
        type=str,
        help="Jalur berkas video atau indeks kamera (misal: 0) untuk streaming video",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["yolo", "gemini", "mock", "gpt4"],
        default="yolo",
        help="Pilihan engine Vision AI (default: yolo)",
    )
    parser.add_argument(
        "--compare",
        type=str,
        help="Bandingkan 2 provider/engine pada satu berkas citra",
    )

    args = parser.parse_args()

    if args.video:
        run_video_cli(args.video)
        return

    if args.demo or len(sys.argv) == 1:
        run_demo()
        return

    if args.provider == "yolo":
        client = YOLOv8VisionClient()
    elif args.provider == "gemini":
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
        p1 = YOLOv8VisionClient()
        p2 = GeminiVisionClient() if GEMINI_API_KEY else MockVisionClient()
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
