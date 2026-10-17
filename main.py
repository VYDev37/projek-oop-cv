"""
Module: main.py
Deskripsi: Script eksekusi utama CLI untuk Sistem Cerdas Pemantauan Lalu Lintas & Keselamatan AI.
           Mendukung:
           1. Pemantauan video langsung (CCTV / Webcam / Demo Video) via OpenCV
           2. Menu interaktif terminal berbasis Object-Oriented Programming (OOP):
              - Analisis Foto Tunggal (YOLOv8 + EasyOCR + Moderasi)
              - Pemrosesan Batch
              - Pencarian Tag (cari_by_tag) & Filter Kategori
              - Statistik Tag Terpopuler via collections.Counter
              - Laporan Moderasi (ContentModerator)
              - Ekspor CSV (CollectionCSVExporter)
"""

import argparse
import sys
import time
import logging
from typing import Union, Set, Optional
from pathlib import Path

from config import (
    DEFAULT_YOLO_MODEL,
    COUNTING_LINE_Y_RATIO,
    SAMPLES_DIR,
    EXPORTS_DIR,
)
from database import db_manager
from pipeline import pipeline
from vision_client import (
    VisionClient,
    YOLOv8VisionClient,
    GeminiVisionClient,
    GPT4VisionClient,
    MockVisionClient,
    ImageAnalysisService,
    PhotoAnalysisResult,
)
from photo_collection import (
    PhotoCollection,
    PhotoCollectionIterator,
    ContentModerator,
    CollectionCSVExporter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrafficSystemMain")

# Global instances for CLI session
app_collection = PhotoCollection()
app_moderator = ContentModerator()
app_exporter = CollectionCSVExporter()
app_service = ImageAnalysisService(strategy=YOLOv8VisionClient())


def parse_arguments() -> argparse.Namespace:
    """Menguraikan argumen baris perintah."""
    parser = argparse.ArgumentParser(
        description="Sistem Cerdas Pemantauan Lalu Lintas & Keselamatan AI (CLI & Computer Vision)"
    )
    parser.add_argument(
        "--source",
        "--video",
        dest="source",
        type=str,
        default="",
        help="Sumber video: '0' untuk webcam, atau path berkas video (.mp4/.avi)",
    )
    parser.add_argument(
        "--file",
        "--image",
        dest="image",
        type=str,
        default="",
        help="Path berkas foto untuk langsung dianalisis via Vision AI",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default="",
        help="Path berkas foto untuk membandingkan 2 engine Vision AI berdampingan",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["yolo", "gemini", "mock", "gpt4"],
        default="",
        help="Pilihan engine Vision AI aktif (yolo / gemini / mock / gpt4)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Jalankan otomatis menggunakan video demo sampel bawaan",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Jalankan tanpa jendela GUI OpenCV (cocok untuk server / edge device)",
    )
    parser.add_argument(
        "--line",
        type=float,
        default=COUNTING_LINE_Y_RATIO,
        help="Posisi vertikal garis hitung (rasio 0.1 - 0.9)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Batas maksimum frame yang diproses (0 = tanpa batas)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        default=False,
        help="Ekspor otomatis laporan pelanggaran ke CSV dan Excel saat selesai",
    )
    return parser.parse_args()


def resolve_video_source(args: argparse.Namespace) -> Union[int, str]:
    """Menentukan sumber video yang valid."""
    if args.demo:
        demo_file = SAMPLES_DIR / "sample_traffic.mp4"
        if demo_file.exists():
            return str(demo_file)
        logger.warning("Berkas demo sample_traffic.mp4 tidak ditemukan di data/samples.")

    if not args.source:
        demo_file = SAMPLES_DIR / "sample_traffic.mp4"
        if demo_file.exists():
            return str(demo_file)
        return 0

    if args.source.isdigit():
        return int(args.source)

    return args.source


def run_video_stream(source: Union[int, str], line_ratio: float, headless: bool, max_frames: int = 0) -> None:
    """Menjalankan loop pemantauan aliran video realtime dengan OpenCV."""
    # Lazy import: modul berat (OpenCV, YOLO, OCR) hanya dimuat untuk mode video,
    # sehingga analisis foto (Mock/Gemini/GPT-4) tidak membutuhkannya.
    import cv2
    from pipeline import pipeline
           
    logger.info("=" * 70)
    logger.info("MEMULAI PEMANTAUAN LALU LINTAS COMPUTER VISION REAL-TIME")
    logger.info("Sumber Video   : %s", source)
    logger.info("Garis Hitung   : %s", line_ratio)
    logger.info("Mode Headless  : %s", headless)
    logger.info("=" * 70)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Gagal membuka sumber video: %s", source)
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    logger.info("Resolusi Input : %dx%d @ %.1f FPS", frame_width, frame_height, fps_in)
    logger.info("Tekan tombol 'q' pada jendela video untuk menghentikan pemantauan.\n")

    frame_count = 0
    start_time = time.time()
    last_reported_alerts: Set[int] = set()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_count += 1
            if max_frames > 0 and frame_count >= max_frames:
                logger.info("Mencapai batas maksimum frame (%d). Berhenti.", max_frames)
                break

            annotated_frame, alerts, meta = pipeline.process_frame(frame, line_ratio=line_ratio)

            for alert in alerts:
                if alert.id not in last_reported_alerts:
                    last_reported_alerts.add(alert.id)
                    logger.warning(
                        "[PELANGGARAN BARU] ID Track: %s | Jenis: %s | Plat: %s | Tindakan: %s",
                        alert.track_id,
                        alert.violation_type,
                        alert.license_plate,
                        alert.recommended_action,
                    )

            if not headless:
                cv2.imshow("Smart Traffic Monitoring System (Tekan 'q' untuk Keluar)", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Pengguna menekan tombol 'q'. Menghentikan stream...")
                    break

    except KeyboardInterrupt:
        logger.info("Pemantauan dihentikan oleh pengguna (Ctrl+C).")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        fps_avg = frame_count / elapsed if elapsed > 0 else 0
        logger.info("=" * 70)
        logger.info("RINGKASAN PEMANTAUAN LALU LINTAS")
        logger.info("Total Frame Diproses : %d frame", frame_count)
        logger.info("Rata-rata FPS        : %.2f FPS", fps_avg)
        logger.info("Total Kendaraan Dihitung : %d", pipeline.detector.total_counted)
        logger.info("Rincian per Jenis    : %s", pipeline.detector.vehicle_counts)
        logger.info("=" * 70)


def display_photo_result(result: PhotoAnalysisResult, top_n_tags: int = 5) -> None:
    """
    Menampilkan hasil analisis foto ke antarmuka terminal/CLI (Rubrik Item 10).
    Secara spesifik menampilkan caption naratif dan top-N tag teratas.
    """
    top_tags = result.tags[:top_n_tags] if result.tags else []
    status_str = "AMAN (SAFE)" if result.is_safe else "UNSAFE (PERLU MODERASI / BAHAYA)"

    print("\n" + "=" * 65)
    print(f"HASIL ANALISIS CITRA ({result.provider})")
    print(f"Status Moderasi : {status_str}")
    print(f"Kategori        : {result.kategori}")
    print(f"Caption Naratif : {result.caption}")
    print(f"Top Tags ({len(top_tags)})  : {', '.join(['#' + t for t in top_tags]) if top_tags else '-'}")
    print(f"Objek Fisik     : {', '.join(result.objects) if result.objects else '-'}")
    print(f"Palet Warna     : {', '.join(result.colors) if result.colors else '-'}")
    if result.ocr_text:
        print(f"Plat Nomor / OCR: {result.ocr_text}")
    print(f"Waktu Proses    : {result.processing_time_ms} ms")
    print("=" * 65)


def analyze_single_photo(image_path_str: str) -> Optional[PhotoAnalysisResult]:
    """Menganalisis satu foto menggunakan strategi Vision AI aktif."""
    path = Path(image_path_str)
    if not path.exists():
        print(f"❌ Berkas tidak ditemukan: {image_path_str}")
        return None

    print(f"\n🔍 Menganalisis '{path.name}' via {app_service.strategy.__class__.__name__}...")
    res = app_service.analyze_image(path)
    app_collection.tambah(res)
    display_photo_result(res, top_n_tags=5)
    return res


def run_batch_samples() -> None:
    """Menganalisis seluruh sampel gambar di data/samples secara batch."""
    samples = list(SAMPLES_DIR.glob("*.jpg"))
    if not samples:
        print("❌ Tidak ada berkas .jpg di data/samples.")
        return

    print(f"\n⚡ Memproses {len(samples)} foto sampel secara batch...")
    results = app_collection.batch_analyze(samples, client=app_service.strategy)
    print(f"✅ Berhasil memproses {len(results)} foto ke dalam koleksi.")


def display_interactive_menu() -> None:
    """Menampilkan menu CLI interaktif OOP."""
    while True:
        print("\n" + "=" * 65)
        print("🚦 SISTEM CERDAS PEMANTAUAN LALU LINTAS & KESELAMATAN AI (OOP CLI)")
        print(f"Engine Aktif: {app_service.strategy.__class__.__name__} | Koleksi: {len(app_collection)} foto")
        print("=" * 65)
        print("1. 📹 Jalankan Pemantauan Video Real-Time (OpenCV Window)")
        print("2. 🔍 Analisis & Deteksi Foto Tunggal")
        print("3. ⚡ Jalankan Analisis Batch Seluruh Sampel Foto")
        print("4. 🏷️ Cari Foto Berdasarkan Tag (PhotoCollection.cari_by_tag)")
        print("5. 📂 Filter Foto Berdasarkan Kategori (PhotoCollection.filter_kategori)")
        print("6. 📊 Tampilkan Tag Paling Populer (collections.Counter)")
        print("7. 🛡️ Tampilkan Laporan Moderasi & Kepatuhan (ContentModerator)")
        print("8. 📥 Ekspor Seluruh Koleksi ke CSV (CollectionCSVExporter)")
        print("9. 🎯 Ganti Engine Vision AI (Strategy Pattern)")
        print("0. 🚪 Keluar")
        print("=" * 65)

        choice = input("Pilih menu [0-9]: ").strip()

        if choice == "1":
            vid_choice = input("Pilih sumber video [1 = Demo Sampel, 2 = Webcam, 3 = Path Berkas]: ").strip()
            if vid_choice == "2":
                run_video_stream(0, line_ratio=COUNTING_LINE_Y_RATIO, headless=False)
            elif vid_choice == "3":
                custom_src = input("Masukkan path berkas video (.mp4/.avi): ").strip()
                run_video_stream(custom_src, line_ratio=COUNTING_LINE_Y_RATIO, headless=False)
            else:
                demo_path = SAMPLES_DIR / "sample_traffic.mp4"
                run_video_stream(str(demo_path), line_ratio=COUNTING_LINE_Y_RATIO, headless=False)

        elif choice == "2":
            samples = list(SAMPLES_DIR.glob("*.jpg"))
            if samples:
                print("\nPilihan Sampel:")
                for i, s in enumerate(samples, 1):
                    print(f"  [{i}] {s.name}")
                p_in = input("Pilih nomor sampel atau ketik path gambar: ").strip()
                if p_in.isdigit() and 1 <= int(p_in) <= len(samples):
                    target_p = str(samples[int(p_in) - 1])
                else:
                    target_p = p_in
                analyze_single_photo(target_p)
            else:
                p_in = input("Ketik path berkas gambar: ").strip()
                analyze_single_photo(p_in)

        elif choice == "3":
            run_batch_samples()

        elif choice == "4":
            tag = input("Masukkan tag yang dicari: ").strip()
            matches = app_collection.cari_by_tag(tag)
            print(f"\nDitemukan {len(matches)} foto dengan tag '{tag}':")
            for m in matches:
                print(f" - [{Path(m.image_path).name}] {m.kategori} | Plat: {m.ocr_text or '-'} | Safe: {m.is_safe}")

        elif choice == "5":
            cat = input("Masukkan kategori yang dicari: ").strip()
            matches = app_collection.filter_kategori(cat)
            print(f"\nDitemukan {len(matches)} foto dengan kategori '{cat}':")
            for m in matches:
                print(f" - [{Path(m.image_path).name}] Plat: {m.ocr_text or '-'} | Tags: {', '.join(m.tags[:4])}")

        elif choice == "6":
            top = app_collection.get_popular_tags(top_n=10)
            print("\n🔖 Top 10 Tag Paling Populer (Counter):")
            for tag, count in top:
                print(f"  #{tag:<20} : {count} kali")

        elif choice == "7":
            report = app_moderator.generate_moderation_report(app_collection)
            print("\n📋 LAPORAN MODERASI KESELAMATAN BERKENDARA:")
            print(f"Total Foto Koleksi : {report['total_photos']}")
            print(f"Foto Tertib (Safe) : {report['safe_count']}")
            print(f"Pelanggaran/Unsafe : {report['unsafe_count']}")
            print(f"Tingkat Kepatuhan  : {report['safe_rate_percentage']}%")

        elif choice == "8":
            if len(app_collection) == 0:
                print("❌ Koleksi masih kosong. Lakukan analisis terlebih dahulu.")
            else:
                out_path = EXPORTS_DIR / f"cli_export_{int(time.time())}.csv"
                exported = app_exporter.export_to_csv(app_collection, out_path)
                print(f"✅ Koleksi berhasil diekspor ke: {exported}")

        elif choice == "9":
            print("\nPilih Engine Vision AI:")
            print("1. YOLOv8 + EasyOCR (Local Computer Vision)")
            print("2. Google Gemini Vision (Cloud Multimodal AI)")
            print("3. Mock Vision Engine (Offline Simulation)")
            e_choice = input("Pilihan [1-3]: ").strip()
            if e_choice == "2":
                app_service.set_strategy(GeminiVisionClient())
            elif e_choice == "3":
                app_service.set_strategy(MockVisionClient())
            else:
                app_service.set_strategy(YOLOv8VisionClient())
            print(f"✅ Strategi aktif saat ini: {app_service.strategy.__class__.__name__}")

        elif choice == "0":
            print("\nTerima kasih telah menggunakan Sistem Cerdas Pemantauan Lalu Lintas AI!")
            break


def main() -> None:
    """Titik masuk eksekusi CLI."""
    args = parse_arguments()

    # Terapkan pilihan provider jika ditentukan via CLI
    if args.provider:
        p = args.provider.lower()
        if p == "gemini":
            app_service.set_strategy(GeminiVisionClient())
        elif p == "mock":
            app_service.set_strategy(MockVisionClient())
        elif p == "gpt4":
            app_service.set_strategy(GPT4VisionClient())
        else:
            app_service.set_strategy(YOLOv8VisionClient())

    # Komparasi 2 provider via CLI (Rubrik Item 17)
    if args.compare:
        compare_path = Path(args.compare)
        if not compare_path.exists():
            print(f"❌ Berkas tidak ditemukan: {args.compare}")
            return
        print(f"\n⚖️ Membandingkan hasil 2 provider untuk '{compare_path.name}'...")
        client_a = YOLOv8VisionClient()
        client_b = MockVisionClient()
        comp = app_service.compare_providers(compare_path, client_a, client_b)
        print("\n" + "=" * 65)
        print("KOMPARASI HASIL 2 PROVIDER:")
        print(f"Provider A ({comp['provider_a']['name']}):")
        print(f" - Caption: {comp['provider_a']['caption']}")
        print(f" - Tags   : {', '.join(comp['provider_a']['tags'])}")
        print(f"Provider B ({comp['provider_b']['name']}):")
        print(f" - Caption: {comp['provider_b']['caption']}")
        print(f" - Tags   : {', '.join(comp['provider_b']['tags'])}")
        print(f"Tag Bersama (Shared): {', '.join(comp['shared_tags']) if comp['shared_tags'] else '-'}")
        print("=" * 65)
        return

    if args.image:
        analyze_single_photo(args.image)
        return

    if args.source or args.demo or args.headless:
        source = resolve_video_source(args)
        run_video_stream(source, line_ratio=args.line, headless=args.headless, max_frames=args.max_frames)
        if args.export:
            csv_path = db_manager.export_violations_to_csv()
            xls_path = db_manager.export_violations_to_excel()
            logger.info("Laporan diekspor: %s dan %s", csv_path.name, xls_path.name)
        return

    # Jika tidak ada parameter khusus, buka menu interaktif
    display_interactive_menu()


if __name__ == "__main__":
    main()
