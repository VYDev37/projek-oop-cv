"""
Module: main.py
Deskripsi: Script eksekusi utama CLI / Standalone OpenCV untuk Sistem Cerdas
           Pemantauan Lalu Lintas dan Keselamatan Pengendara Real-Time.
           Dapat dijalankan langsung pada CCTV / Webcam / Berkas Video.
"""

import argparse
import sys
import time
import logging
from typing import Union, Set
from pathlib import Path
import cv2

from config import (
    DEFAULT_YOLO_MODEL,
    COUNTING_LINE_Y_RATIO,
    SAMPLES_DIR,
    ensure_directories,
)
from database import db_manager
from pipeline import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrafficSystemMain")


def parse_arguments() -> argparse.Namespace:
    """Menguraikan argumen baris perintah."""
    parser = argparse.ArgumentParser(
        description="Sistem Cerdas Pemantauan Lalu Lintas & Keselamatan Pengendara Real-Time"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Sumber video: '0' untuk webcam, atau path berkas video (.mp4/.avi)",
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
        default=True,
        help="Ekspor otomatis laporan pelanggaran ke CSV dan Excel saat selesai",
    )
    return parser.parse_args()


def resolve_video_source(args: argparse.Namespace) -> Union[int, str]:
    """Menentukan sumber video yang valid."""
    if args.demo:
        demo_file = SAMPLES_DIR / "sample_traffic.mp4"
        if not demo_file.exists():
            from generate_samples import generate_all_samples
            generate_all_samples()
        return str(demo_file)

    if not args.source:
        # Coba demo file terlebih dahulu jika ada, atau fallback webcam 0
        demo_file = SAMPLES_DIR / "sample_traffic.mp4"
        if demo_file.exists():
            return str(demo_file)
        return 0

    if args.source.isdigit():
        return int(args.source)

    return args.source


def main() -> None:
    """Fungsi utama eksekusi program."""
    ensure_directories()
    args = parse_arguments()
    source = resolve_video_source(args)

    logger.info("=" * 70)
    logger.info("MEMULAI SISTEM CERDAS PEMANTAUAN LALU LINTAS & KESELAMATAN AI")
    logger.info("=" * 70)
    logger.info("Sumber Video   : %s", source)
    logger.info("Garis Hitung   : %s", args.line)
    logger.info("Mode Headless  : %s", args.headless)
    logger.info("=" * 70)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Gagal membuka sumber video: %s", source)
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    logger.info("Resolusi Input : %dx%d @ %.1f FPS", frame_width, frame_height, fps_in)
    logger.info("Tekan tombol 'q' pada jendela video untuk keluar.\n")

    frame_count = 0
    start_time = time.time()
    last_reported_alerts = set()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Jika berkas video, ulangi dari awal untuk continuous stream
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_count += 1
            if args.max_frames > 0 and frame_count >= args.max_frames:
                logger.info("Mencapai batas maksimum frame (%d). Menghentikan loop...", args.max_frames)
                break

            # Proses frame menggunakan pipeline terintegrasi
            annotated_frame, alerts, meta = pipeline.process_frame(frame, line_ratio=args.line)

            # Notifikasi konsol jika ada pelanggaran baru
            for alert in alerts:
                if alert.id not in last_reported_alerts:
                    last_reported_alerts.add(alert.id)
                    logger.warning(
                        "[PELANGGARAN BARU] ID Track: %s | Jenis: %s | Nopol: %s | Tindakan: %s",
                        alert.track_id,
                        alert.violation_type,
                        alert.license_plate,
                        alert.recommended_action,
                    )

            # Hitung dan tampilkan FPS pemrosesan
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0

            # Gambar indikator status di pojok kanan bawah
            cv2.putText(
                annotated_frame,
                f"FPS: {current_fps:.1f}",
                (frame.shape[1] - 120, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Tampilkan jendela OpenCV jika bukan mode headless
            if not args.headless:
                cv2.imshow("Smart Traffic AI - Korlantas Monitoring", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Pengguna menekan tombol 'q'. Menghentikan program...")
                    break

    except KeyboardInterrupt:
        logger.info("Program dihentikan oleh pengguna (Ctrl+C).")

    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

        pipeline.shutdown()

        # Ringkasan Akhir
        logger.info("\n" + "=" * 70)
        logger.info("RINGKASAN AKHIR PEMANTAUAN LALU LINTAS")
        logger.info("=" * 70)
        stats = db_manager.get_system_statistics()
        logger.info("Total Kendaraan Melintas : %d", max(stats["total_vehicles"], pipeline.detector.total_counted))
        logger.info("Total Pelanggaran        : %d", stats["total_violations"])
        logger.info("Kepatuhan Helm           : %.1f%%", stats["compliance_rate"])
        logger.info("Detail Kendaraan         : %s", stats["vehicles_by_type"])
        logger.info("Detail Pelanggaran       : %s", stats["violations_by_type"])

        if args.export:
            csv_path = db_manager.export_violations_to_csv()
            xlsx_path = db_manager.export_violations_to_excel()
            logger.info("Laporan CSV tersimpan   : %s", csv_path)
            logger.info("Laporan Excel tersimpan : %s", xlsx_path)
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
