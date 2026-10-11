"""
Module: generate_samples.py
Deskripsi: Generator data uji coba mandiri yang membuat citra plat nomor,
           citra pengendara (dengan dan tanpa helm), dokumen simulasi STNK,
           serta video animasi lalu lintas sintetis untuk pengujian langsung tanpa perlu webcam luar.
"""

from pathlib import Path
import cv2
import numpy as np

from config import SAMPLES_DIR, ensure_directories

ensure_directories()

def create_sample_plate_image(plate_text: str = "B 1234 ABC", filename: str = "sample_plate.jpg") -> Path:
    """Membuat citra simulasi plat nomor kendaraan Indonesia standar (latar hitam, teks putih)."""
    h, w = 150, 420
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Latar belakang plat hitam dengan border putih tipis
    img[:] = (20, 20, 20)
    cv2.rectangle(img, (8, 8), (w - 8, h - 8), (220, 220, 220), 3)

    # Teks nomor plat utama
    cv2.putText(
        img,
        plate_text,
        (35, 80),
        cv2.FONT_HERSHEY_DUPLEX,
        1.7,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    # Bulan dan tahun pajak (misal 05.28)
    cv2.putText(
        img,
        "05 . 28",
        (160, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    out_path = SAMPLES_DIR / filename
    cv2.imwrite(str(out_path), img)
    return out_path


def create_sample_stnk_image(
    nopol: str = "B 1234 ABC",
    nama: str = "BUDI SANTOSO",
    masa_berlaku: str = "12-05-2028",
    filename: str = "sample_stnk.jpg",
) -> Path:
    """Membuat citra simulasi kartu STNK (Surat Tanda Nomor Kendaraan)."""
    h, w = 450, 700
    img = np.full((h, w, 3), 245, dtype=np.uint8)  # Warna krem / dokumen resmi

    # Header Korlantas Polri
    cv2.rectangle(img, (20, 20), (w - 20, 75), (40, 70, 140), -1)
    cv2.putText(
        img,
        "SURAT TANDA NOMOR KENDARAAN (STNK)",
        (50, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Border dokumen
    cv2.rectangle(img, (15, 15), (w - 15, h - 15), (40, 70, 140), 2)

    fields = [
        ("NO. REGISTRASI / NOPOL", nopol),
        ("NAMA PEMILIK", nama),
        ("ALAMAT", "JL. SUDIRMAN KAV. 45, JAKARTA"),
        ("MERK / TIPE", "HONDA VARIO 160 CC"),
        ("JENIS KENDARAAN", "SEPEDA MOTOR"),
        ("TAHUN PEMBUATAN", "2023"),
        ("BERLAKU S/D", masa_berlaku),
        ("STATUS DOKUMEN", "TERDAFTAR & SAH DI POLRI"),
    ]

    y = 120
    for label, val in fields:
        cv2.putText(
            img,
            f"{label.ljust(25)} : {val}",
            (40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        y += 38

    out_path = SAMPLES_DIR / filename
    cv2.imwrite(str(out_path), img)
    return out_path


def create_sample_rider_image(has_helmet: bool = False, filename: str = "sample_rider.jpg") -> Path:
    """Membuat citra representasi pengendara motor (untuk pengujian crop & validasi AI)."""
    h, w = 400, 300
    img = np.full((h, w, 3), 200, dtype=np.uint8)

    # Badan pengendara (baju/jaket)
    body_color = (60, 60, 60) if has_helmet else (180, 100, 50)
    cv2.rectangle(img, (75, 180), (225, 380), body_color, -1)

    # Kepala
    head_center = (150, 120)
    if has_helmet:
        # Helm full-face hitam / merah
        cv2.circle(img, head_center, 55, (30, 30, 180), -1)  # Helm merah
        cv2.ellipse(img, (150, 125), (40, 20), 0, 0, 180, (20, 20, 20), -1)  # Kaca visor
        cv2.putText(
            img, "SNI", (135, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )
    else:
        # Rambut dan wajah tanpa helm
        cv2.circle(img, head_center, 45, (170, 200, 230), -1)  # Kulit
        cv2.circle(img, (150, 95), 45, (20, 20, 20), -1)      # Rambut hitam terlihat jelas
        cv2.circle(img, (135, 120), 4, (10, 10, 10), -1)      # Mata
        cv2.circle(img, (165, 120), 4, (10, 10, 10), -1)

    # Keterangan visual
    status_label = "DENGAN HELM" if has_helmet else "TANPA HELM (PELANGGARAN)"
    label_color = (0, 150, 0) if has_helmet else (0, 0, 200)
    cv2.putText(
        img,
        status_label,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        label_color,
        2,
        cv2.LINE_AA,
    )

    out_path = SAMPLES_DIR / filename
    cv2.imwrite(str(out_path), img)
    return out_path


def create_sample_traffic_video(filename: str = "sample_traffic.mp4", num_frames: int = 150) -> Path:
    """
    Membuat video sintetis 30 FPS berisi pergerakan mobil dan sepeda motor
    yang melintasi garis pemantauan lalu lintas.
    """
    out_path = SAMPLES_DIR / filename
    width, height = 640, 480
    fps = 25

    # Gunakan fourcc mp4v
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_path = SAMPLES_DIR / "sample_traffic.avi"
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    for frame_idx in range(num_frames):
        frame = np.full((height, width, 3), 70, dtype=np.uint8)  # Warna aspal jalan

        # Marka jalan garis putus-putus
        for y_dash in range(0, height, 40):
            cv2.line(frame, (width // 2, y_dash), (width // 2, y_dash + 20), (255, 255, 255), 3)

        # 1. Mobil bergerak ke bawah
        car_y = int((frame_idx * 5) % (height + 100)) - 80
        if 0 <= car_y <= height:
            cv2.rectangle(frame, (120, car_y), (240, car_y + 110), (0, 180, 240), -1)
            cv2.rectangle(frame, (140, car_y + 85), (220, car_y + 105), (10, 10, 10), -1)  # Plat
            cv2.putText(
                frame, "B 2049 XYZ", (145, car_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1
            )

        # 2. Sepeda motor bergerak ke bawah (lajur kanan)
        motor_y = int((frame_idx * 6 + 50) % (height + 100)) - 70
        if 0 <= motor_y <= height:
            # Motor
            cv2.rectangle(frame, (420, motor_y + 20), (470, motor_y + 80), (200, 100, 30), -1)
            # Pengendara
            cv2.circle(frame, (445, motor_y + 15), 15, (180, 180, 240), -1)
            # Plat motor
            cv2.rectangle(frame, (430, motor_y + 65), (460, motor_y + 78), (10, 10, 10), -1)

        writer.write(frame)

    writer.release()
    return out_path


def get_sample_traffic_video(filename: str = "sample_traffic.mp4") -> Path:
    """
    Mengunduh klip video CCTV lalu lintas nyata berdurasi pendek (~2MB) dari repositori
    open-source terverifikasi agar YOLOv8 mendeteksi mobil/motor sungguhan,
    dengan fallback otomatis ke video sintetis jika offline.
    """
    out_path = SAMPLES_DIR / filename
    if out_path.exists() and out_path.stat().st_size > 50000:
        return out_path

    # Video demo lalu lintas CCTV standar publik dari Intel OpenVINO dataset
    sample_url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4"
    try:
        import requests
        print("Mengunduh klip video lalu lintas CCTV realistis...")
        resp = requests.get(sample_url, timeout=12)
        if resp.status_code == 200 and len(resp.content) > 50000:
            with open(out_path, "wb") as f:
                f.write(resp.content)
            print("Klip video lalu lintas nyata berhasil diunduh.")
            return out_path
    except Exception as e:
        print(f"Gagal mengunduh klip lalu lintas ({e}), membuat video sintetis cadangan...")

    return create_sample_traffic_video(filename=filename, num_frames=120)


def generate_all_samples() -> None:
    """Menghasilkan seluruh sampel uji coba sekaligus."""
    print("Membuat aset sampel untuk demonstrasi...")
    p1 = create_sample_plate_image("B 1234 ABC", "sample_plate_valid.jpg")
    p2 = create_sample_plate_image("D 9876 XYZ", "sample_plate_bandung.jpg")
    s1 = create_sample_stnk_image("B 1234 ABC", "BUDI SANTOSO", "12-05-2028", "sample_stnk_cocok.jpg")
    s2 = create_sample_stnk_image("B 5555 DEF", "SITI AMINAH", "10-01-2022", "sample_stnk_expired.jpg")
    r1 = create_sample_rider_image(has_helmet=False, filename="sample_rider_no_helmet.jpg")
    r2 = create_sample_rider_image(has_helmet=True, filename="sample_rider_with_helmet.jpg")
    v1 = get_sample_traffic_video("sample_traffic.mp4")

    print(f"Sampel berhasil dibuat di: {SAMPLES_DIR}")
    print(f"- Plat: {p1.name}, {p2.name}")
    print(f"- STNK: {s1.name}, {s2.name}")
    print(f"- Pengendara: {r1.name}, {r2.name}")
    print(f"- Video Demo: {v1.name}")


if __name__ == "__main__":
    generate_all_samples()
