"""
Script: generate_samples.py
Deskripsi: Membuat gambar-gambar sampel sintetis untuk demo dan pengujian:
           1. pantai_sunset.jpg (Kategori: Alam)
           2. kopi_dan_roti.jpg (Kategori: Makanan)
           3. mobil_sedan.jpg (Kategori: Kendaraan)
           4. surat_dokumen.jpg (Kategori: Dokumen)
           5. hazard_unsafe_demo.jpg (Kategori: Sensitif / Unsafe untuk uji ContentModerator)
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from config import SAMPLES_DIR


def create_gradient_sample(filename: str, top_color: tuple, bottom_color: tuple, title: str, subtitle: str) -> Path:
    """Membuat gambar gradien dengan teks informatif."""
    width, height = 640, 480
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)

    # Render gradien vertikal
    for y in range(height):
        ratio = y / height
        r = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
        g = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
        b = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Bingkai estetis
    draw.rectangle([(20, 20), (width - 20, height - 20)], outline=(255, 255, 255), width=3)
    draw.rectangle([(25, 25), (width - 25, height - 25)], outline=(0, 0, 0), width=1)

    # Teks label
    draw.text((40, 50), title, fill=(255, 255, 255))
    draw.text((40, 80), subtitle, fill=(240, 240, 240))
    draw.text((40, 420), f"SAMPLE FILE: {filename}", fill=(200, 200, 200))

    out_path = SAMPLES_DIR / filename
    image.save(out_path, quality=92)
    return out_path


def generate_all_samples():
    """Menghasilkan seluruh berkas sampel ke direktori data/samples."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    created = []

    # 1. Alam / Sunset
    created.append(create_gradient_sample(
        "pantai_sunset.jpg",
        (255, 94, 77), (255, 165, 2),
        "PANTAI & MATAHARI TERBENAM",
        "Pemandangan alam pantai saat senja dengan gradien langit keemasan."
    ))

    # 2. Makanan / Kafe
    created.append(create_gradient_sample(
        "kopi_dan_roti.jpg",
        (109, 76, 65), (215, 204, 200),
        "KOPI & KULINER KAFE",
        "Secangkir kopi espresso hangat dan roti panggang di kafe santai."
    ))

    # 3. Kendaraan
    created.append(create_gradient_sample(
        "mobil_sedan.jpg",
        (30, 60, 114), (42, 82, 152),
        "KENDARAAN MOBIL SEDAN",
        "Mobil sedan melaju di jalan raya dengan plat nomor B 1234 XYZ."
    ))

    # 4. Dokumen
    created.append(create_gradient_sample(
        "surat_dokumen.jpg",
        (236, 240, 241), (189, 195, 199),
        "DOKUMEN RESMI NEGARA",
        "SURAT KETERANGAN RESMI No. 123/2026 TERTANGGAL AKTIF."
    ))

    # 5. Unsafe / Hazard demo
    created.append(create_gradient_sample(
        "hazard_unsafe_demo.jpg",
        (192, 57, 43), (44, 62, 80),
        "KONTEN BERBAHAYA / HAZARD (DEMO)",
        "Citra simulasi konten terlarang untuk pengujian ContentModerator."
    ))

    print(f"Berhasil menghasilkan {len(created)} file sampel di: {SAMPLES_DIR}")
    for p in created:
        print(f" - {p.name}")


if __name__ == "__main__":
    generate_all_samples()
