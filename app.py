"""
Module: app.py
Deskripsi: Dashboard Web Interaktif berbasis Streamlit untuk Platform:
           Sistem Computer Vision & AI Image Analysis & Tagging Otomatis [I2]
           Mencakup fitur:
           - Analisis Citra Tunggal (Upload Gambar / Snapshot Kamera) via YOLOv8 / Gemini / Mock
           - Pemantauan Video & Live Stream Kontinu (Tracking Kendaraan, Virtual Line Crossing, ETLE Auto-Logging)
           - Pemilihan Engine via Strategy Pattern (YOLOv8 Local CV, Gemini Multimodal, Mock Engine)
           - Batch Analysis Banyak Foto Sekaligus
           - Komparasi 2 Engine Berdampingan
           - Koleksi Foto dengan Iterator Pattern, Pencarian Tag, Filter Kategori
           - Statistik Tag Terpopuler (Counter)
           - Laporan Kepatuhan / Moderasi Keselamatan (ContentModerator)
           - Ekspor Koleksi ke CSV (CollectionCSVExporter - SOLID: SRP)
"""

import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
import cv2
import pandas as pd
from PIL import Image
import streamlit as st

from config import SAMPLES_DIR, EXPORTS_DIR, UPLOADS_DIR, GEMINI_API_KEY, COUNTING_LINE_Y_RATIO
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
from pipeline import TrafficMonitoringPipeline
from detector import TrafficDetector
from visualizer import TrafficVisualizer
from ocr_engine import plate_ocr
from database import db_manager

# KONFIGURASI HALAMAN STREAMLIT & CUSTOM STYLING
st.set_page_config(
    page_title="AI Vision & Traffic Monitoring [I2]",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-safe {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-unsafe {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .tag-pill {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #DBEAFE;
        padding: 3px 9px;
        border-radius: 16px;
        font-size: 0.82rem;
        margin-right: 6px;
        margin-bottom: 6px;
        display: inline-block;
    }
    .color-pill {
        background-color: #F1F5F9;
        color: #334155;
        border: 1px solid #E2E8F0;
        padding: 3px 9px;
        border-radius: 16px;
        font-size: 0.82rem;
        margin-right: 6px;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# INISIALISASI SESSION STATE
if "collection" not in st.session_state:
    st.session_state.collection = PhotoCollection()
    init_files = list(SAMPLES_DIR.glob("*.jpg"))
    if init_files:
        mock_init = MockVisionClient()
        for f in init_files[:3]:
            try:
                res = mock_init.analyze(f)
                st.session_state.collection.tambah(res)
            except Exception:
                pass

if "service" not in st.session_state or not isinstance(st.session_state.service, ImageAnalysisService):
    default_strategy = YOLOv8VisionClient()
    st.session_state.service = ImageAnalysisService(strategy=default_strategy)

if "moderator" not in st.session_state:
    st.session_state.moderator = ContentModerator()

if "exporter" not in st.session_state:
    st.session_state.exporter = CollectionCSVExporter()

# SIDEBAR: PENGATURAN STRATEGY PATTERN & STATUS SISTEM
with st.sidebar:
    st.markdown("### ⚙️ Engine Vision (Strategy Pattern)")
    provider_choice = st.selectbox(
        "Pilih Active Strategy Provider:",
        [
            "YOLOv8 + EasyOCR (Local Computer Vision)",
            "Google Gemini Vision (Multimodal Cloud)",
            "Mock Vision Engine (Offline Demo)",
            "OpenAI GPT-4o (Simulated/API)",
        ],
        index=0,
        help="Menerapkan Design Pattern Strategy: menukar engine Vision AI saat runtime secara transparan.",
    )

    if provider_choice.startswith("YOLOv8"):
        new_strategy = YOLOv8VisionClient()
    elif provider_choice.startswith("Google Gemini"):
        if not GEMINI_API_KEY:
            st.warning("⚠️ GEMINI_API_KEY belum disetel di .env. Menggunakan simulasi mock.")
            new_strategy = MockVisionClient(model_name="mock-gemini-fallback")
        else:
            new_strategy = GeminiVisionClient()
    elif provider_choice.startswith("OpenAI GPT-4o"):
        new_strategy = GPT4VisionClient()
    else:
        new_strategy = MockVisionClient()

    try:
        st.session_state.service.set_strategy(new_strategy)
    except TypeError:
        st.session_state.service = ImageAnalysisService(strategy=new_strategy)

    st.success(f"Strategi Aktif: **{st.session_state.service.strategy.__class__.__name__}**")

    st.markdown("---")
    st.markdown("### 📊 Ringkasan Koleksi")
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.metric("Total Foto", len(st.session_state.collection))
    with col_k2:
        report_quick = st.session_state.moderator.generate_moderation_report(st.session_state.collection)
        st.metric("Tertib / Aman", f"{report_quick['safe_rate_percentage']}%")

    if st.button("🗑️ Kosongkan Koleksi", use_container_width=True):
        st.session_state.collection.clear()
        st.rerun()

# HEADER UTAMA APLIKASI
st.markdown('<div class="main-header">🚦 Sistem Computer Vision & AI Traffic Monitoring</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Platform terpadu berbasis <b>Object-Oriented Programming (OOP)</b>: '
    'Deteksi objek, pelacakan kendaraan, pengenalan plat OCR, evaluasi keselamatan berkendara (helm), '
    'dan manajemen koleksi cerdas.'
    '</div>',
    unsafe_allow_html=True,
)

# TABS NAVIGASI UTAMA (5 TAB LENGKAP)
tab_single, tab_video, tab_batch, tab_compare, tab_gallery = st.tabs([
    "🔍 1. Analisis Citra (Foto / Kamera)",
    "📹 2. Pemantauan Video & Live Stream",
    "⚡ 3. Analisis Batch",
    "⚖️ 4. Komparasi 2 Engine",
    "📚 5. Galeri Koleksi & Moderasi",
])

# ==============================================================================
# TAB 1: ANALISIS CITRA TUNGGAL (UPLOAD / KAMERA)
# ==============================================================================
with tab_single:
    st.subheader("Analisis Citra Tunggal (Foto / Kamera)")

    col_input, col_display = st.columns([1, 1], gap="medium")

    with col_input:
        input_source = st.radio(
            "Pilih Sumber Citra:",
            ["Pilih dari Sampel Bawaan", "Unggah Berkas Gambar (JPG/PNG)", "Ambil Foto Langsung (Kamera/Webcam)"],
            horizontal=True,
        )

        selected_image: Optional[Image.Image] = None
        image_name_label: str = "foto_input.jpg"

        if input_source == "Pilih dari Sampel Bawaan":
            samples = list(SAMPLES_DIR.glob("*.jpg"))
            if samples:
                sample_map = {p.name: p for p in samples}
                chosen_name = st.selectbox("Pilih Sampel Foto:", list(sample_map.keys()))
                chosen_path = sample_map[chosen_name]
                selected_image = Image.open(chosen_path)
                image_name_label = chosen_name
            else:
                st.info("Sampel belum ada. Silakan jalankan generate_samples.py.")

        elif input_source == "Unggah Berkas Gambar (JPG/PNG)":
            uploaded_file = st.file_uploader("Unggah berkas foto:", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file)
                image_name_label = uploaded_file.name

        elif input_source == "Ambil Foto Langsung (Kamera/Webcam)":
            camera_file = st.camera_input("Jepret foto langsung:")
            if camera_file is not None:
                selected_image = Image.open(camera_file)
                image_name_label = f"camera_snap_{datetime.now().strftime('%H%M%S')}.jpg"

        if selected_image is not None:
            st.image(selected_image, caption=f"Pratinjau Citra Asli: {image_name_label}", use_container_width=True)
            btn_analyze = st.button("🚀 Jalankan Analisis AI", type="primary", use_container_width=True)
        else:
            btn_analyze = False

    with col_display:
        st.markdown("#### 📋 Hasil Analisis Computer Vision")

        if btn_analyze and selected_image is not None:
            with st.spinner(f"Memproses via {st.session_state.service.strategy.__class__.__name__}..."):
                try:
                    result = st.session_state.service.analyze_image(selected_image, image_name=image_name_label)
                    st.session_state.last_result = result
                    st.session_state.collection.tambah(result)
                    st.success("✅ Analisis Berhasil & Otomatis Disimpan ke Koleksi!")
                except VisionAPIException as vae:
                    st.error(f"❌ VisionAPIException: {vae}")
                    st.session_state.last_result = None
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan tak terduga: {e}")
                    st.session_state.last_result = None

        if "last_result" in st.session_state and st.session_state.last_result:
            res: PhotoAnalysisResult = st.session_state.last_result

            # Tampilkan gambar beranotasi jika tersedia
            if res.annotated_image_path and Path(res.annotated_image_path).exists():
                st.image(res.annotated_image_path, caption="Deteksi Visual Beranotasi (Bounding Box)", use_container_width=True)

            if res.is_safe:
                st.markdown('<span class="badge-safe">🛡️ TERTIB / PATUHI ATURAN (SAFE)</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-unsafe">⚠️ TERDETEKSI PELANGGARAN / UNSAFE</span>', unsafe_allow_html=True)

            st.markdown(f"**Provider:** `{res.provider}` | ⏱️ `{res.processing_time_ms} ms`")
            st.markdown(f"**Kategori:** `{res.kategori}`")

            st.markdown("##### 📝 Ringkasan Naratif:")
            st.info(res.caption)

            if res.ocr_text:
                st.markdown("##### 🚗 Plat Nomor Terdeteksi (OCR):")
                st.code(res.ocr_text, language="text")

            st.markdown("##### 🔖 Tags Terdeteksi:")
            tags_html = "".join([f'<span class="tag-pill">#{t}</span>' for t in res.tags])
            st.markdown(tags_html, unsafe_allow_html=True)

            st.markdown("##### 🔍 Objek Fisik:")
            st.write(", ".join(res.objects) if res.objects else "-")

            st.markdown("##### 🎨 Palet Warna Dominan:")
            colors_html = "".join([f'<span class="color-pill">🎨 {c}</span>' for c in res.colors])
            st.markdown(colors_html, unsafe_allow_html=True)
        else:
            if not btn_analyze:
                st.info("Pilih atau ambil foto di sebelah kiri, lalu klik **Jalankan Analisis AI**.")

# ==============================================================================
# TAB 2: PEMANTAUAN VIDEO & LIVE STREAM (CONTINUOUS STREAM)
# ==============================================================================
with tab_video:
    st.subheader("📹 Pemantauan Video & Live Stream (Traffic Tracking & ETLE)")
    st.write(
        "Menjalankan pelacakan objek kendaraan secara kontinu (YOLOv8 + ByteTrack), "
        "penghitungan garis virtual (*virtual line crossing*), serta logging pelanggaran otomatis ke koleksi."
    )

    col_v_conf, col_v_stream = st.columns([1, 2], gap="medium")

    with col_v_conf:
        stream_mode = st.radio(
            "Pilih Sumber Video:",
            [
                "Gunakan Sampel Video Bawaan (sample_traffic.mp4)",
                "Unggah Berkas Video Sendiri (MP4/AVI/MOV)",
                "Gunakan Webcam Komputer (Kamera Live)",
            ],
        )

        video_path = None
        if stream_mode.startswith("Gunakan Sampel Video"):
            sample_vid = SAMPLES_DIR / "sample_traffic.mp4"
            if not sample_vid.exists():
                from generate_samples import generate_all_samples
                generate_all_samples()
            video_path = str(sample_vid)
            st.info("📹 Berkas aktif: `data/samples/sample_traffic.mp4`")

        elif stream_mode.startswith("Unggah Berkas Video"):
            v_upload = st.file_uploader("Unggah berkas video rekaman:", type=["mp4", "avi", "mov", "mkv"])
            if v_upload is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(v_upload.read())
                video_path = tfile.name
                st.success(f"Video berhasil dimuat: {v_upload.name}")
        else:
            cam_idx = st.number_input("Indeks Kamera:", min_value=0, max_value=5, value=0)
            video_path = int(cam_idx)

        max_frames = st.slider("Maksimum Frame yang Diproses:", min_value=30, max_value=600, value=150)
        start_stream = st.button("▶️ Mulai Pemantauan Video", type="primary", use_container_width=True)

    with col_v_stream:
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        m_active = metric_col1.empty()
        m_crossed = metric_col2.empty()
        m_alerts = metric_col3.empty()

        frame_placeholder = st.empty()
        status_placeholder = st.empty()

        if start_stream and video_path is not None:
            pipeline = TrafficMonitoringPipeline(
                photo_collection=st.session_state.collection,
                async_ai=True,
            )

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Gagal membuka sumber video.")
            else:
                frame_count = 0
                status_placeholder.info("Sedang menjalankan pelacakan Computer Vision...")

                while cap.isOpened() and frame_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Resize frame untuk performa stabil
                    frame_resized = cv2.resize(frame, (640, 480))
                    annotated_frame, alerts, metadata = pipeline.process_frame(frame_resized)

                    # Update metrik real-time
                    v_counts = metadata.get("vehicle_counts", {})
                    total_active = sum(v_counts.values()) if isinstance(v_counts, dict) else 0
                    crossed_count = metadata.get("total_line_crossings", 0)

                    m_active.metric("Kendaraan Aktif", total_active)
                    m_crossed.metric("Melintasi Garis", crossed_count)
                    m_alerts.metric("Pelanggaran", len(alerts))

                    # Tampilkan frame di Streamlit
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                    frame_count += 1
                    time.sleep(0.02)

                cap.release()
                status_placeholder.success(
                    f"Pemrosesan selesai ({frame_count} frame diproses). "
                    f"Tangkapan pelanggaran baru telah otomatis dicatat ke Koleksi!"
                )
                time.sleep(1.0)
                st.rerun()

# ==============================================================================
# TAB 3: ANALISIS BATCH
# ==============================================================================
with tab_batch:
    st.subheader("⚡ Pemrosesan Batch Banyak Foto Sekaligus")
    st.write("Proses banyak foto dalam satu antrean kerja dan otomatis tambahkan hasilnya ke koleksi.")

    batch_source = st.radio(
        "Pilih Sumber Batch:",
        ["Gunakan Seluruh Foto Sampel", "Unggah Banyak Berkas Sekaligus (Multi-Upload)"],
        horizontal=True,
    )

    batch_images = []
    if batch_source == "Gunakan Seluruh Foto Sampel":
        sample_files = list(SAMPLES_DIR.glob("*.jpg"))
        st.write(f"Ditemukan **{len(sample_files)} berkas sampel** di direktori data.")
        batch_images = sample_files
    else:
        uploaded_multi = st.file_uploader(
            "Unggah banyak foto sekaligus:",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        if uploaded_multi:
            batch_images = uploaded_multi

    if batch_images:
        if st.button("🚀 Mulai Analisis Batch", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(curr, tot, msg):
                progress_bar.progress(curr / tot)
                status_text.text(f"[{curr}/{tot}] {msg}")

            results = st.session_state.collection.batch_analyze(
                batch_images,
                client=st.session_state.service.strategy,
                progress_callback=update_progress,
            )

            progress_bar.progress(1.0)
            status_text.text(f"Selesai! Berhasil memproses {len(results)} dari {len(batch_images)} foto.")
            st.success(f"Analisis batch selesai! {len(results)} foto baru telah masuk ke koleksi.")

            table_data = []
            for r in results:
                table_data.append({
                    "Berkas": Path(r.image_path).name if r.image_path else "-",
                    "Kategori": r.kategori,
                    "Moderasi": "AMAN" if r.is_safe else "UNSAFE",
                    "Plat OCR": r.ocr_text or "-",
                    "Caption": r.caption[:55] + "...",
                    "Tags": ", ".join(r.tags[:4]),
                    "Waktu (ms)": r.processing_time_ms,
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

# ==============================================================================
# TAB 4: KOMPARASI 2 ENGINE
# ==============================================================================
with tab_compare:
    st.subheader("⚖️ Komparasi Hasil 2 Engine Berbeda")
    st.write("Bandingkan hasil analisis dua engine Vision AI secara berdampingan pada citra yang sama.")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        p1_choice = st.selectbox(
            "Engine A:",
            ["YOLOv8 + EasyOCR (Local CV)", "Google Gemini Vision", "Mock Vision (Offline)"],
            index=0,
        )
    with col_p2:
        p2_choice = st.selectbox(
            "Engine B:",
            ["Google Gemini Vision", "Mock Vision (Offline)", "YOLOv8 + EasyOCR (Local CV)"],
            index=1,
        )

    def get_client_by_name(name: str) -> VisionClient:
        if name.startswith("YOLOv8"):
            return YOLOv8VisionClient()
        elif name.startswith("Google Gemini"):
            return GeminiVisionClient() if GEMINI_API_KEY else MockVisionClient(model_name="mock-gemini")
        else:
            return MockVisionClient()

    client_a = get_client_by_name(p1_choice)
    client_b = get_client_by_name(p2_choice)

    samples_avail = list(SAMPLES_DIR.glob("*.jpg"))
    if samples_avail:
        compare_image_path = st.selectbox(
            "Pilih Gambar untuk Dikomparasi:",
            samples_avail,
            format_func=lambda p: p.name,
        )
        st.image(str(compare_image_path), width=350)

        if st.button("⚖️ Jalankan Komparasi Engine", type="primary"):
            with st.spinner("Menjalankan komparasi kedua engine..."):
                comp = st.session_state.service.compare_providers(
                    compare_image_path, client_a, client_b, image_name=compare_image_path.name
                )

                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.markdown(f"#### 🅰️ {comp['provider_a']['name']}")
                    st.write(f"**Caption:** {comp['provider_a']['caption']}")
                    st.write(f"**Kategori:** {comp['provider_a']['kategori']}")
                    st.write(f"**Waktu:** {comp['provider_a']['processing_time_ms']} ms")
                    st.write("**Tags:** " + ", ".join(comp['provider_a']['tags']))

                with col_res2:
                    st.markdown(f"#### 🅱️ {comp['provider_b']['name']}")
                    st.write(f"**Caption:** {comp['provider_b']['caption']}")
                    st.write(f"**Kategori:** {comp['provider_b']['kategori']}")
                    st.write(f"**Waktu:** {comp['provider_b']['processing_time_ms']} ms")
                    st.write("**Tags:** " + ", ".join(comp['provider_b']['tags']))

                st.markdown("---")
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Shared Tags (Tag Bersama)", len(comp['shared_tags']))
                    st.write(", ".join(comp['shared_tags']) if comp['shared_tags'] else "Tidak ada tag identik.")
                with col_m2:
                    st.metric("Tingkat Keselarasan Tag", f"{comp['tag_agreement_rate']}%")

# ==============================================================================
# TAB 5: GALERI KOLEKSI, STATISTIK COUNTER & EKSPOR CSV
# ==============================================================================
with tab_gallery:
    st.subheader("📚 Galeri Koleksi, Pencarian & Moderasi")
    report = st.session_state.moderator.generate_moderation_report(st.session_state.collection)
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Foto di Koleksi", report["total_photos"])
    with col_m2:
        st.metric("Tertib / Aman (Safe)", report["safe_count"])
    with col_m3:
        st.metric("Pelanggaran (Unsafe)", report["unsafe_count"])
    with col_m4:
        st.metric("Kepatuhan Aturan", f"{report['safe_rate_percentage']}%")

    st.markdown("---")

    col_search, col_filter, col_export = st.columns([2, 2, 2])

    with col_search:
        search_tag = st.text_input("🔍 Cari Berdasarkan Tag (cari_by_tag):", placeholder="misal: motorcycle, tertib, pelanggaran")

    with col_filter:
        kategori_options = ["Semua Kategori"] + list(st.session_state.collection.get_kategori_distribution().keys())
        selected_kat = st.selectbox("📂 Filter Per Kategori (filter_kategori):", kategori_options)

    with col_export:
        st.markdown("<br>", unsafe_allow_html=True)
        if len(st.session_state.collection) > 0:
            csv_path = EXPORTS_DIR / "koleksi_foto_export.csv"
            st.session_state.exporter.export_to_csv(st.session_state.collection, csv_path)

            with open(csv_path, "rb") as f:
                st.download_button(
                    label="📥 Unduh CSV Koleksi (SOLID: SRP)",
                    data=f,
                    file_name="koleksi_foto_ai.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    displayed_photos = st.session_state.collection.get_all_photos()

    if search_tag:
        displayed_photos = st.session_state.collection.cari_by_tag(search_tag)
        st.info(f"Menampilkan {len(displayed_photos)} foto yang memuat tag: `{search_tag}`")

    if selected_kat != "Semua Kategori":
        displayed_photos = [p for p in displayed_photos if p.kategori == selected_kat]
        st.info(f"Menampilkan {len(displayed_photos)} foto dalam kategori: `{selected_kat}`")

    with st.expander("📊 Lihat Statistik Tag Terpopuler (Counter) & Distribusi Kategori", expanded=False):
        col_st1, col_st2 = st.columns(2)
        with col_st1:
            st.markdown("##### 🏷️ Top 5 Tag Paling Sering Muncul (Counter):")
            pop_tags = st.session_state.collection.get_popular_tags(top_n=5)
            if pop_tags:
                df_tags = pd.DataFrame(pop_tags, columns=["Tag", "Frekuensi"])
                st.bar_chart(df_tags.set_index("Tag"))
            else:
                st.write("Belum ada data tag.")

        with col_st2:
            st.markdown("##### 📁 Distribusi Foto per Kategori:")
            dist = st.session_state.collection.get_kategori_distribution()
            if dist:
                df_kat = pd.DataFrame(list(dist.items()), columns=["Kategori", "Jumlah"])
                st.bar_chart(df_kat.set_index("Kategori"))
            else:
                st.write("Belum ada data kategori.")

    st.markdown(f"#### 🖼️ Daftar Foto dalam Koleksi ({len(displayed_photos)} Foto)")

    if displayed_photos:
        cols = st.columns(3)
        for i, photo in enumerate(displayed_photos):
            with cols[i % 3]:
                annotated = photo.annotated_image_path if photo.annotated_image_path and Path(photo.annotated_image_path).exists() else None
                if annotated:
                    st.image(annotated, use_container_width=True)

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <b>{photo.kategori}</b> | <small>{photo.timestamp}</small><br>
                        <span class="{'badge-safe' if photo.is_safe else 'badge-unsafe'}">
                            {'AMAN / TERTIB' if photo.is_safe else 'PELANGGARAN'}
                        </span>
                        <p style="margin-top: 8px; font-size: 0.9rem;">{photo.caption}</p>
                        <small style="color: #64748B;">Tags: {', '.join(photo.tags[:4])}</small>
                        {f'<br><small style="color: #1E3A8A; font-weight: bold;">Plat: {photo.ocr_text}</small>' if photo.ocr_text else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.warning("Tidak ada foto yang cocok dengan kriteria pencarian atau filter.")
