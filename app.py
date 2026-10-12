"""
Module: app.py
Deskripsi: Dashboard Web Interaktif berbasis Streamlit untuk Platform:
           [I2] Sistem AI Image Analysis & Tagging Otomatis
           Mencakup fitur-fitur:
           - Analisis Foto Tunggal (Caption, Tags, Objects, Colors, OCR, Moderasi)
           - Pemilihan Provider via Strategy Pattern (Gemini / GPT-4o / Mock)
           - Batch Analysis
           - Komparasi 2 Provider
           - Koleksi Foto dengan Iterator Pattern, Pencarian Tag, Filter Kategori
           - Statistik Tag Terpopuler (Counter)
           - Laporan Moderasi (ContentModerator)
           - Ekspor Koleksi ke CSV (CollectionCSVExporter)
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
from PIL import Image
import streamlit as st

from config import SAMPLES_DIR, EXPORTS_DIR, GEMINI_API_KEY, OPENAI_API_KEY
from vision_client import (
    VisionClient,
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

# KONFIGURASI HALAMAN STREAMLIT & CUSTOM STYLING
st.set_page_config(
    page_title="AI Image Analysis & Tagging [I2]",
    page_icon="📸",
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

# INISIALISASI SESSION STATE (KOLEKSI & SERVICE)
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
    default_strategy = GeminiVisionClient() if GEMINI_API_KEY else MockVisionClient()
    st.session_state.service = ImageAnalysisService(strategy=default_strategy)

if "moderator" not in st.session_state:
    st.session_state.moderator = ContentModerator()

if "exporter" not in st.session_state:
    st.session_state.exporter = CollectionCSVExporter()

# SIDEBAR: PENGATURAN STRATEGY PATTERN & STATUS SISTEM
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Provider (Strategy Pattern)")
    provider_choice = st.selectbox(
        "Pilih Active Strategy Provider:",
        ["Google Gemini Vision", "OpenAI GPT-4o Vision", "Mock Vision (Offline Demo)"],
        index=0 if GEMINI_API_KEY else 2,
        help="Menerapkan Design Pattern Strategy: menukar provider AI saat runtime tanpa merombak sistem.",
    )

    if provider_choice == "Google Gemini Vision":
        if not GEMINI_API_KEY:
            st.warning("⚠️ GEMINI_API_KEY belum disetel di .env. Menggunakan simulasi mock.")
            new_strategy = MockVisionClient(model_name="mock-gemini-fallback")
        else:
            new_strategy = GeminiVisionClient()
    elif provider_choice == "OpenAI GPT-4o Vision":
        if not OPENAI_API_KEY:
            st.info("ℹ️ OPENAI_API_KEY tidak disetel. Menggunakan simulasi cerdas GPT-4o.")
        new_strategy = GPT4VisionClient()
    else:
        new_strategy = MockVisionClient()

    try:
        st.session_state.service.set_strategy(new_strategy)
    except TypeError:
        st.session_state.service = ImageAnalysisService(strategy=new_strategy)

    st.success(f"Strategi Aktif: **{st.session_state.service.strategy.__class__.__name__}**")

    st.markdown("---")
    st.markdown("### 📊 Statistik Koleksi Cepat")
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.metric("Total Foto", len(st.session_state.collection))
    with col_k2:
        report_quick = st.session_state.moderator.generate_moderation_report(st.session_state.collection)
        st.metric("Aman (Safe)", f"{report_quick['safe_rate_percentage']}%")

    if st.button("🗑️ Kosongkan Koleksi Foto", use_container_width=True):
        st.session_state.collection.clear()
        st.rerun()

# HEADER UTAMA APLIKASI
st.markdown('<div class="main-header">📸 Sistem AI Image Analysis & Tagging Otomatis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Platform kurasi foto cerdas berbasis <b>Object-Oriented Programming (OOP)</b> '
    'dengan integrasi Vision AI (Gemini / GPT-4o / Mock): Captioning, Tagging, Object Extraction, Kategori, Moderasi, '
    'dan Koleksi Foto terstruktur.'
    '</div>',
    unsafe_allow_html=True,
)

# TABS NAVIGASI UTAMA (4 TAB LENGKAP)
tab_single, tab_batch, tab_compare, tab_gallery = st.tabs([
    "🔍 1. Analisis Foto Tunggal",
    "⚡ 2. Analisis Batch",
    "⚖️ 3. Komparasi 2 Provider",
    "📚 4. Galeri Koleksi & Moderasi",
])

# TAB 1: ANALISIS FOTO TUNGGAL
with tab_single:
    st.subheader("Analisis Mendalam Citra Tunggal")

    col_input, col_display = st.columns([1, 1], gap="medium")

    with col_input:
        input_source = st.radio(
            "Pilih Sumber Foto:",
            ["Pilih dari Sampel Bawaan", "Unggah Berkas Gambar (JPG/PNG)", "Ambil Foto Langsung (Kamera)"],
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

        elif input_source == "Ambil Foto Langsung (Kamera)":
            camera_file = st.camera_input("Jepret foto langsung:")
            if camera_file is not None:
                selected_image = Image.open(camera_file)
                image_name_label = f"camera_snap_{datetime.now().strftime('%H%M%S')}.jpg"

        if selected_image is not None:
            st.image(selected_image, caption=f"Pratinjau: {image_name_label}", use_container_width=True)
            btn_analyze = st.button("🚀 Jalankan Analisis AI", type="primary", use_container_width=True)
        else:
            btn_analyze = False

    with col_display:
        st.markdown("#### 📋 Hasil Analisis Vision AI")

        if btn_analyze and selected_image is not None:
            with st.spinner(f"Menganalisis menggunakan {st.session_state.service.strategy.__class__.__name__}..."):
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

            if res.is_safe:
                st.markdown('<span class="badge-safe">🛡️ AMAN (SAFE)</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-unsafe">⚠️ TIDAK AMAN (UNSAFE / SENSITIF)</span>', unsafe_allow_html=True)

            st.markdown(f"**Provider:** `{res.provider}` | ⏱️ `{res.processing_time_ms} ms`")
            st.markdown(f"**Kategori:** `{res.kategori}`")

            st.markdown("##### 📝 Caption Deskriptif:")
            st.info(res.caption)

            st.markdown("##### 🔖 Tags Kata Kunci:")
            tags_html = "".join([f'<span class="tag-pill">#{t}</span>' for t in res.tags])
            st.markdown(tags_html, unsafe_allow_html=True)

            st.markdown("##### 🔍 Objek Fisik Terdeteksi:")
            if res.objects:
                st.write(", ".join(res.objects))
            else:
                st.write("Tidak ada objek terpisah yang terdaftar.")

            st.markdown("##### 🎨 Palet Warna Dominan:")
            colors_html = "".join([f'<span class="color-pill">🎨 {c}</span>' for c in res.colors])
            st.markdown(colors_html, unsafe_allow_html=True)

            if res.ocr_text:
                st.markdown("##### 📄 Teks OCR:")
                st.code(res.ocr_text, language="text")
        else:
            if not btn_analyze:
                st.info("Pilih atau unggah foto di sebelah kiri, lalu klik **Jalankan Analisis AI**.")

# TAB 2: ANALISIS BATCH
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

            processed_count = 0
            total_items = len(batch_images)

            def update_progress(curr, tot, msg):
                progress_bar.progress(curr / tot)
                status_text.text(f"[{curr}/{tot}] {msg}")

            results = st.session_state.collection.batch_analyze(
                batch_images,
                client=st.session_state.service.strategy,
                progress_callback=update_progress,
            )

            progress_bar.progress(1.0)
            status_text.text(f"Selesai! Berhasil memproses {len(results)} dari {total_items} foto.")
            st.success(f"Analisis batch selesai! {len(results)} foto baru telah masuk ke koleksi.")

            table_data = []
            for r in results:
                table_data.append({
                    "Berkas": Path(r.image_path).name if r.image_path else "-",
                    "Kategori": r.kategori,
                    "Moderasi": "AMAN" if r.is_safe else "UNSAFE",
                    "Caption": r.caption[:50] + "...",
                    "Tags": ", ".join(r.tags[:4]),
                    "Waktu (ms)": r.processing_time_ms,
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

# TAB 3: KOMPARASI 2 PROVIDER 
with tab_compare:
    st.subheader("⚖️ Komparasi Hasil 2 Provider Berbeda")
    st.write("Bandingkan hasil analisis dua provider Vision AI secara berdampingan pada citra yang sama.")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        p1_choice = st.selectbox(
            "Provider A:",
            ["Google Gemini Vision", "Mock Vision (Offline)", "OpenAI GPT-4o"],
            index=0,
        )
    with col_p2:
        p2_choice = st.selectbox(
            "Provider B:",
            ["OpenAI GPT-4o", "Mock Vision (Offline)", "Google Gemini Vision"],
            index=1,
        )

    def get_client_by_name(name: str) -> VisionClient:
        if name == "Google Gemini Vision":
            return GeminiVisionClient() if GEMINI_API_KEY else MockVisionClient(model_name="mock-gemini")
        elif name == "OpenAI GPT-4o":
            return GPT4VisionClient()
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

        if st.button("⚖️ Jalankan Komparasi Provider", type="primary"):
            with st.spinner("Menjalankan komparasi kedua provider..."):
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

# TAB 4: GALERI KOLEKSI, STATISTIK COUNTER & EKSPOR CSV
with tab_gallery:
    st.subheader("📚 Galeri Koleksi, Pencarian & Moderasi")
    report = st.session_state.moderator.generate_moderation_report(st.session_state.collection)
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Foto di Koleksi", report["total_photos"])
    with col_m2:
        st.metric("Foto Aman (Safe)", report["safe_count"])
    with col_m3:
        st.metric("Foto Unsafe / Bahaya", report["unsafe_count"])
    with col_m4:
        st.metric("Kepatuhan Keamanan", f"{report['safe_rate_percentage']}%")

    st.markdown("---")

    col_search, col_filter, col_export = st.columns([2, 2, 2])

    with col_search:
        search_tag = st.text_input("🔍 Cari Berdasarkan Tag (cari_by_tag):", placeholder="misal: sunset, kopi")

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
            st.markdown("##### 🏷️ Top 5 Tag Paling Sering Muncul:")
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
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <b>{photo.kategori}</b> | <small>{photo.timestamp}</small><br>
                        <span class="{'badge-safe' if photo.is_safe else 'badge-unsafe'}">
                            {'AMAN' if photo.is_safe else 'UNSAFE'}
                        </span>
                        <p style="margin-top: 8px; font-size: 0.9rem;">{photo.caption}</p>
                        <small style="color: #64748B;">Tags: {', '.join(photo.tags[:4])}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.warning("Tidak ada foto yang cocok dengan kriteria pencarian atau filter.")
