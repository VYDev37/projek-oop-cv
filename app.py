"""
Module: app.py
Deskripsi: Antarmuka Dashboard Visual Berbasis Streamlit untuk:
           1. Live Stream Monitoring & Vehicle Counting (YOLO + ByteTrack)
           2. Log Pelanggaran & Galeri Bukti (Rider & Plate Crops)
           3. Validasi Dokumen SIM/STNK & Pencocokan Nopol (Gemini AI Multimodal)
           4. Analitik & Laporan Ekspor (CSV / Excel)
"""

import time
import tempfile
from pathlib import Path
from typing import Optional
import cv2
import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np

from config import (
    GEMINI_API_KEY,
    DEFAULT_GEMINI_MODEL,
    SAMPLES_DIR,
    EVIDENCE_DIR,
    COUNTING_LINE_Y_RATIO,
)
from database import db_manager
from pipeline import pipeline, TrafficMonitoringPipeline
from gemini_client import gemini_client

# ==============================================================================
# KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Sistem Cerdas Pemantauan Lalu Lintas & Keselamatan AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS untuk tampilan modern dan rapi
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #616161;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .badge-violation {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-success {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# SIDEBAR - PENGATURAN & KREDENSIAL
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3208/3208677.png", width=80)
    st.title("Panel Kontrol AI")
    st.markdown("---")

    # Status Gemini AI
    st.subheader("🔑 Google Gemini API")
    input_api_key = st.text_input(
        "Gemini API Key",
        value=GEMINI_API_KEY,
        type="password",
        help="Dapatkan API key di https://aistudio.google.com/",
    )

    if input_api_key:
        gemini_client.api_key = input_api_key
        gemini_client._init_client()
        st.success("🟢 Gemini API Terhubung")
    else:
        st.warning("🟡 Mode Simulasi (Tanpa API Key)")

    selected_model = st.selectbox(
        "Pilihan Model Multimodal",
        ["gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash"],
        index=0,
    )
    gemini_client.model_name = selected_model

    st.markdown("---")
    st.subheader("⚙️ Parameter Computer Vision")
    conf_thresh = st.slider("YOLO Confidence", 0.1, 0.9, 0.35, 0.05)
    pipeline.detector.conf_threshold = conf_thresh

    line_pos = st.slider("Garis Hitung Virtual (% Tinggi)", 30, 90, int(COUNTING_LINE_Y_RATIO * 100), 5)
    line_ratio = line_pos / 100.0

    st.markdown("---")
    if st.button("🔄 Reset Penghitung & Database"):
        pipeline.detector.reset_counters()
        st.toast("Penghitung berhasil di-reset!", icon="✅")

# ==============================================================================
# HEADER UTAMA
# ==============================================================================
st.markdown('<div class="main-title">🚦 Sistem Cerdas Pemantauan Lalu Lintas & Keselamatan Pengendara</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Computer Vision (YOLOv8 + ByteTrack) • OCR Plat Nomor • Validasi Multimodal Google Gemini AI • SQLite Database</div>',
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "🎥 Live Stream & Monitoring",
    "🚨 Log Pelanggaran & Bukti",
    "📄 Validasi Dokumen (SIM/STNK)",
    "📊 Statistik Lalu Lintas",
])

# ==============================================================================
# TAB 1: LIVE STREAM & MONITORING
# ==============================================================================
with tabs[0]:
    st.subheader("Pemantauan Lalu Lintas Real-Time & Inspeksi ETLE")

    col_ctrl, col_stats = st.columns([2, 1])

    with col_ctrl:
        source_type = st.radio(
            "Pilih Sumber Pemantauan:",
            [
                "Demo Video Sampel",
                "Unggah Berkas Video",
                "Webcam / Kamera (Live)",
                "Unggah Foto (Gambar)",
                "Ambil Foto Langsung (Kamera)",
            ],
            horizontal=True,
        )

        is_image_mode = source_type in ("Unggah Foto (Gambar)", "Ambil Foto Langsung (Kamera)")
        static_image_to_process: Optional[np.ndarray] = None
        video_path: Optional[str] = None
        run_monitoring: bool = False

        if is_image_mode:
            if source_type == "Unggah Foto (Gambar)":
                uploaded_img = st.file_uploader(
                    "Pilih Berkas Foto Kendaraan / Pengendara (JPG/PNG)",
                    type=["jpg", "jpeg", "png"],
                )
                if uploaded_img is not None:
                    pil_img = Image.open(uploaded_img)
                    static_image_to_process = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    st.image(pil_img, caption="Foto yang Diunggah", use_container_width=True)

            elif source_type == "Ambil Foto Langsung (Kamera)":
                camera_snap = st.camera_input("Ambil Foto dari Kamera Laptop / HP")
                if camera_snap is not None:
                    pil_img = Image.open(camera_snap)
                    static_image_to_process = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    st.image(pil_img, caption="Hasil Jepretan Kamera", use_container_width=True)

        else:
            if source_type == "Demo Video Sampel":
                sample_mp4 = SAMPLES_DIR / "sample_traffic.mp4"
                if sample_mp4.exists():
                    video_path = str(sample_mp4)
                else:
                    st.info("Sampel video demo belum ada, sistem akan menggunakan file otomatis.")
            elif source_type == "Unggah Berkas Video":
                uploaded_video = st.file_uploader("Pilih Berkas Video (MP4/AVI/MOV)", type=["mp4", "avi", "mov"])
                if uploaded_video is not None:
                    file_ext = Path(uploaded_video.name).suffix or ".mp4"
                    temp_video_path = EVIDENCE_DIR / f"temp_upload{file_ext}"
                    with open(temp_video_path, "wb") as f:
                        f.write(uploaded_video.getbuffer())
                    video_path = str(temp_video_path)
            else:
                # Webcam index
                video_path = 0

            run_monitoring = st.checkbox("Mulai Pemantauan Video", value=False)

    with col_stats:
        stats = db_manager.get_system_statistics()
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>Ringkasan Pemantauan</h4>
                <p><b>Total Kendaraan:</b> {max(stats['total_vehicles'], pipeline.detector.total_counted)}</p>
                <p><b>Total Pelanggaran:</b> {max(stats['total_violations'], len(pipeline.recent_alerts))}</p>
                <p><b>Tingkat Kepatuhan Helm:</b> {stats['compliance_rate']}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # A. MODE CITRA STATIS (INSPEKSI ETLE SNAPSHOT)
    if is_image_mode:
        if static_image_to_process is not None:
            if st.button("🔍 Proses Inspeksi ETLE (Deteksi & AI)", type="primary"):
                with st.spinner("Menjalankan analisis Computer Vision & Gemini AI..."):
                    ann_img, inspections, summary = pipeline.process_static_image(static_image_to_process)

                    st.success(
                        f"Inspeksi Selesai! Terdeteksi {summary['total_vehicles']} kendaraan, "
                        f"{summary['total_violations']} pelanggaran. Kepatuhan: {summary['compliance_rate']}%."
                    )

                    col_res_img, col_res_cards = st.columns([3, 2])
                    with col_res_img:
                        st.image(
                            cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB),
                            caption="Hasil Deteksi & Anotasi Bounding Box",
                            use_container_width=True,
                        )

                    with col_res_cards:
                        st.write("#### 📋 Hasil Investigasi Tiap Kendaraan")
                        if inspections:
                            for item in inspections:
                                is_viol = item["violation_detected"]
                                bg_color = "#ffebee" if is_viol else "#e8f5e9"
                                border_color = "#ffcdd2" if is_viol else "#c8e6c9"
                                badge_class = "badge-violation" if is_viol else "badge-success"
                                status_label = "TILANG ELEKTRONIK" if is_viol else "PATUH ATURAN"

                                st.markdown(
                                    f"""
                                    <div style="border: 1px solid {border_color}; background: {bg_color}; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                                        <span class="{badge_class}">{status_label}</span>
                                        <span style="float:right; font-size:0.8rem; color:#666;">ID #{item['track_id']}</span>
                                        <br><b>Kendaraan:</b> {item['vehicle_type'].upper()}
                                        <br><b>Plat Nomor (OCR):</b> <code>{item['license_plate']}</code>
                                        <br><b>Status Helm:</b> {item['helmet_status']}
                                        <br><b>Keparahan:</b> {item['severity']} | <b>Tindakan:</b> {item['recommended_action']}
                                        <br><small style="color:#444;"><b>Catatan:</b> {item['officer_notes']}</small>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.info("Tidak ada kendaraan yang terdeteksi pada citra ini.")
        else:
            st.info("Silakan unggah gambar atau ambil foto menggunakan kamera di atas untuk memulai inspeksi.")

    # B. MODE VIDEO STREAMING
    else:
        col_video, col_feed = st.columns([3, 2])

        with col_video:
            video_placeholder = st.empty()

        with col_feed:
            st.write("#### 🔔 Umpan Pelanggaran Langsung")
            feed_placeholder = st.empty()

        if run_monitoring and video_path is not None:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error(f"Gagal membuka sumber video: {video_path}")
            else:
                fps_timer = time.time()
                frame_count = 0

                while cap.isOpened() and run_monitoring:
                    ret, frame = cap.read()
                    if not ret:
                        # Loop video jika file
                        if isinstance(video_path, str) and not source_type.startswith("Webcam"):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        break

                    # Frame processing via pipeline
                    annotated_frame, alerts, meta = pipeline.process_frame(frame, line_ratio=line_ratio)

                    # Hitung FPS
                    frame_count += 1
                    curr_time = time.time()
                    fps = frame_count / (curr_time - fps_timer) if (curr_time - fps_timer) > 0 else 0

                    # Tampilkan di Streamlit
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(
                        frame_rgb,
                        caption=f"Real-Time Stream | FPS: {fps:.1f} | Garis: {int(line_ratio*100)}%",
                        use_container_width=True,
                    )

                    # Perbarui umpan alert pelanggaran
                    with feed_placeholder.container():
                        if alerts:
                            for alert in alerts[:4]:
                                st.markdown(
                                    f"""
                                    <div style="border: 1px solid #ffcdd2; background: #ffebee; border-radius: 6px; padding: 10px; margin-bottom: 8px;">
                                        <span class="badge-violation">{alert.violation_type}</span>
                                        <span style="float:right; font-size:0.8rem; color:#666;">{alert.timestamp}</span>
                                        <br><b>Kendaraan:</b> {alert.vehicle_type} (Track #{alert.track_id})
                                        <br><b>Plat Nomor (OCR):</b> <code>{alert.license_plate}</code>
                                        <br><b>Keparahan:</b> {alert.severity} | <b>Tindakan:</b> {alert.recommended_action}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.info("Belum ada pelanggaran terdeteksi pada aliran saat ini.")

                    time.sleep(0.03)

                cap.release()

# ==============================================================================
# TAB 2: LOG PELANGGARAN & BUKTI
# ==============================================================================
with tabs[1]:
    st.subheader("Buku Register Pelanggaran & Galeri Barang Bukti")

    col_btn1, col_btn2, col_filter = st.columns([1, 1, 2])
    with col_btn1:
        if st.button("📥 Ekspor ke CSV"):
            csv_path = db_manager.export_violations_to_csv()
            st.success(f"Terekspor: {csv_path.name}")
    with col_btn2:
        if st.button("📊 Ekspor ke Excel (.xlsx)"):
            xls_path = db_manager.export_violations_to_excel()
            st.success(f"Terekspor: {xls_path.name}")

    df_violations = db_manager.get_violations_dataframe(limit=100)

    if not df_violations.empty:
        st.dataframe(
            df_violations[[
                "id", "timestamp", "track_id", "vehicle_type", "license_plate",
                "violation_type", "helmet_status", "severity", "status_tilang"
            ]],
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("🔍 Peninjauan Bukti Forensik Pelanggaran")
        selected_id = st.selectbox("Pilih ID Pelanggaran untuk Melihat Bukti:", df_violations["id"].tolist())

        if selected_id:
            row = df_violations[df_violations["id"] == selected_id].iloc[0]

            col_ev1, col_ev2, col_ev3 = st.columns([1.5, 1.5, 2])

            with col_ev1:
                st.write("**Foto Bukti Pengendara:**")
                r_path = row["evidence_rider_path"]
                if r_path and Path(r_path).exists():
                    st.image(str(r_path), use_container_width=True)
                else:
                    st.info("Citra bukti pengendara tidak tersedia.")

            with col_ev2:
                st.write("**Foto Bukti Plat Nomor:**")
                p_path = row["evidence_plate_path"]
                if p_path and Path(p_path).exists():
                    st.image(str(p_path), use_container_width=True)
                else:
                    st.info("Citra bukti plat tidak tersedia.")

            with col_ev3:
                st.write("**Detail Analisis Gemini AI:**")
                st.write(f"- **Nomor Polisi (OCR):** `{row['license_plate']}`")
                st.write(f"- **Jenis Pelanggaran:** {row['violation_type']}")
                st.write(f"- **Tingkat Keparahan:** {row['severity']}")
                st.write(f"- **Status Helm:** {row['helmet_status']}")
                st.write(f"- **Confidence Score:** {row['gemini_confidence']}")
                st.write(f"- **Status Penindakan:** {row['status_tilang']}")

                # Detail JSON
                raw_json = row["gemini_safety_validation"]
                if raw_json:
                    with st.expander("Lihat Payload Structured JSON Gemini"):
                        st.json(raw_json)
    else:
        st.info("Belum ada data pelanggaran di database. Jalankan pemantauan video terlebih dahulu.")

# ==============================================================================
# TAB 3: VALIDASI DOKUMEN (SIM/STNK) & PENCOCOKAN NOPOL
# ==============================================================================
with tabs[2]:
    st.subheader("Verifikasi Dokumen Berkendara & Cross-Check Nopol Kamera")
    st.markdown("Gunakan model multimodal Gemini AI untuk mengekstrak data dari SIM/STNK dan mencocokkannya dengan nomor plat yang tertangkap kamera.")

    col_doc_in, col_doc_out = st.columns([1, 1])

    with col_doc_in:
        doc_source = st.radio(
            "Pilih Sampel atau Unggah Sendiri:",
            ["Sampel STNK Sesuai (Match)", "Sampel STNK Kedaluwarsa (Expired)", "Unggah Gambar Dokumen"],
            horizontal=True,
        )

        doc_image_to_process = None
        target_plate_input = st.text_input("Nomor Plat Kendaraan Hasil Tangkapan Kamera:", value="B 1234 ABC")

        if doc_source == "Sampel STNK Sesuai (Match)":
            s_path = SAMPLES_DIR / "sample_stnk_cocok.jpg"
            if s_path.exists():
                doc_image_to_process = str(s_path)
                st.image(doc_image_to_process, caption="Sampel Dokumen STNK Cocok", use_container_width=True)
        elif doc_source == "Sampel STNK Kedaluwarsa (Expired)":
            s_path = SAMPLES_DIR / "sample_stnk_expired.jpg"
            if s_path.exists():
                doc_image_to_process = str(s_path)
                st.image(doc_image_to_process, caption="Sampel Dokumen STNK Expired", use_container_width=True)
        else:
            uploaded_doc = st.file_uploader("Unggah Foto SIM atau STNK", type=["jpg", "jpeg", "png"])
            if uploaded_doc:
                doc_image_to_process = Image.open(uploaded_doc)
                st.image(doc_image_to_process, caption="Dokumen Diunggah", use_container_width=True)

        verify_button = st.button("🔍 Verifikasi dengan Gemini AI", type="primary")

    with col_doc_out:
        if verify_button and doc_image_to_process is not None:
            with st.spinner("Memproses analisis multimodal Gemini AI..."):
                doc_res, match_status, row_id = pipeline.verify_document(
                    doc_image_to_process, target_detected_plate=target_plate_input
                )

                st.success("Analisis Dokumen Selesai!")

                # Kartu Hasil
                st.markdown("### 📋 Hasil Ekstraksi Dokumen")
                st.write(f"- **Jenis Dokumen:** `{doc_res.document_type}`")
                st.write(f"- **Nomor Dokumen:** `{doc_res.document_number}`")
                st.write(f"- **Nama Pemilik:** **{doc_res.holder_name}**")
                st.write(f"- **Plat Terdaftar di Dokumen:** `{doc_res.registered_plate_number}`")
                st.write(f"- **Tipe Kendaraan:** {doc_res.vehicle_type_stated}")
                st.write(f"- **Masa Berlaku:** `{doc_res.expiry_date}`")

                st.markdown("---")
                st.markdown("### ⚖️ Status Validasi & Keabsahan Dokumen")

                # Guardrail: Cek apakah citra merupakan dokumen resmi SIM/STNK yang valid
                is_invalid_doc = (not doc_res.is_document_valid) or (doc_res.document_type in ("TIDAK_DIKETAHUI", "TIDAK_VALID"))

                if is_invalid_doc:
                    st.markdown('<span class="badge-violation">🚫 DOKUMEN TIDAK VALID / BUKAN SIM/STNK</span>', unsafe_allow_html=True)
                    st.error("❌ Berkas yang diunggah DITOLAK oleh sistem karena bukan merupakan dokumen resmi SIM atau STNK yang sah.")
                else:
                    st.markdown('<span class="badge-success">📄 DOKUMEN RESMI TERVERIFIKASI</span>', unsafe_allow_html=True)
                    st.write("")

                    # 1. Match status badge
                    if match_status == "MATCH":
                        st.markdown('<span class="badge-success">✅ PLAT COCOK (MATCH)</span>', unsafe_allow_html=True)
                    elif match_status == "MISMATCH":
                        st.markdown('<span class="badge-violation">❌ PLAT TIDAK COCOK (MISMATCH)</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge-violation">⚠️ PLAT TIDAK DAPAT DIVERIFIKASI</span>', unsafe_allow_html=True)

                    st.write("")
                    # 2. Expiry status
                    if doc_res.is_expired:
                        st.markdown('<span class="badge-violation">⚠️ DOKUMEN KEDALUWARSA / MATI PAJAK</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge-success">✅ MASA BERLAKU AKTIF</span>', unsafe_allow_html=True)

                st.write("")
                st.info(f"**Catatan Petugas:** {doc_res.notes}")

# ==============================================================================
# TAB 4: STATISTIK & ANALITIK LALU LINTAS
# ==============================================================================
with tabs[3]:
    st.subheader("Statistik Arus Lalu Lintas & Kepatuhan")

    stats = db_manager.get_system_statistics()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Kendaraan", stats["total_vehicles"])
    m2.metric("Total Pelanggaran", stats["total_violations"])
    m3.metric("Kepatuhan Helm", f"{stats['compliance_rate']}%")
    m4.metric("Kendaraan Motor", stats["vehicles_by_type"].get("motorcycle", 0))

    st.markdown("---")
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.write("#### Komposisi Jenis Kendaraan")
        if stats["vehicles_by_type"]:
            v_df = pd.DataFrame(
                list(stats["vehicles_by_type"].items()), columns=["Jenis Kendaraan", "Jumlah"]
            ).set_index("Jenis Kendaraan")
            st.bar_chart(v_df)
        else:
            st.info("Belum ada data kendaraan terhitung.")

    with col_c2:
        st.write("#### Distribusi Kategori Pelanggaran")
        if stats["violations_by_type"]:
            viol_df = pd.DataFrame(
                list(stats["violations_by_type"].items()), columns=["Jenis Pelanggaran", "Jumlah"]
            ).set_index("Jenis Pelanggaran")
            st.bar_chart(viol_df)
        else:
            st.info("Belum ada data pelanggaran tercatat.")
