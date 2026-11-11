"""
Module: app.py
Deskripsi: Antarmuka Dashboard Visual Berbasis Streamlit untuk:
           1. Live Stream Monitoring & Vehicle Counting (YOLO + ByteTrack) serta Inspeksi Foto Statis
           2. Log Pelanggaran & Galeri Bukti (Rider & Plate Crops)
           3. Validasi Dokumen SIM/STNK & Pencocokan Nopol (Gemini AI Multimodal)
           4. Statistik Lalu Lintas, Galeri Koleksi (Iterator Pattern), Moderasi Konten & Ekspor CSV
"""

import html
import time
from pathlib import Path
from typing import Optional, List
from datetime import datetime
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
    EXPORTS_DIR,
    COUNTING_LINE_Y_RATIO,
)
from database import db_manager
from pipeline import pipeline, TrafficMonitoringPipeline
from gemini_client import gemini_client
from vision_client import (
    VisionClient,
    YOLOv8VisionClient,
    GeminiVisionClient,
    MockVisionClient,
    ImageAnalysisService,
    PhotoAnalysisResult,
    VisionAPIException,
)
from photo_collection import (
    PhotoCollection,
    PhotoCollectionIterator,
    ContentModerator,
    CollectionCSVExporter,
)

# --- Inisialisasi State Sesi Aplikasi ---
if "collection" not in st.session_state:
    st.session_state.collection = PhotoCollection()

if "moderator" not in st.session_state:
    st.session_state.moderator = ContentModerator()

if "exporter" not in st.session_state:
    st.session_state.exporter = CollectionCSVExporter()

if "service" not in st.session_state:
    st.session_state.service = ImageAnalysisService(strategy=YOLOv8VisionClient())

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Sistem Cerdas Pemantauan Lalu Lintas & Keselamatan AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS untuk tampilan modern, rapi, dan responsif
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
    .tag-pill {
        display: inline-block;
        background: #e3f2fd;
        color: #1565c0;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px 3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar: Pengaturan, Strategy AI & Parameter ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3208/3208677.png", width=70)
    st.title("Panel Kontrol AI")
    st.markdown("---")

    # Status Gemini API
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

    selected_gemini_model = st.selectbox(
        "Pilihan Model Gemini",
        ["gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash"],
        index=0,
    )
    gemini_client.model_name = selected_gemini_model

    st.markdown("---")
    # Strategy Pattern Selector
    st.subheader("🎯 Strategi Vision AI (Strategy Pattern)")
    provider_choice = st.selectbox(
        "Pilih Engine Vision AI:",
        [
            "YOLOv8 + EasyOCR (Local Computer Vision)",
            "Google Gemini Vision (Cloud Multimodal AI)",
            "Mock Vision Engine (Offline Deterministic)",
        ],
        index=0,
    )

    new_strategy: VisionClient
    if provider_choice.startswith("YOLOv8"):
        new_strategy = YOLOv8VisionClient()
    elif provider_choice.startswith("Google Gemini"):
        new_strategy = GeminiVisionClient(api_key=input_api_key, model_name=selected_gemini_model)
    else:
        new_strategy = MockVisionClient()

    try:
        st.session_state.service.set_strategy(new_strategy)
    except TypeError:
        st.session_state.service = ImageAnalysisService(strategy=new_strategy)

    st.info(f"Strategi Aktif: **{st.session_state.service.strategy.__class__.__name__}**")

    st.markdown("---")
    # Parameter Computer Vision
    st.subheader("⚙️ Parameter Computer Vision")
    conf_thresh = st.slider("YOLO Confidence", 0.1, 0.9, 0.25, 0.05)
    pipeline.detector.conf_threshold = conf_thresh

    line_pos = st.slider("Garis Hitung Virtual (% Tinggi)", 30, 90, int(COUNTING_LINE_Y_RATIO * 100), 5)
    line_ratio = line_pos / 100.0

    st.markdown("---")
    # Ringkasan Koleksi
    st.subheader("📚 Status Koleksi Foto")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric("Total Foto", len(st.session_state.collection))
    with col_sb2:
        mod_report = st.session_state.moderator.generate_moderation_report(st.session_state.collection)
        st.metric("Tertib / Safe", f"{mod_report['safe_rate_percentage']}%")

    if st.button("🔄 Reset Penghitung & Koleksi", use_container_width=True):
        pipeline.detector.reset_counters()
        st.session_state.collection.clear()
        st.toast("Penghitung dan koleksi berhasil di-reset!", icon="✅")
        st.rerun()

# --- Header Utama ---
st.markdown('<div class="main-title">🚦 Sistem Cerdas Pemantauan Lalu Lintas & Keselamatan Pengendara</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Computer Vision (YOLOv8 + ByteTrack) • OCR Plat Nomor • Validasi Multimodal Google Gemini AI • Arsitektur Object-Oriented Programming</div>',
    unsafe_allow_html=True,
)

# 4 Tab Navigasi Utama
tabs = st.tabs([
    "🎥 Live Stream & Monitoring",
    "🚨 Log Pelanggaran & Bukti",
    "📄 Validasi Dokumen (SIM/STNK)",
    "📊 Statistik & Galeri Koleksi",
])

# --- Tab 1: Live Stream & Monitoring ---
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
        static_pil_image: Optional[Image.Image] = None
        static_filename_label: str = "foto_input.jpg"
        video_path: Optional[str] = None
        run_monitoring: bool = False

        if is_image_mode:
            if source_type == "Unggah Foto (Gambar)":
                uploaded_img = st.file_uploader(
                    "Pilih Berkas Foto Kendaraan / Pengendara (JPG/PNG)",
                    type=["jpg", "jpeg", "png"],
                )
                if uploaded_img is not None:
                    static_pil_image = Image.open(uploaded_img).convert("RGB")
                    static_filename_label = uploaded_img.name
                    static_image_to_process = cv2.cvtColor(np.array(static_pil_image), cv2.COLOR_RGB2BGR)
                    st.image(static_pil_image, caption=f"Foto yang Diunggah: {static_filename_label}", use_container_width=True)

            elif source_type == "Ambil Foto Langsung (Kamera)":
                camera_snap = st.camera_input("Ambil Foto dari Kamera Laptop / HP")
                if camera_snap is not None:
                    static_pil_image = Image.open(camera_snap).convert("RGB")
                    static_filename_label = f"camera_snap_{datetime.now().strftime('%H%M%S')}.jpg"
                    static_image_to_process = cv2.cvtColor(np.array(static_pil_image), cv2.COLOR_RGB2BGR)
                    st.image(static_pil_image, caption="Hasil Jepretan Kamera", use_container_width=True)

        else:
            if source_type == "Demo Video Sampel":
                sample_mp4 = SAMPLES_DIR / "sample_traffic.mp4"
                if sample_mp4.exists():
                    video_path = str(sample_mp4)
                else:
                    st.info("Sampel video demo belum ada, silakan periksa direktori data/samples.")
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
                <p><b>Koleksi Foto Teranalisis:</b> {len(st.session_state.collection)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Mode Citra Statis (Inspeksi ETLE Snapshot)
    if is_image_mode:
        if static_image_to_process is not None and static_pil_image is not None:
            if st.button("🔍 Proses Inspeksi ETLE (Deteksi & AI)", type="primary"):
                with st.spinner(f"Menjalankan analisis via {st.session_state.service.strategy.__class__.__name__}..."):
                    # Jalankan pipeline Computer Vision
                    ann_img, inspections, summary = pipeline.process_static_image(static_image_to_process)

                    # Simpan hasil ke PhotoAnalysisResult dan masukkan ke koleksi
                    try:
                        oop_result = st.session_state.service.analyze_image(
                            static_pil_image, image_name=static_filename_label
                        )
                        st.session_state.last_result = oop_result
                        st.session_state.collection.tambah(oop_result)
                    except Exception as e:
                        st.warning(f"Catatan integrasi OOP: {e}")

                    analysis_mode = summary.get("analysis_mode", "Computer Vision")
                    st.success(
                        f"Inspeksi Selesai ({analysis_mode})! Terdeteksi {summary.get('total_vehicles', 0)} kendaraan, "
                        f"{summary.get('total_violations', 0)} pelanggaran. Hasil otomatis disimpan ke Koleksi!"
                    )
                    if summary.get("scene_summary"):
                        st.info(f"🧠 **Ringkasan Adegan Gemini (Full-Image):** {summary['scene_summary']}")

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

                                source_tag = html.escape(" ".join(str(item.get("analysis_source", "ETLE Pipeline")).split()))
                                role_desc = html.escape(" ".join(str(item.get("rider_role_description", "")).split()))
                                reasons = item.get("violation_reasons") or []
                                reasons_text = "; ".join(" ".join(str(r).split()) for r in reasons if r)

                                v_type = html.escape(str(item["vehicle_type"]).upper())
                                plate_str = html.escape(" ".join(str(item["license_plate"]).split()))
                                helmet_str = html.escape(str(item["helmet_status"]))
                                sev_str = html.escape(str(item["severity"]))
                                act_str = html.escape(str(item["recommended_action"]))
                                notes_str = html.escape(" ".join(str(item["officer_notes"]).split()))

                                card_parts = [
                                    f'<div style="border: 1px solid {border_color}; background: {bg_color}; border-radius: 8px; padding: 12px; margin-bottom: 10px;">',
                                    f'<span class="{badge_class}">{status_label}</span>',
                                    f'<span style="float:right; font-size:0.8rem; color:#666;">ID #{item["track_id"]} • {source_tag}</span>',
                                ]
                                if role_desc:
                                    card_parts.append(f'<br><small><b>Posisi/Ciri:</b> {role_desc}</small>')
                                card_parts.extend([
                                    f'<br><b>Kendaraan:</b> {v_type}',
                                    f'<br><b>Plat Nomor (OCR/AI):</b> <code>{plate_str}</code>',
                                    f'<br><b>Status Helm:</b> {helmet_str}',
                                    f'<br><b>Keparahan:</b> {sev_str} | <b>Tindakan:</b> {act_str}',
                                ])
                                if reasons_text:
                                    card_parts.append(
                                        f'<br><small style="color:#b71c1c;"><b>Alasan:</b> {html.escape(reasons_text)}</small>'
                                    )
                                card_parts.extend([
                                    f'<br><small style="color:#444;"><b>Catatan:</b> {notes_str}</small>',
                                    '</div>',
                                ])

                                st.markdown("\n".join(card_parts), unsafe_allow_html=True)
                        else:
                            st.warning(
                                f"⚠️ Tidak ada kendaraan yang terdeteksi oleh YOLO (confidence ≥ {conf_thresh:.2f}) maupun Gemini Full-Image.\n\n"
                                "💡 **Tips:** Turunkan slider **YOLO Confidence** di panel kiri ke **0.20 - 0.25** dan pastikan **GEMINI_API_KEY** aktif untuk analisis adegan penuh."
                            )
                            # Jika tidak ada bounding box kendaraan utuh (misal foto plat jarak dekat, dokumen STNK, atau analisis foto umum)
                            if st.session_state.last_result:
                                res = st.session_state.last_result
                                is_safe = res.is_safe
                                bg_color = "#e8f5e9" if is_safe else "#ffebee"
                                border_color = "#c8e6c9" if is_safe else "#ffcdd2"
                                badge_class = "badge-success" if is_safe else "badge-violation"
                                status_label = "TERTIB / SAFE" if is_safe else "PELANGGARAN / UNSAFE"

                                raw_ocr = res.ocr_text or "-"
                                is_watermark = any(w in raw_ocr.lower() for w in ["www.", ".com", ".id", "http"])
                                label_field = "Teks Terdeteksi (Watermark/Umum)" if is_watermark else "Teks / Plat Terbaca"

                                fallback_card = "\n".join([
                                    f'<div style="border: 1px solid {border_color}; background: {bg_color}; border-radius: 8px; padding: 12px; margin-bottom: 10px;">',
                                    f'<span class="{badge_class}">{status_label}</span>',
                                    f'<br><b>Engine Analisis:</b> {html.escape(" ".join(str(res.provider).split()))}',
                                    f'<br><b>Kategori:</b> {html.escape(" ".join(str(res.kategori).split()))}',
                                    f'<br><b>{label_field}:</b> <code>{html.escape(" ".join(str(raw_ocr).split()))}</code>',
                                    f'<br><b>Objek Terdeteksi:</b> {html.escape(", ".join(res.objects))}',
                                    f'<br><b>Keterangan:</b> {html.escape(" ".join(str(res.caption).split()))}',
                                    '</div>',
                                ])
                                st.markdown(fallback_card, unsafe_allow_html=True)
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
                        if isinstance(video_path, str) and not str(source_type).startswith("Webcam"):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        break

                    # Frame processing via pipeline terintegrasi
                    annotated_frame, alerts, meta = pipeline.process_frame(frame, line_ratio=line_ratio)

                    # Hitung FPS
                    frame_count += 1
                    curr_time = time.time()
                    fps = frame_count / (curr_time - fps_timer) if (curr_time - fps_timer) > 0 else 0

                    # Tampilkan di Streamlit
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(
                        frame_rgb,
                        caption=f"Real-Time Stream | FPS: {fps:.1f} | Garis Hitung: {int(line_ratio*100)}%",
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

                    time.sleep(0.02)

                cap.release()

# --- Tab 2: Log Pelanggaran & Bukti ---
with tabs[1]:
    st.subheader("Buku Register Pelanggaran & Galeri Barang Bukti")

    col_btn1, col_btn2, col_space = st.columns([1, 1, 2])
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
                st.write("**Detail Analisis AI Keselamatan:**")
                st.write(f"- **Nomor Polisi (OCR):** `{row['license_plate']}`")
                st.write(f"- **Jenis Pelanggaran:** {row['violation_type']}")
                st.write(f"- **Tingkat Keparahan:** {row['severity']}")
                st.write(f"- **Status Helm:** {row['helmet_status']}")
                st.write(f"- **Confidence Score:** {row['gemini_confidence']}")
                st.write(f"- **Status Penindakan:** {row['status_tilang']}")

                raw_json = row["gemini_safety_validation"]
                if raw_json:
                    with st.expander("Lihat Payload Structured JSON Validation"):
                        st.json(raw_json)
    else:
        st.info("Belum ada data pelanggaran di database. Jalankan pemantauan video terlebih dahulu.")

# --- Tab 3: Validasi Dokumen (SIM/STNK) & Pencocokan Nopol ---
with tabs[2]:
    st.subheader("Verifikasi Dokumen Berkendara & Cross-Check Nopol Kamera")
    st.markdown("Mengekstrak data dari dokumen resmi (SIM/STNK) dan membandingkan kecocokan nomor polisi dengan tangkapan kamera.")

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

        verify_button = st.button("🔍 Verifikasi Keabsahan Dokumen", type="primary")

    with col_doc_out:
        if verify_button and doc_image_to_process is not None:
            with st.spinner("Memproses analisis dokumen dan pencocokan nopol..."):
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

                is_invalid_doc = (not doc_res.is_document_valid) or (doc_res.document_type in ("TIDAK_DIKETAHUI", "TIDAK_VALID"))

                if is_invalid_doc:
                    st.markdown('<span class="badge-violation">🚫 DOKUMEN TIDAK VALID / BUKAN SIM/STNK</span>', unsafe_allow_html=True)
                    st.error("❌ Berkas yang diunggah DITOLAK karena bukan merupakan dokumen resmi SIM atau STNK yang sah.")
                else:
                    st.markdown('<span class="badge-success">📄 DOKUMEN RESMI TERVERIFIKASI</span>', unsafe_allow_html=True)
                    st.write("")

                    # Match status badge
                    if match_status == "MATCH":
                        st.markdown('<span class="badge-success">✅ PLAT COCOK (MATCH)</span>', unsafe_allow_html=True)
                    elif match_status == "MISMATCH":
                        st.markdown('<span class="badge-violation">❌ PLAT TIDAK COCOK (MISMATCH)</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge-violation">⚠️ PLAT TIDAK DAPAT DIVERIFIKASI</span>', unsafe_allow_html=True)

                    st.write("")
                    # Expiry status
                    if doc_res.is_expired:
                        st.markdown('<span class="badge-violation">⚠️ DOKUMEN KEDALUWARSA / MATI PAJAK</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge-success">✅ MASA BERLAKU AKTIF</span>', unsafe_allow_html=True)

                st.write("")
                st.info(f"**Catatan Petugas:** {doc_res.notes}")

# --- Tab 4: Statistik & Galeri Koleksi ---
with tabs[3]:
    st.subheader("Statistik Arus Lalu Lintas & Galeri Koleksi")

    # Statistik Umum Database
    stats = db_manager.get_system_statistics()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Kendaraan Terhitung", stats["total_vehicles"])
    m2.metric("Total Pelanggaran", stats["total_violations"])
    m3.metric("Kepatuhan Helm", f"{stats['compliance_rate']}%")
    m4.metric("Koleksi Foto", len(st.session_state.collection))

    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.write("#### 🚗 Komposisi Jenis Kendaraan")
        if stats["vehicles_by_type"]:
            v_df = pd.DataFrame(
                list(stats["vehicles_by_type"].items()), columns=["Jenis Kendaraan", "Jumlah"]
            ).set_index("Jenis Kendaraan")
            st.bar_chart(v_df)
        else:
            st.info("Belum ada data kendaraan terhitung.")

    with col_chart2:
        st.write("#### ⚠️ Distribusi Kategori Pelanggaran")
        if stats["violations_by_type"]:
            viol_df = pd.DataFrame(
                list(stats["violations_by_type"].items()), columns=["Jenis Pelanggaran", "Jumlah"]
            ).set_index("Jenis Pelanggaran")
            st.bar_chart(viol_df)
        else:
            st.info("Belum ada data pelanggaran tercatat.")

    st.markdown("---")
    # Galeri Koleksi Foto
    st.subheader("📚 Galeri Koleksi Foto & Audit Moderasi")

    col_oop_ctrl1, col_oop_ctrl2, col_oop_ctrl3 = st.columns([1.5, 1.5, 1])

    with col_oop_ctrl1:
        search_tag = st.text_input("🔍 Cari Berdasarkan Tag:", "")
    with col_oop_ctrl2:
        category_options = ["Semua Kategori", "Patuhi Aturan (Tertib Lalu Lintas)", "Pelanggaran Helm / Keselamatan", "Plat Nomor Kendaraan (ETLE)", "Dokumen STNK (Verifikasi Korlantas)", "Umum"]
        selected_cat = st.selectbox("📂 Filter Kategori:", category_options)
    with col_oop_ctrl3:
        st.write("")
        st.write("")
        if st.button("📥 Ekspor Koleksi ke CSV", use_container_width=True):
            if len(st.session_state.collection) > 0:
                out_csv = EXPORTS_DIR / f"koleksi_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                exported = st.session_state.exporter.export_to_csv(st.session_state.collection, out_csv)
                st.success(f"Terekspor: `{Path(exported).name}`")
            else:
                st.warning("Koleksi masih kosong. Lakukan analisis citra terlebih dahulu.")

    # Tag Terpopuler via Counter
    popular_tags = st.session_state.collection.get_popular_tags(top_n=8)
    if popular_tags:
        st.write("**🔖 Tag Paling Populer (collections.Counter):**")
        tag_html = " ".join([f'<span class="tag-pill">#{t} ({c})</span>' for t, c in popular_tags])
        st.markdown(tag_html, unsafe_allow_html=True)
        st.write("")

    # Evaluasi daftar foto yang ditampilkan
    photos_to_display: List[PhotoAnalysisResult] = []
    if search_tag.strip():
        photos_to_display = st.session_state.collection.cari_by_tag(search_tag)
    elif selected_cat != "Semua Kategori":
        photos_to_display = st.session_state.collection.filter_kategori(selected_cat)
    else:
        # Gunakan Iterator Pattern (__iter__ & __next__)
        photos_to_display = [photo for photo in st.session_state.collection]

    if photos_to_display:
        st.write(f"Menampilkan **{len(photos_to_display)}** foto:")
        # Render grid galeri
        cols = st.columns(3)
        for idx, photo in enumerate(photos_to_display):
            with cols[idx % 3]:
                badge_cls = "badge-success" if photo.is_safe else "badge-violation"
                badge_lbl = "TERTIB / SAFE" if photo.is_safe else "PELANGGARAN / UNSAFE"

                img_path_obj = Path(photo.annotated_image_path or photo.image_path)
                if img_path_obj.exists():
                    st.image(str(img_path_obj), use_container_width=True)

                st.markdown(
                    f"""
                    <div style="border: 1px solid #e0e0e0; border-radius: 6px; padding: 10px; margin-bottom: 12px; background: #fafafa;">
                        <span class="{badge_cls}">{badge_lbl}</span>
                        <br><b>Kategori:</b> {photo.kategori}
                        <br><b>Plat OCR:</b> <code>{photo.ocr_text or '-'}</code>
                        <br><b>Provider:</b> <small>{photo.provider}</small>
                        <br><b>Waktu:</b> {photo.processing_time_ms} ms
                        <br><small style="color:#555;">{photo.caption[:90]}...</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("Belum ada foto dalam koleksi yang sesuai filter. Lakukan analisis citra di Tab 1.")
