"""
Module: database.py
Deskripsi: Pengelola database lokal SQLite untuk mencatat log lalu lintas,
           pelanggaran pengendara, serta validasi dokumen berkendara.
           Dilengkapi fitur ekspor ke CSV dan Microsoft Excel melalui Pandas.
"""

import sqlite3
import json
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from config import DB_PATH, EXPORTS_DIR


class DatabaseManager:
    """Manajer SQLite untuk penyimpanan terstruktur dan pelaporan data lalu lintas."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self._write_lock = threading.Lock()
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Membuat koneksi ke database SQLite dengan timeout aman."""
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self):
        """Context manager koneksi: commit jika sukses, rollback jika gagal, dan selalu menutup koneksi."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        """Inisialisasi skema tabel jika belum ada."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

            # Tabel Log Lalu Lintas (Semua kendaraan melintas)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS traffic_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    vehicle_type TEXT NOT NULL,
                    direction TEXT DEFAULT 'MENDEKAT',
                    confidence REAL DEFAULT 0.0
                )
                """
            )

            # Tabel Catatan Pelanggaran
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    track_id INTEGER,
                    vehicle_type TEXT NOT NULL,
                    license_plate TEXT DEFAULT 'TIDAK_TERBACA',
                    violation_type TEXT NOT NULL,
                    helmet_status TEXT DEFAULT 'TIDAK_DIKETAHUI',
                    gemini_safety_validation TEXT,
                    gemini_confidence REAL DEFAULT 0.0,
                    severity TEXT DEFAULT 'MEDIUM',
                    status_tilang TEXT DEFAULT 'PENDING_REVIEW',
                    evidence_rider_path TEXT,
                    evidence_plate_path TEXT
                )
                """
            )

            # Tabel Validasi Dokumen (SIM / STNK)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS document_validations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    document_number TEXT,
                    holder_name TEXT,
                    extracted_plate TEXT,
                    target_detected_plate TEXT,
                    plate_match_status TEXT,
                    expiry_date TEXT,
                    is_expired INTEGER DEFAULT 0,
                    legal_status TEXT,
                    gemini_raw_response TEXT,
                    document_image_path TEXT
                )
                """
            )
            conn.commit()

    def log_traffic_vehicle(
        self,
        track_id: int,
        vehicle_type: str,
        direction: str = "MENDEKAT",
        confidence: float = 0.0,
        timestamp: Optional[str] = None,
    ) -> int:
        """Mencatat kendaraan yang terdeteksi melintas."""
        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._write_lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO traffic_logs (timestamp, track_id, vehicle_type, direction, confidence)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (ts, track_id, vehicle_type, direction, confidence),
                )
                conn.commit()
                return cursor.lastrowid

    def log_violation(
        self,
        vehicle_type: str,
        violation_type: str,
        track_id: Optional[int] = None,
        license_plate: str = "TIDAK_TERBACA",
        helmet_status: str = "TIDAK_PAKAI",
        gemini_safety_validation: Optional[Dict[str, Any]] = None,
        gemini_confidence: float = 0.0,
        severity: str = "MEDIUM",
        status_tilang: str = "TERKONFIRMASI",
        evidence_rider_path: Optional[str] = None,
        evidence_plate_path: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """Menyimpan data insiden pelanggaran lalu lintas."""
        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        validation_json = json.dumps(gemini_safety_validation or {}, ensure_ascii=False)

        with self._write_lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO violations (
                        timestamp, track_id, vehicle_type, license_plate,
                        violation_type, helmet_status, gemini_safety_validation,
                        gemini_confidence, severity, status_tilang,
                        evidence_rider_path, evidence_plate_path
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        track_id,
                        vehicle_type,
                        license_plate,
                        violation_type,
                        helmet_status,
                        validation_json,
                        gemini_confidence,
                        severity,
                        status_tilang,
                        str(evidence_rider_path) if evidence_rider_path else "",
                        str(evidence_plate_path) if evidence_plate_path else "",
                    ),
                )
                conn.commit()
                return cursor.lastrowid

    def log_document_validation(
        self,
        document_type: str,
        document_number: Optional[str],
        holder_name: Optional[str],
        extracted_plate: Optional[str],
        target_detected_plate: Optional[str],
        plate_match_status: str,
        expiry_date: Optional[str],
        is_expired: bool,
        legal_status: str,
        gemini_raw_response: Optional[Dict[str, Any]] = None,
        document_image_path: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """Mencatat hasil verifikasi dokumen berkendara (SIM/STNK)."""
        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        raw_json = json.dumps(gemini_raw_response or {}, ensure_ascii=False)

        with self._write_lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO document_validations (
                        timestamp, document_type, document_number, holder_name,
                        extracted_plate, target_detected_plate, plate_match_status,
                        expiry_date, is_expired, legal_status,
                        gemini_raw_response, document_image_path
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        document_type,
                        document_number or "-",
                        holder_name or "-",
                        extracted_plate or "-",
                        target_detected_plate or "-",
                        plate_match_status,
                        expiry_date or "-",
                        1 if is_expired else 0,
                        legal_status,
                        raw_json,
                        str(document_image_path) if document_image_path else "",
                    ),
                )
                conn.commit()
                return cursor.lastrowid

    def get_violations_dataframe(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Mengambil data pelanggaran sebagai Pandas DataFrame."""
        with self._get_connection() as conn:
            query = "SELECT * FROM violations ORDER BY id DESC"
            if limit:
                query += f" LIMIT {limit}"
            df = pd.read_sql_query(query, conn)
        return df

    def get_traffic_logs_dataframe(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Mengambil data log arus lalu lintas sebagai Pandas DataFrame."""
        with self._get_connection() as conn:
            query = "SELECT * FROM traffic_logs ORDER BY id DESC"
            if limit:
                query += f" LIMIT {limit}"
            df = pd.read_sql_query(query, conn)
        return df

    def get_document_validations_dataframe(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Mengambil data log validasi dokumen sebagai Pandas DataFrame."""
        with self._get_connection() as conn:
            query = "SELECT * FROM document_validations ORDER BY id DESC"
            if limit:
                query += f" LIMIT {limit}"
            df = pd.read_sql_query(query, conn)
        return df

    def export_violations_to_csv(self, output_path: Optional[Path] = None) -> Path:
        """Mengekspor seluruh riwayat pelanggaran ke berkas CSV."""
        df = self.get_violations_dataframe()
        if output_path is None:
            filename = f"laporan_pelanggaran_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            output_path = EXPORTS_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return output_path

    def export_violations_to_excel(self, output_path: Optional[Path] = None) -> Path:
        """Mengekspor seluruh riwayat pelanggaran ke berkas Excel (.xlsx)."""
        df_violations = self.get_violations_dataframe()
        df_traffic = self.get_traffic_logs_dataframe(limit=500)
        df_docs = self.get_document_validations_dataframe()

        if output_path is None:
            filename = f"laporan_lalu_lintas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            output_path = EXPORTS_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_violations.to_excel(writer, sheet_name="Pelanggaran", index=False)
            df_traffic.to_excel(writer, sheet_name="Log Kendaraan", index=False)
            df_docs.to_excel(writer, sheet_name="Validasi Dokumen", index=False)

        return output_path

    def get_system_statistics(self) -> Dict[str, Any]:
        """Menghitung ringkasan statistik pemantauan lalu lintas."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total kendaraan terhitung
            cursor.execute("SELECT COUNT(*) FROM traffic_logs")
            total_vehicles = cursor.fetchone()[0]

            # Total pelanggaran
            cursor.execute("SELECT COUNT(*) FROM violations")
            total_violations = cursor.fetchone()[0]

            # Pelanggaran per jenis
            cursor.execute(
                "SELECT violation_type, COUNT(*) as count FROM violations GROUP BY violation_type"
            )
            violations_by_type = dict(cursor.fetchall())

            # Kendaraan per tipe
            cursor.execute(
                "SELECT vehicle_type, COUNT(*) as count FROM traffic_logs GROUP BY vehicle_type"
            )
            vehicles_by_type = dict(cursor.fetchall())

            # Kepatuhan helm (estimasi motor vs pelanggaran tanpa helm)
            motorcycle_count = vehicles_by_type.get("motorcycle", 0)
            no_helmet_count = violations_by_type.get("TANPA_HELM", 0)
            if motorcycle_count > 0:
                compliance_rate = max(0.0, 100.0 - (no_helmet_count / motorcycle_count * 100.0))
            else:
                compliance_rate = 100.0 if no_helmet_count == 0 else 0.0

            return {
                "total_vehicles": total_vehicles,
                "total_violations": total_violations,
                "violations_by_type": violations_by_type,
                "vehicles_by_type": vehicles_by_type,
                "compliance_rate": round(compliance_rate, 1),
            }


# Singleton instance untuk kemudahan pemakaian lintas modul
db_manager = DatabaseManager()
