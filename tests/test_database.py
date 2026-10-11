import os
import unittest
import tempfile
import threading
from pathlib import Path

from database import DatabaseManager
from domain_models import VehicleType, ViolationType, HelmetStatus, EnforcementStatus


class TestDatabaseManager(unittest.TestCase):
    """Pengujian fungsional dan konkurensi SQLite DatabaseManager."""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_file.close()
        self.db = DatabaseManager(db_path=Path(self.temp_file.name))

    def tearDown(self):
        try:
            if os.path.exists(self.temp_file.name):
                os.remove(self.temp_file.name)
        except Exception:
            pass

    def test_log_traffic_vehicle(self):
        row_id = self.db.log_traffic_vehicle(
            track_id=1,
            vehicle_type=VehicleType.CAR.value,
            direction="MENDEKAT",
            confidence=0.88,
        )
        self.assertGreater(row_id, 0)

        df = self.db.get_traffic_logs_dataframe()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["track_id"], 1)
        self.assertEqual(df.iloc[0]["vehicle_type"], "car")

    def test_log_violation_and_statistics(self):
        # 1. Catat 2 motor melintas
        self.db.log_traffic_vehicle(track_id=10, vehicle_type=VehicleType.MOTORCYCLE.value)
        self.db.log_traffic_vehicle(track_id=11, vehicle_type=VehicleType.MOTORCYCLE.value)

        # 2. Catat 1 pelanggaran tanpa helm
        viol_id = self.db.log_violation(
            vehicle_type=VehicleType.MOTORCYCLE.value,
            violation_type=ViolationType.NO_HELMET.value,
            track_id=10,
            license_plate="B 1234 ABC",
            helmet_status=HelmetStatus.NOT_WORN.value,
            gemini_confidence=0.95,
            severity="HIGH",
            status_tilang=EnforcementStatus.CONFIRMED_TICKET.value,
        )
        self.assertGreater(viol_id, 0)

        # 3. Validasi statistik (1 dari 2 melanggar = kepatuhan 50%)
        stats = self.db.get_system_statistics()
        self.assertEqual(stats["total_vehicles"], 2)
        self.assertEqual(stats["total_violations"], 1)
        self.assertEqual(stats["compliance_rate"], 50.0)

    def test_concurrent_writes(self):
        """Memverifikasi bahwa multiple background threads aman menulis ke SQLite secara bersamaan."""
        num_threads = 10
        errors = []

        def worker(thread_idx):
            try:
                self.db.log_traffic_vehicle(
                    track_id=thread_idx,
                    vehicle_type=VehicleType.CAR.value,
                    confidence=0.9,
                )
                self.db.log_violation(
                    vehicle_type=VehicleType.CAR.value,
                    violation_type=ViolationType.COMPLIANT.value,
                    track_id=thread_idx,
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Terjadi error database saat konkurensi: {errors}")
        df_traffic = self.db.get_traffic_logs_dataframe()
        self.assertEqual(len(df_traffic), num_threads)


if __name__ == "__main__":
    unittest.main()
