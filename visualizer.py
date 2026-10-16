"""
Module: visualizer.py
Deskripsi: Komponen visualisasi terdedikasi untuk Sistem Pemantauan Lalu Lintas.
           Menerapkan prinsip Single Responsibility Principle (SRP) untuk memisahkan
           tanggung jawab rendering grafis dari logika inferensi deteksi objek.
"""

from typing import Dict, List, Tuple, Any
import cv2
import numpy as np

from config import COLOR_PALETTE


class TrafficVisualizer:
    """Renderer khusus untuk menggambar anotasi Computer Vision, jejak lintasan, dan HUD."""

    def __init__(self, palette: Dict[str, Tuple[int, int, int]] = COLOR_PALETTE):
        self.palette = palette

    def draw_counting_line(self, frame: np.ndarray, line_y: int) -> None:
        """Menggambar garis hitung virtual melintang."""
        w = frame.shape[1]
        line_color = self.palette.get("counting_line", (255, 0, 0))
        cv2.line(frame, (0, line_y), (w, line_y), line_color, 2)
        cv2.putText(
            frame,
            "GARIS PEMANTAUAN LALU LINTAS",
            (20, line_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            line_color,
            2,
        )

    def draw_track_trails(
        self, frame: np.ndarray, track_history: Dict[int, List[Tuple[int, int]]]
    ) -> None:
        """Menggambar jejak pergerakan (trail line) kendaraan."""
        for tid, history in track_history.items():
            for i in range(1, len(history)):
                cv2.line(frame, history[i - 1], history[i], (0, 255, 255), 2)

    def draw_tracked_objects(
        self,
        frame: np.ndarray,
        tracked_vehicles: List[Any],
        detected_persons: List[Any],
    ) -> None:
        """Menggambar bounding box dan label kendaraan serta pengendara/pejalan kaki."""
        # 1. Anotasi Kendaraan
        for vehicle in tracked_vehicles:
            vx1, vy1, vx2, vy2 = vehicle.bbox
            v_type = vehicle.class_name
            tid = vehicle.track_id
            box_color = self.palette.get(v_type, (255, 255, 255))

            label = f"ID:{tid} {v_type} {vehicle.confidence:.2f}"
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), box_color, 2)
            cv2.rectangle(
                frame,
                (vx1, vy1 - 25),
                (vx1 + len(label) * 11, vy1),
                box_color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (vx1 + 5, vy1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

        # 2. Anotasi Manusia (Person)
        person_color = self.palette.get("person", (255, 255, 0))
        for person in detected_persons:
            px1, py1, px2, py2 = person.bbox
            cv2.rectangle(frame, (px1, py1), (px2, py2), person_color, 1)

    def draw_inspected_static_objects(
        self,
        frame: np.ndarray,
        inspections: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Menggambar hasil inspeksi ETLE statis dengan indikator kepatuhan (Hijau) atau pelanggaran (Merah)."""
        annotated = frame.copy()
        for item in inspections:
            vx1, vy1, vx2, vy2 = item["bbox"]
            is_violation = item.get("violation_detected", False)
            box_color = (
                self.palette.get("violation", (0, 0, 255))
                if is_violation
                else self.palette.get("compliant", (0, 255, 0))
            )

            plate = item.get("license_plate", "TIDAK_TERBACA")
            v_type = item.get("vehicle_type", "kendaraan")
            status_text = "TILANG" if is_violation else "PATUH"
            label = f"{v_type.upper()} | {plate} | {status_text}"

            cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), box_color, 3)
            # Label background box
            cv2.rectangle(
                annotated,
                (vx1, max(0, vy1 - 25)),
                (vx1 + len(label) * 10, max(25, vy1)),
                box_color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (vx1 + 5, max(18, vy1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255) if is_violation else (0, 0, 0),
                2,
            )
        return annotated

    def draw_dashboard_overlay(
        self,
        frame: np.ndarray,
        total_counted: int,
        vehicle_counts: Dict[str, int],
    ) -> None:
        """Menggambar HUD transparan di pojok kiri atas frame."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 150), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(
            frame,
            "TRAFFIC MONITORING SYSTEM",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Total Kendaraan: {total_counted}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Mobil: {vehicle_counts.get('car', 0)} | Motor: {vehicle_counts.get('motorcycle', 0)}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            frame,
            f"Bus: {vehicle_counts.get('bus', 0)} | Truk: {vehicle_counts.get('truck', 0)}",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

    def annotate_frame(
        self,
        frame: np.ndarray,
        line_y: int,
        tracked_vehicles: List[Any],
        detected_persons: List[Any],
        track_history: Dict[int, List[Tuple[int, int]]],
        total_counted: int,
        vehicle_counts: Dict[str, int],
    ) -> np.ndarray:
        """Pipeline anotasi terpadu pada salinan frame."""
        annotated = frame.copy()
        self.draw_counting_line(annotated, line_y)
        self.draw_track_trails(annotated, track_history)
        self.draw_tracked_objects(annotated, tracked_vehicles, detected_persons)
        self.draw_dashboard_overlay(annotated, total_counted, vehicle_counts)
        return annotated
