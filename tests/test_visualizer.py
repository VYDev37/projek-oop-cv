import unittest
import numpy as np
from visualizer import TrafficVisualizer
from detector import TrackedObject


class TestTrafficVisualizer(unittest.TestCase):
    """Pengujian komponen rendering grafis TrafficVisualizer."""

    def setUp(self):
        self.visualizer = TrafficVisualizer()

    def test_annotate_frame_output_integrity(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        line_y = 300

        dummy_vehicle = TrackedObject(
            track_id=1,
            class_name="car",
            bbox=(50, 100, 150, 200),
            confidence=0.92,
            center=(100, 150),
        )

        dummy_person = TrackedObject(
            track_id=2,
            class_name="person",
            bbox=(200, 120, 250, 220),
            confidence=0.85,
            center=(225, 170),
        )

        track_history = {1: [(100, 120), (100, 150)]}
        vehicle_counts = {"car": 1, "motorcycle": 0}

        annotated = self.visualizer.annotate_frame(
            frame=frame,
            line_y=line_y,
            tracked_vehicles=[dummy_vehicle],
            detected_persons=[dummy_person],
            track_history=track_history,
            total_counted=1,
            vehicle_counts=vehicle_counts,
        )

        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, (480, 640, 3))
        self.assertEqual(annotated.dtype, np.uint8)

    def test_draw_inspected_static_objects(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        inspections = [
            {
                "bbox": (50, 50, 200, 200),
                "violation_detected": True,
                "license_plate": "B 1234 CD",
                "vehicle_type": "motorcycle",
            },
            {
                "bbox": (250, 100, 450, 300),
                "violation_detected": False,
                "license_plate": "D 5678 EF",
                "vehicle_type": "car",
            },
        ]

        annotated = self.visualizer.draw_inspected_static_objects(frame, inspections)
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, (480, 640, 3))
        self.assertFalse(np.array_equal(frame, annotated))


if __name__ == "__main__":
    unittest.main()
