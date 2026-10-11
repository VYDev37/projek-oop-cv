import unittest
from domain_models import (
    VehicleType,
    ViolationType,
    HelmetStatus,
    EnforcementStatus,
    PlateMatchStatus,
)


class TestDomainModels(unittest.TestCase):
    """Unit test untuk memvalidasi integritas enumerasi domain."""

    def test_vehicle_types(self):
        self.assertEqual(VehicleType.CAR.value, "car")
        self.assertEqual(VehicleType.MOTORCYCLE.value, "motorcycle")
        self.assertEqual(VehicleType.BUS.value, "bus")
        self.assertEqual(VehicleType.TRUCK.value, "truck")
        self.assertIn("car", [v.value for v in VehicleType])

    def test_violation_types(self):
        self.assertEqual(ViolationType.NO_HELMET.value, "TANPA_HELM")
        self.assertEqual(ViolationType.NON_STANDARD.value, "ATRIBUT_TIDAK_STANDAR")
        self.assertEqual(ViolationType.COMPLIANT.value, "PATUH")

    def test_helmet_statuses(self):
        self.assertEqual(HelmetStatus.WORN.value, "HELM_TERPASANG")
        self.assertEqual(HelmetStatus.NOT_WORN.value, "TIDAK_PAKAI")
        self.assertEqual(HelmetStatus.NON_STANDARD.value, "TIDAK_STANDAR")

    def test_plate_match_statuses(self):
        self.assertEqual(PlateMatchStatus.MATCH.value, "MATCH")
        self.assertEqual(PlateMatchStatus.MISMATCH.value, "MISMATCH")
        self.assertEqual(PlateMatchStatus.NOT_CHECKED.value, "NOT_CHECKED")


if __name__ == "__main__":
    unittest.main()
