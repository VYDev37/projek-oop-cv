import unittest
from domain_models import PlateMatchStatus
from gemini_client import (
    GeminiTrafficClient,
    RiderSafetyValidation,
    DocumentValidation,
)


class TestGeminiClient(unittest.TestCase):
    """Pengujian logika pencocokan plat nomor dan skema Pydantic Gemini AI."""

    def setUp(self):
        self.client = GeminiTrafficClient.__new__(GeminiTrafficClient)
        self.client.client = None

    def test_pydantic_schema_validation(self):
        mock_safety = self.client._mock_rider_safety_analysis()
        self.assertIsInstance(mock_safety, RiderSafetyValidation)
        self.assertTrue(mock_safety.violation_detected)
        self.assertIn("HELM", str(mock_safety.violation_reasons).upper())

        mock_doc = self.client._mock_document_analysis("B 1234 ABC")
        self.assertIsInstance(mock_doc, DocumentValidation)
        self.assertEqual(mock_doc.registered_plate_number, "B 1234 ABC")

    def test_plate_matching_logic(self):
        # 1. Plat sama persis
        res = self.client._check_plate_match("B 1234 ABC", "B 1234 ABC")
        self.assertEqual(res, PlateMatchStatus.MATCH.value)

        # 2. Plat dengan variasi spasi / huruf kecil
        res = self.client._check_plate_match("b 1234 abc", "B1234ABC")
        self.assertEqual(res, PlateMatchStatus.MATCH.value)

        # 3. Plat berbeda (mismatch)
        res = self.client._check_plate_match("B 1234 ABC", "D 9999 XYZ")
        self.assertEqual(res, PlateMatchStatus.MISMATCH.value)

        # 4. Plat tidak terbaca (not checked)
        res = self.client._check_plate_match("B 1234 ABC", "TIDAK_TERBACA")
        self.assertEqual(res, PlateMatchStatus.NOT_CHECKED.value)


if __name__ == "__main__":
    unittest.main()
