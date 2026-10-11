import unittest
import numpy as np
from ocr_engine import LicensePlateOCR


class TestLicensePlateOCR(unittest.TestCase):
    """Pengujian aturan Regex dan pra-pemrosesan citra plat nomor."""

    def setUp(self):
        # Inisialisasi tanpa mengunduh EasyOCR untuk pengujian cepat unit test
        self.ocr = LicensePlateOCR.__new__(LicensePlateOCR)
        self.ocr.reader = None

    def test_clean_standard_plate(self):
        result = self.ocr.clean_plate_text("B 1234 ABC")
        self.assertEqual(result, "B 1234 ABC")

    def test_clean_noisy_and_lowercase_plate(self):
        result = self.ocr.clean_plate_text("  b 9876 xyz \n ")
        self.assertEqual(result, "B 9876 XYZ")

    def test_ambiguous_digits_correction(self):
        # Huruf 'O' pada nomor harus dikoreksi menjadi '0', dan 'I' menjadi '1'
        result = self.ocr.clean_plate_text("B O2I4 ABC")
        self.assertEqual(result, "B 0214 ABC")

    def test_preprocess_plate_shape(self):
        dummy_crop = np.zeros((100, 250, 3), dtype=np.uint8)
        processed = self.ocr.preprocess_plate(dummy_crop)
        self.assertIsNotNone(processed)
        self.assertEqual(len(processed.shape), 2)  # Grayscale / binarized


if __name__ == "__main__":
    unittest.main()
