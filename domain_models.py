"""
Module: domain_models.py
Deskripsi: Definisi enumerasi (Enums) dan domain models untuk type safety
           dan standarisasi data di seluruh modul sistem lalu lintas.
"""

from enum import Enum


class VehicleType(str, Enum):
    """Kategori kendaraan yang dikenali sistem."""
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"
    TRUCK = "truck"
    BICYCLE = "bicycle"
    PERSON = "person"


class ViolationType(str, Enum):
    """Kategori pelanggaran aturan lalu lintas."""
    NO_HELMET = "TANPA_HELM"
    NON_STANDARD = "ATRIBUT_TIDAK_STANDAR"
    COMPLIANT = "PATUH"


class HelmetStatus(str, Enum):
    """Status kepatuhan penggunaan helm pengendara."""
    WORN = "HELM_TERPASANG"
    NOT_WORN = "TIDAK_PAKAI"
    NON_STANDARD = "TIDAK_STANDAR"
    UNKNOWN = "TIDAK_DIKETAHUI"


class EnforcementStatus(str, Enum):
    """Status proses penindakan hukum/tilang."""
    RELEASED = "DILEPASKAN"
    CONFIRMED_TICKET = "TERKONFIRMASI_TILANG"
    PENDING_REVIEW = "PENDING_REVIEW"


class PlateMatchStatus(str, Enum):
    """Status kecocokan nomor plat dokumen dengan kamera."""
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    NOT_CHECKED = "NOT_CHECKED"
