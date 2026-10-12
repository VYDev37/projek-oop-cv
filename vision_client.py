"""
Module: vision_client.py
Deskripsi: Modul inti Vision AI yang mengimplementasikan konsep OOP:
           1. Abstract Base Class (ABC) & Inheritance: VisionClient
           2. Subclass Konkrit: GeminiVisionClient, GPT4VisionClient, MockVisionClient
           3. Custom Exception Handling: VisionAPIException
           4. Helper Base64 Encoding: _encode_image()
           5. Structured Data Model: PhotoAnalysisResult (Pydantic BaseModel)
           6. Design Pattern: Strategy Pattern via ImageAnalysisService
"""

import base64
import io
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from pydantic import BaseModel, Field

from config import (
    GEMINI_API_KEY,
    DEFAULT_GEMINI_MODEL,
    OPENAI_API_KEY,
    DEFAULT_GPT4_MODEL,
)

logger = logging.getLogger(__name__)


# 4. CUSTOM EXCEPTION HANDLING

class VisionAPIException(Exception):
    """
    Eksepsi kustom untuk menangani kegagalan pemanggilan Vision API,
    kesalahan komunikasi jaringan, atau kegagalan parsing respons JSON.
    """

    def __init__(self, message: str, provider: str = "Unknown", status_code: Optional[int] = None):
        super().__init__(f"[{provider}] {message}")
        self.message = message
        self.provider = provider
        self.status_code = status_code


# 6. MODEL HASIL ANALISIS CITRA

class PhotoAnalysisResult(BaseModel):
    """
    Model representasi data terstruktur untuk hasil analisis foto.
    Menyimpan caption, tags, objects, kategori, is_safe, colors, dan metadata pendukung.
    """
    caption: str = Field(
        ...,
        description="Deskripsi naratif singkat dan informatif tentang isi foto.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Daftar label/kata kunci yang relevan dengan citra (misal: ['sunset', 'pantai', 'senja']).",
    )
    objects: List[str] = Field(
        default_factory=list,
        description="Objek-objek fisik yang terdeteksi di dalam foto (misal: ['mobil', 'pohon', 'orang']).",
    )
    kategori: str = Field(
        ...,
        description="Kategori utama citra (misal: Alam, Makanan, Perkotaan, Dokumen, Kendaraan, Potret).",
    )
    is_safe: bool = Field(
        ...,
        description="Status moderasi konten: True jika aman untuk publik, False jika mengandung konten tidak pantas.",
    )
    colors: List[str] = Field(
        default_factory=list,
        description="Palet warna dominan pada foto (misal: ['Biru Laut', 'Oranye', 'Emas']).",
    )
    ocr_text: Optional[str] = Field(
        default="",
        description="Teks yang berhasil diekstraksi dari citra jika ada.",
    )
    provider: str = Field(
        default="Unknown",
        description="Nama provider AI yang melakukan analisis (Gemini, GPT-4o, atau Mock).",
    )
    image_path: Optional[str] = Field(
        default="",
        description="Path file gambar asli jika tersedia.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        description="Waktu analisis dilakukan.",
    )
    processing_time_ms: Optional[float] = Field(
        default=0.0,
        description="Waktu eksekusi analisis dalam milidetik.",
    )


# 1. DEFINISI VISION CLIENT (ABSTRACT BASE CLASS)

class VisionClient(ABC):
    """
    Abstract Base Class (ABC) yang mendefinisikan antarmuka kontrak standar
    bagi seluruh provider Vision AI. Mendukung Polymorphism dan OCP (SOLID).
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    # 5. Helper Encoding Citra ke Base64
    def _encode_image(self, image_input: Union[str, Path, bytes, Image.Image]) -> str:
        """
        Mengonversi input foto (path berkas, bytes mentah, atau PIL.Image)
        menjadi representasi string base64 UTF-8.
        """
        try:
            if isinstance(image_input, (str, Path)):
                file_path = Path(image_input)
                if not file_path.exists():
                    raise FileNotFoundError(f"Berkas gambar tidak ditemukan: {file_path}")
                with open(file_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")

            elif isinstance(image_input, bytes):
                return base64.b64encode(image_input).decode("utf-8")

            elif isinstance(image_input, Image.Image):
                buffer = io.BytesIO()
                # Simpan dalam format RGB JPEG
                rgb_img = image_input.convert("RGB")
                rgb_img.save(buffer, format="JPEG", quality=90)
                return base64.b64encode(buffer.getvalue()).decode("utf-8")

            else:
                raise ValueError(f"Tipe input gambar tidak didukung: {type(image_input)}")

        except Exception as e:
            raise VisionAPIException(
                f"Gagal melakukan encoding gambar ke base64: {e}",
                provider=self.__class__.__name__,
            )

    def _prepare_pil_image(self, image_input: Union[str, Path, bytes, Image.Image]) -> Image.Image:
        """Helper untuk mendapatkan objek PIL.Image dari berbagai format input."""
        if isinstance(image_input, Image.Image):
            return image_input
        elif isinstance(image_input, (str, Path)):
            return Image.open(image_input)
        elif isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input))
        else:
            raise ValueError(f"Tipe input gambar tidak didukung: {type(image_input)}")

    # 7. Method Kontrak Polimorfik
    @abstractmethod
    def analyze(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        image_name: Optional[str] = None,
    ) -> PhotoAnalysisResult:
        """
        Menganalisis citra dan mengembalikan objek terstruktur PhotoAnalysisResult.
        Harus diimplementasikan oleh setiap subclass konkrit.
        """
        pass


# 2. GEMINI VISION CLIENT

class GeminiVisionClient(VisionClient):
    """
    Subclass konkrit VisionClient yang memanfaatkan Google GenAI SDK (google-genai).
    Mengirimkan gambar multimodal dan mem-parse Structured JSON respons ke PhotoAnalysisResult.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
    ):
        super().__init__(model_name=model_name, api_key=api_key or GEMINI_API_KEY)
        self.client = None
        self._init_client()

    def _init_client(self) -> None:
        """Inisialisasi klien SDK google-genai."""
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info("GeminiVisionClient berhasil tersambung dengan model: %s", self.model_name)
            except Exception as e:
                logger.warning("Gagal inisialisasi google-genai Client: %s", e)
                self.client = None
        else:
            logger.info("Gemini API Key tidak disetel. GeminiVisionClient tidak aktif.")

    def analyze(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        image_name: Optional[str] = None,
    ) -> PhotoAnalysisResult:
        """Mengirimkan foto ke Google Gemini dan mem-parse respons terstruktur."""
        start_time = time.time()
        image_path_str = str(image_input) if isinstance(image_input, (str, Path)) else (image_name or "in_memory_image")

        if not self.client:
            raise VisionAPIException(
                "Gemini API Client tidak tersedia. Pastikan GEMINI_API_KEY telah diatur di .env.",
                provider="GeminiVisionClient",
                status_code=401,
            )

        try:
            pil_image = self._prepare_pil_image(image_input)
            from google.genai import types

            prompt = (
                "Lakukan analisis mendalam terhadap foto ini untuk platform kurasi gambar cerdas.\n"
                "1. Berikan caption naratif deskriptif dalam Bahasa Indonesia.\n"
                "2. Ekstraksi 4-8 tags kata kunci paling relevan.\n"
                "3. Sebutkan objek fisik utama yang terlihat.\n"
                "4. Tentukan satu kategori utama (Alam, Makanan, Perkotaan, Dokumen, Kendaraan, Seni, Potret, dll).\n"
                "5. Moderasi konten: tentukan apakah foto ini aman (is_safe=True) atau berbahaya/tidak senonoh/kekerasan (is_safe=False).\n"
                "6. Sebutkan 2-4 warna dominan.\n"
                "7. Jika terdapat teks pada gambar, baca teks tersebut ke ocr_text (kosongkan jika tidak ada teks)."
            )

            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PhotoAnalysisResult,
                system_instruction=(
                    "Anda adalah AI Ahli Kurasi & Analisis Citra Digital profesional. "
                    "Hasil analisis harus selalu akurat, objektif, dan mengikuti skema JSON yang ditentukan."
                ),
                temperature=0.2,
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[pil_image, prompt],
                config=config,
            )

            if not response.text:
                raise VisionAPIException(
                    "Respons kosong dari Gemini API.",
                    provider="GeminiVisionClient",
                    status_code=500,
                )

            # Parse respons JSON ke PhotoAnalysisResult
            raw_data = json.loads(response.text)
            elapsed_ms = (time.time() - start_time) * 1000

            result = PhotoAnalysisResult(
                caption=raw_data.get("caption", "Foto tanpa caption"),
                tags=[str(t).lower() for t in raw_data.get("tags", [])],
                objects=raw_data.get("objects", []),
                kategori=raw_data.get("kategori", "Umum"),
                is_safe=bool(raw_data.get("is_safe", True)),
                colors=raw_data.get("colors", []),
                ocr_text=raw_data.get("ocr_text", ""),
                provider=f"Google Gemini ({self.model_name})",
                image_path=image_path_str,
                processing_time_ms=round(elapsed_ms, 2),
            )
            return result

        except json.JSONDecodeError as jde:
            raise VisionAPIException(
                f"Gagal mem-parse JSON hasil Gemini: {jde}",
                provider="GeminiVisionClient",
                status_code=422,
            )
        except VisionAPIException:
            raise
        except Exception as e:
            raise VisionAPIException(
                f"Terjadi kesalahan saat memproses gambar dengan Gemini: {e}",
                provider="GeminiVisionClient",
                status_code=500,
            )


# 3. MOCK VISION CLIENT

class MockVisionClient(VisionClient):
    """
    Subclass konkrit VisionClient untuk mode demo offline dan unit testing.
    Menghasilkan hasil analisis deterministik tanpa memerlukan koneksi internet atau API key.
    """

    def __init__(self, model_name: str = "mock-vision-v1", api_key: Optional[str] = "mock_key"):
        super().__init__(model_name=model_name, api_key=api_key)

    def analyze(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        image_name: Optional[str] = None,
    ) -> PhotoAnalysisResult:
        """Menghasilkan analisis simulasi cerdas berdasarkan nama atau metadata berkas."""
        start_time = time.time()

        # Validasi bahwa input gambar valid (memastikan _encode_image bekerja)
        _ = self._encode_image(image_input)

        name_lower = ""
        if isinstance(image_input, (str, Path)):
            name_lower = Path(image_input).name.lower()
        elif image_name:
            name_lower = image_name.lower()

        # Simulasi deteksi konten unsafe jika nama mengandung kata sensitif
        is_safe = True
        if any(w in name_lower for w in ["unsafe", "hazard", "bahaya", "nsfw", "violence"]):
            is_safe = False

        # Simulasi kategori & deskripsi berbasis konten nama
        if any(w in name_lower for w in ["pantai", "beach", "sunset", "alam", "nature", "gunung"]):
            caption = "Pemandangan lanskap alam yang indah dan menenangkan saat matahari terbenam."
            tags = ["alam", "pemandangan", "sunset", "outdoor", "langit"]
            objects = ["matahari", "awan", "pepohonan", "horizon"]
            kategori = "Alam"
            colors = ["Oranye Keemasan", "Biru Langit", "Kuning"]
            ocr_text = ""
        elif any(w in name_lower for w in ["makanan", "food", "kopi", "cafe", "restoran"]):
            caption = "Sajian kuliner lezat yang ditata menarik di atas meja kafe."
            tags = ["makanan", "kuliner", "kafe", "hidangan", "minuman"]
            objects = ["piring", "cangkir", "meja", "sendok"]
            kategori = "Makanan"
            colors = ["Cokelat Hangat", "Putih Krem", "Merah"]
            ocr_text = "SPECIAL MENU TODAY"
        elif any(w in name_lower for w in ["mobil", "motor", "traffic", "kendaraan", "vehicle"]):
            caption = "Kendaraan bermotor yang sedang melintas di area jalan raya."
            tags = ["kendaraan", "transportasi", "jalanan", "otomotif"]
            objects = ["mobil", "motor", "aspal", "lampu lalu lintas"]
            kategori = "Kendaraan"
            colors = ["Hitam Metalik", "Abu-abu", "Merah"]
            ocr_text = "B 1234 XYZ"
        elif any(w in name_lower for w in ["dokumen", "surat", "stnk", "sim", "teks"]):
            caption = "Dokumen tertulis resmi dengan informasi identitas dan nomor registrasi."
            tags = ["dokumen", "berkas", "teks", "resmi", "arsip"]
            objects = ["kertas", "kartu identitas", "tanda tangan"]
            kategori = "Dokumen"
            colors = ["Putih", "Biru Tua", "Hitam"]
            ocr_text = "SURAT KETERANGAN RESMI No. 123/2026"
        elif not is_safe:
            caption = "Gambar terdeteksi memuat konten tidak aman yang melanggar panduan moderasi."
            tags = ["unsafe", "peringatan", "moderasi"]
            objects = ["objek berbahaya"]
            kategori = "Sensitif"
            colors = ["Merah Pekat", "Hitam"]
            ocr_text = "WARNING: UNSAFE CONTENT"
        else:
            caption = "Citra visual digital dengan komposisi objek yang jelas dan seimbang."
            tags = ["foto", "visual", "koleksi", "kreatif"]
            objects = ["objek utama", "latar belakang"]
            kategori = "Umum"
            colors = ["Biru", "Putih", "Abu-abu"]
            ocr_text = ""

        image_path_str = str(image_input) if isinstance(image_input, (str, Path)) else (image_name or "mock_image.jpg")
        elapsed_ms = (time.time() - start_time) * 1000

        return PhotoAnalysisResult(
            caption=caption,
            tags=tags,
            objects=objects,
            kategori=kategori,
            is_safe=is_safe,
            colors=colors,
            ocr_text=ocr_text,
            provider="Mock Vision Engine (Offline)",
            image_path=image_path_str,
            processing_time_ms=round(elapsed_ms, 2),
        )


# 16. GPT-4 VISION CLIENT

class GPT4VisionClient(VisionClient):
    """
    Subclass konkrit VisionClient yang memanfaatkan model OpenAI GPT-4o Vision API.
    Mendukung pengiriman base64 image payload dan ekstraksi JSON terstruktur.
    Jika OPENAI_API_KEY tidak tersedia, menyediakan simulasi inferensi GPT-4o yang cerdas.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_GPT4_MODEL,
    ):
        super().__init__(model_name=model_name, api_key=api_key or OPENAI_API_KEY)

    def analyze(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        image_name: Optional[str] = None,
    ) -> PhotoAnalysisResult:
        """Mengirimkan foto ke OpenAI GPT-4o Vision API atau menjalankan inferensi terpadu."""
        start_time = time.time()
        image_path_str = str(image_input) if isinstance(image_input, (str, Path)) else (image_name or "in_memory_image")

        b64_image = self._encode_image(image_input)

        if self.api_key:
            try:
                import requests

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }

                system_prompt = (
                    "Anda adalah AI vision analitik tingkat tinggi. Analisis citra dan balas HANYA dalam JSON murni "
                    "dengan format berikut tanpa markdown backticks:\n"
                    "{\n"
                    '  "caption": "deskripsi naratif bahasa indonesia",\n'
                    '  "tags": ["tag1", "tag2", "tag3"],\n'
                    '  "objects": ["objek1", "objek2"],\n'
                    '  "kategori": "kategori utama",\n'
                    '  "is_safe": true,\n'
                    '  "colors": ["warna1", "warna2"],\n'
                    '  "ocr_text": "teks terbaca atau kosong"\n'
                    "}"
                )

                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Analisis gambar ini secara detail sesuai skema JSON."},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                                },
                            ],
                        },
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 800,
                }

                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )

                if response.status_code != 200:
                    raise VisionAPIException(
                        f"OpenAI API Error: {response.text}",
                        provider="GPT4VisionClient",
                        status_code=response.status_code,
                    )

                resp_json = response.json()
                content_text = resp_json["choices"][0]["message"]["content"]
                parsed = json.loads(content_text)
                elapsed_ms = (time.time() - start_time) * 1000

                return PhotoAnalysisResult(
                    caption=parsed.get("caption", "Analisis GPT-4o"),
                    tags=[str(t).lower() for t in parsed.get("tags", [])],
                    objects=parsed.get("objects", []),
                    kategori=parsed.get("kategori", "Umum"),
                    is_safe=bool(parsed.get("is_safe", True)),
                    colors=parsed.get("colors", []),
                    ocr_text=parsed.get("ocr_text", ""),
                    provider=f"OpenAI ({self.model_name})",
                    image_path=image_path_str,
                    processing_time_ms=round(elapsed_ms, 2),
                )

            except VisionAPIException:
                raise
            except Exception as e:
                raise VisionAPIException(
                    f"Gagal memproses gambar via OpenAI: {e}",
                    provider="GPT4VisionClient",
                    status_code=500,
                )

        else:
            # Fallback simulasi GPT-4o yang cerdas jika OpenAI API Key belum disetel
            time.sleep(0.1)  # Simulasi latensi jaringan
            elapsed_ms = (time.time() - start_time) * 1000
            pil_img = self._prepare_pil_image(image_input)
            w, h = pil_img.size

            return PhotoAnalysisResult(
                caption=f"Hasil penalaran visual mendalam GPT-4o Vision pada citra beresolusi {w}x{h} piksel.",
                tags=["gpt4o-vision", "high-resolution", "ai-analysis", "deep-reasoning"],
                objects=["elemen visual utama", "tekstur permukaan", "latar belakang"],
                kategori="Analisis Lanjut",
                is_safe=True,
                colors=["Abu-abu Netral", "Kontras Tinggi"],
                ocr_text="GPT-4o Vision OCR Mock Text",
                provider=f"OpenAI ({self.model_name} Simulated)",
                image_path=image_path_str,
                processing_time_ms=round(elapsed_ms, 2),
            )


# 20. DESIGN PATTERN: STRATEGY PATTERN

class ImageAnalysisService:
    """
    Context class dalam Design Pattern Strategy.
    Menerima strategi VisionClient secara dinamis, memungkinkan aplikasi untuk
    menukar provider (Gemini / GPT-4o / Mock) saat runtime tanpa mengubah kode pemanggil (OCP & DIP).
    """

    def __init__(self, strategy: Optional[VisionClient] = None):
        self._strategy: VisionClient = strategy or MockVisionClient()

    @property
    def strategy(self) -> VisionClient:
        """Mengambil strategi VisionClient saat ini."""
        return self._strategy

    def set_strategy(self, strategy: VisionClient) -> None:
        """
        Menukar provider strategi secara dinamis saat runtime.
        (Memenuhi konsep Polymorphism & Strategy Pattern).
        """
        is_valid_strategy = isinstance(strategy, VisionClient) or any(
            getattr(base, "__name__", "") == "VisionClient"
            for base in getattr(strategy.__class__, "__mro__", [])
        )
        if not is_valid_strategy:
            raise TypeError("Strategi harus merupakan turunan dari kelas abstrak VisionClient.")
        self._strategy = strategy
        logger.info("Strategi provider Vision AI ditukar menjadi: %s", strategy.__class__.__name__)

    def analyze_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        image_name: Optional[str] = None,
    ) -> PhotoAnalysisResult:
        """Mendelegasikan proses analisis ke objek strategi aktif."""
        return self._strategy.analyze(image_input, image_name=image_name)

    # 17. Komparasi Hasil 2 Provider Berbeda
    def compare_providers(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        client_a: VisionClient,
        client_b: VisionClient,
        image_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Menganalisis foto yang sama menggunakan dua provider berbeda dan membandingkan
        caption, tag, kecepatan pemrosesan, serta kesamaan tag.
        """
        res_a = client_a.analyze(image_input, image_name=image_name)
        res_b = client_b.analyze(image_input, image_name=image_name)

        set_tags_a = set(res_a.tags)
        set_tags_b = set(res_b.tags)
        shared_tags = list(set_tags_a.intersection(set_tags_b))

        return {
            "provider_a": {
                "name": res_a.provider,
                "caption": res_a.caption,
                "tags": res_a.tags,
                "kategori": res_a.kategori,
                "is_safe": res_a.is_safe,
                "processing_time_ms": res_a.processing_time_ms,
            },
            "provider_b": {
                "name": res_b.provider,
                "caption": res_b.caption,
                "tags": res_b.tags,
                "kategori": res_b.kategori,
                "is_safe": res_b.is_safe,
                "processing_time_ms": res_b.processing_time_ms,
            },
            "shared_tags": shared_tags,
            "tag_agreement_rate": (
                round(len(shared_tags) / max(1, len(set_tags_a.union(set_tags_b))) * 100, 1)
            ),
        }
