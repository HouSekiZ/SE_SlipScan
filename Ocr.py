"""
SlipScan OCR Service
====================
Dependencies:
    pip install typhoon-ocr pillow opencv-python-headless

Setup:
    export TYPHOON_OCR_API_KEY=your_api_key_here
    # รับ API key ได้ที่ https://opentyphoon.ai

Usage:
    ocr = SlipOCR()
    raw_text = ocr.read("slip.jpg")
"""

import logging
from pathlib import Path

logger = logging.getLogger("slipscan.ocr")

try:
    from PIL import Image, ImageFilter, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from typhoon_ocr import ocr_document
    TYPHOON_AVAILABLE = True
except ImportError:
    TYPHOON_AVAILABLE = False


# ─────────────────────────────────────────────
# IMAGE PREPROCESSOR
# ─────────────────────────────────────────────

class ImagePreprocessor:

    @staticmethod
    def preprocess(image_path: str) -> str:
        """ปรับปรุงคุณภาพภาพ → คืน path ไฟล์ที่ปรับแล้ว"""
        if CV2_AVAILABLE:
            return ImagePreprocessor._cv2_preprocess(image_path)
        elif PIL_AVAILABLE:
            return ImagePreprocessor._pil_preprocess(image_path)
        else:
            logger.warning("ไม่มี opencv/PIL — ใช้ภาพต้นฉบับ")
            return image_path

    @staticmethod
    def _cv2_preprocess(image_path: str) -> str:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"ไม่สามารถอ่านไฟล์ภาพ: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize ถ้าเล็กกว่า 1200px
        h, w = gray.shape
        if max(h, w) < 1200:
            scale = 1200 / max(h, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Adaptive Threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )

        # Denoise + Sharpen
        denoised = cv2.medianBlur(thresh, 3)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp = cv2.filter2D(denoised, -1, kernel)

        out_path = image_path.replace(".", "_processed.")
        cv2.imwrite(out_path, sharp)
        return out_path

    @staticmethod
    def _pil_preprocess(image_path: str) -> str:
        img = Image.open(image_path).convert("L")
        w, h = img.size
        if max(w, h) < 1200:
            scale = 1200 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img = img.filter(ImageFilter.SHARPEN)
        img = ImageEnhance.Contrast(img).enhance(1.5)
        out_path = image_path.replace(".", "_processed.")
        img.save(out_path)
        return out_path


# ─────────────────────────────────────────────
# OCR ENGINE
# ─────────────────────────────────────────────

class TyphoonOCREngine:
    """
    Typhoon OCR API (opentyphoon.ai)
    ต้องตั้ง env: TYPHOON_OCR_API_KEY

    Args:
        base_url: None = ใช้ opentyphoon.ai cloud
                  "http://localhost:8000/v1" = self-hosted via vllm
        api_key:  None = อ่านจาก TYPHOON_OCR_API_KEY env
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        if not TYPHOON_AVAILABLE:
            raise ImportError("pip install typhoon-ocr")
        self.base_url = base_url
        self.api_key  = api_key

    def read(self, image_path: str) -> str:
        """คืน raw_text (markdown format)"""
        kwargs = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.api_key:
            kwargs["api_key"] = self.api_key

        markdown = ocr_document(pdf_or_image_path=image_path, **kwargs)
        return markdown


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

class SlipOCR:
    """
    Example:
        # Cloud API
        ocr = SlipOCR()
        text = ocr.read("slip.jpg")

        # Self-hosted (vllm)
        ocr = SlipOCR(base_url="http://localhost:8000/v1", api_key="no-key")
        text = ocr.read("slip.jpg")
    """

    ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".pdf"}

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        preprocess: bool = True,
    ):
        self.preprocess = preprocess
        self._engine = TyphoonOCREngine(base_url=base_url, api_key=api_key)

    def read(self, image_path: str) -> str:
        """
        อ่านสลิปจากไฟล์ภาพ

        Returns:
            raw_text (markdown)
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์: {image_path}")
        if path.suffix.lower() not in self.ALLOWED_EXT:
            raise ValueError(f"ไม่รองรับไฟล์ประเภท: {path.suffix}")

        processed = str(path)
        if self.preprocess and path.suffix.lower() != ".pdf":
            try:
                processed = ImagePreprocessor.preprocess(str(path))
            except Exception as e:
                logger.warning(f"Preprocess ล้มเหลว: {e} — ใช้ภาพต้นฉบับ")

        try:
            return self._engine.read(processed)
        except Exception as e:
            raise RuntimeError(f"OCR ล้มเหลว: {e}") from e


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_service.py <image_path> [--local]")
        print("")
        print("  --local   ใช้ self-hosted vllm ที่ localhost:8000")
        print("")
        print("Environment:")
        print("  TYPHOON_OCR_API_KEY=your_key   (สำหรับ cloud)")
        sys.exit(1)

    use_local = "--local" in sys.argv
    base_url  = "http://localhost:8000/v1" if use_local else None
    api_key   = "no-key" if use_local else None

    ocr  = SlipOCR(base_url=base_url, api_key=api_key)
    text = ocr.read(sys.argv[1])
    print(text)