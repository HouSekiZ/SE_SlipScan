import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

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
# SLIP PARSER
# ─────────────────────────────────────────────

class SlipParser:

    # ธนาคารไทย
    BANK_PATTERNS = {
        'กสิกรไทย':     r'(?:kbank|กสิกร|kasikorn)',
        'ไทยพาณิชย์':   r'(?:scb|ไทยพาณิชย์|siam\s*commercial)',
        'กรุงไทย':      r'(?:ktb|กรุงไทย|krungthai)',
        'กรุงเทพ':      r'(?:bbl|กรุงเทพ|bangkok\s*bank)',
        'ทหารไทยธนชาต': r'(?:ttb|tmb|ทหารไทย|ธนชาต)',
        'ออมสิน':       r'(?:gsb|ออมสิน|government\s*savings)',
        'กรุงศรี':      r'(?:bay|กรุงศรี|krungsri)',
        'ธนชาต':        r'(?:tbank|ธนชาต|thanachart)',
        'ซีไอเอ็มบี':   r'(?:cimb)',
        'ยูโอบี':       r'(?:uob)',
    }

    # Regex patterns
    AMOUNT_REGEX = re.compile(
        r'(?:จำนวน|amount|total|ยอดโอน|฿|thb)?\s*([\d,]+\.?\d{0,2})',
        re.IGNORECASE
    )
    
    DATE_REGEX = re.compile(
        r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})'
    )
    
    TIME_REGEX = re.compile(
        r'(\d{1,2}):(\d{2})(?::(\d{2}))?'
    )
    
    REF_REGEX = re.compile(
        r'(?:ref|อ้างอิง|หมายเลข|reference)[.\s:]*([A-Z0-9]{6,20})',
        re.IGNORECASE
    )
    
    ACCOUNT_REGEX = re.compile(
        r'(\d{3}[\-]?\d{1}[\-]?\d{4,5}[\-]?\d{1})'
    )

    def parse(self, raw_text: str) -> dict[str, Any]:
        """
        แยกข้อมูลจาก raw OCR text

        Returns:
            {
                "sender_name": str,
                "bank_name": str,
                "amount": float,
                "slip_date": str,  # YYYY-MM-DD
                "slip_time": str,  # HH:MM:SS
                "ref_no": str,
                "receiver_name": str,
                "receiver_account": str,
                "raw_ocr": str
            }
        """
        text = raw_text.lower()

        return {
            "sender_name": self._extract_sender_name(raw_text),
            "bank_name": self._extract_bank_name(text),
            "amount": self._extract_amount(text),
            "slip_date": self._extract_date(text),
            "slip_time": self._extract_time(text),
            "ref_no": self._extract_ref_no(raw_text),
            "receiver_name": self._extract_receiver_name(raw_text),
            "receiver_account": self._extract_account(raw_text),
            "raw_ocr": raw_text,
        }

    def _extract_amount(self, text: str) -> float | None:
        """ดึงจำนวนเงิน"""
        matches = self.AMOUNT_REGEX.findall(text)
        if matches:
            # เลือกจำนวนที่สูงสุด (มักเป็นยอดโอนจริง)
            amounts = [float(m.replace(',', '')) for m in matches if m]
            return max(amounts) if amounts else None
        return None

    def _extract_bank_name(self, text: str) -> str | None:
        """ดึงชื่อธนาคาร"""
        for bank, pattern in self.BANK_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return bank
        return None

    def _extract_date(self, text: str) -> str | None:
        """ดึงวันที่ในรูปแบบ YYYY-MM-DD"""
        match = self.DATE_REGEX.search(text)
        if match:
            day, month, year = match.groups()
            # แปลง พ.ศ. เป็น ค.ศ.
            year = int(year)
            if year > 2500:
                year -= 543
            elif year < 100:
                year += 2000
            try:
                return f"{year:04d}-{int(month):02d}-{int(day):02d}"
            except ValueError:
                return None
        return None

    def _extract_time(self, text: str) -> str | None:
        """ดึงเวลาในรูปแบบ HH:MM:SS"""
        match = self.TIME_REGEX.search(text)
        if match:
            hour, minute, second = match.groups()
            second = second or "00"
            return f"{int(hour):02d}:{int(minute):02d}:{int(second):02d}"
        return None

    def _extract_ref_no(self, text: str) -> str | None:
        """ดึงหมายเลขอ้างอิง"""
        match = self.REF_REGEX.search(text)
        return match.group(1) if match else None

    def _extract_sender_name(self, text: str) -> str | None:
        """ดึงชื่อผู้โอน (ต้องปรับตาม format ของแต่ละธนาคาร)"""
        # ตัวอย่างเบื้องต้น: หาชื่อที่อยู่หลังคำว่า "จาก" หรือ "from"
        patterns = [
            r'(?:จาก|from)[:\s]+([\u0E00-\u0E7Fa-zA-Z\s]+)',
            r'(?:ผู้โอน|sender)[:\s]+([\u0E00-\u0E7Fa-zA-Z\s]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_receiver_name(self, text: str) -> str | None:
        """ดึงชื่อผู้รับ"""
        patterns = [
            r'(?:ถึง|to|ผู้รับ|receiver)[:\s]+([\u0E00-\u0E7Fa-zA-Z\s]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_account(self, text: str) -> str | None:
        """ดึงเลขบัญชี"""
        match = self.ACCOUNT_REGEX.search(text)
        return match.group(1) if match else None

    @staticmethod
    def export_json(data: dict[str, Any], output_path: str, indent: int = 2) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        
        logger.info(f"✅ Exported JSON to: {output_path}")

    @staticmethod
    def pretty_print(data: dict[str, Any]) -> None:
        """แสดงข้อมูลในรูปแบบที่อ่านง่าย"""
        print("\n" + "="*60)
        print("📄 SLIP DATA")
        print("="*60)
        for key, value in data.items():
            if key == "raw_ocr":
                print(f"{key:20s}: [ซ่อนเพื่อความชัดเจน]")
            else:
                print(f"{key:20s}: {value}")
        print("="*60 + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

class SlipOCR:

    ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".pdf"}

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        preprocess: bool = True,
        auto_parse: bool = False,
        auto_export: bool = False,
    ):
        """
        Args:
            base_url: None = ใช้ cloud API, "http://..." = self-hosted
            api_key: None = อ่านจาก env, "xxx" = ระบุเอง
            preprocess: ปรับปรุงภาพก่อน OCR
            auto_parse: แปลงผลลัพธ์เป็น JSON อัตโนมัติ
            auto_export: export เป็นไฟล์ JSON อัตโนมัติ (ต้อง auto_parse=True)
        """
        self.preprocess = preprocess
        self.auto_parse = auto_parse
        self.auto_export = auto_export
        self._engine = TyphoonOCREngine(base_url=base_url, api_key=api_key)
        self._parser = SlipParser() if auto_parse else None

    def read(
        self,
        image_path: str,
        output_json: str | None = None
    ) -> str | dict[str, Any]:
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
            raw_text = self._engine.read(processed)
        except Exception as e:
            raise RuntimeError(f"OCR ล้มเหลว: {e}") from e

        # ถ้าไม่ต้อง parse, คืน raw text
        if not self.auto_parse:
            return raw_text

        # Parse เป็น structured data
        data = self._parser.parse(raw_text)

        # Auto export ถ้าเปิดใช้งาน
        if self.auto_export or output_json:
            json_path = output_json or str(path.with_suffix('.json'))
            self._parser.export_json(data, json_path)

        return data



if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python Ocr.py <image_path> [options]")
        print("")
        print("Options:")
        print("  --local              ใช้ self-hosted vllm ที่ localhost:8000")
        print("  --json               แปลงผลลัพธ์เป็น JSON และแสดงบนหน้าจอ")
        print("  --export <path>      export เป็นไฟล์ JSON (default: <image_name>.json)")
        print("")
        print("Examples:")
        print("  python Ocr.py slip.jpg")
        print("  python Ocr.py slip.jpg --json")
        print("  python Ocr.py slip.jpg --json --export output.json")
        print("  python Ocr.py slip.jpg --local --json")
        print("")
        print("Environment:")
        print("  TYPHOON_OCR_API_KEY=your_key   (สำหรับ cloud)")
        sys.exit(1)

    image_path = sys.argv[1]
    use_local = "--local" in sys.argv
    use_json = "--json" in sys.argv
    
    # ดึง path สำหรับ export
    output_json = None
    if "--export" in sys.argv:
        idx = sys.argv.index("--export")
        if idx + 1 < len(sys.argv):
            output_json = sys.argv[idx + 1]
        else:
            print("❌ Error: --export requires a file path")
            sys.exit(1)

    base_url = "http://localhost:8000/v1" if use_local else None
    api_key = "no-key" if use_local else None

    # สร้าง OCR instance
    ocr = SlipOCR(
        base_url=base_url,
        api_key=api_key,
        auto_parse=use_json,
        auto_export=False  # ควบคุมผ่าน output_json parameter
    )
    
    result = ocr.read(image_path, output_json=output_json)

    if use_json:
        # แสดงผลแบบสวยงาม
        parser = SlipParser()
        parser.pretty_print(result)
        
        # แสดง JSON แบบ compact
        print("JSON Output:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # แสดง raw text
        print(result)