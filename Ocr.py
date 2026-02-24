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
        r'(?:จำนวน|amount|total|ยอดโอน)[:\s]*([\d,]+\.?\d{0,2})\s*(?:บาท|baht|thb)?',
        re.IGNORECASE
    )
    
    DATE_REGEX = re.compile(
        r'(\d{1,2})\s*(?:ก\.พ\.|ม\.ค\.|มี\.ค\.|เม\.ย\.|พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|ก\.ย\.|ต\.ค\.|พ\.ย\.|ธ\.ค\.)\s*(\d{2,4})|(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})'
    )
    
    TIME_REGEX = re.compile(
        r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(?:น\.|AM|PM)?'
    )
    
    REF_REGEX = re.compile(
        r'(?:เลขที่รายการ|ref|อ้างอิง|หมายเลข|reference)[.\s:]*([A-Z0-9]{10,30})',
        re.IGNORECASE
    )
    
    # บัญชีธนาคารทั่วไป (xxx-x-xxxxx-x)
    BANK_ACCOUNT_REGEX = re.compile(
        r'(xxx[\-]?x[\-]?x\d{4}[\-]?x|\d{3}[\-]?\d{1}[\-]?\d{4,5}[\-]?\d{1})'
    )
    
    # รหัสผู้รับ/Merchant ID (ตัวเลขยาว 10-20 หลัก)
    MERCHANT_ID_REGEX = re.compile(
        r'(?<!เลขที่รายการ:\s)(?<!ref\s)(\d{12,20})(?!\s*บาท)',
        re.IGNORECASE
    )

    def parse(self, raw_text: str) -> dict[str, Any]:

        text = raw_text.lower()

        return {
            "sender_name": self._extract_sender_name(raw_text),
            "sender_account": self._extract_sender_account(raw_text),
            "bank_name": self._extract_bank_name(text),
            "amount": self._extract_amount(text),
            "slip_date": self._extract_date(text),
            "slip_time": self._extract_time(text),
            "ref_no": self._extract_ref_no(raw_text),
            "receiver_name": self._extract_receiver_name(raw_text),
            "receiver_account": self._extract_receiver_account(raw_text),
            "raw_ocr": raw_text,
        }

    def _extract_amount(self, text: str) -> float | None:
        # ลองหา pattern ที่ชัดเจนก่อน
        match = self.AMOUNT_REGEX.search(text)
        if match:
            amount_str = match.group(1).replace(',', '')
            try:
                return float(amount_str)
            except ValueError:
                pass
        
        # Fallback: หาตัวเลขที่มี .00 และไม่ใช่เลขยาวเกินไป
        fallback_pattern = re.compile(r'(\d{1,6}\.?\d{0,2})\s*(?:บาท|baht)')
        matches = fallback_pattern.findall(text)
        if matches:
            amounts = []
            for m in matches:
                try:
                    amt = float(m.replace(',', ''))
                    # กรองเฉพาะจำนวนที่สมเหตุสมผล (0.01 - 999,999.99)
                    if 0.01 <= amt <= 999999.99:
                        amounts.append(amt)
                except ValueError:
                    pass
            # คืนจำนวนที่มากที่สุด (มักเป็นยอดโอนจริง)
            return max(amounts) if amounts else None
        
        return None

    def _extract_bank_name(self, text: str) -> str | None:
        for bank, pattern in self.BANK_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return bank
        return None

    def _extract_date(self, text: str) -> str | None:
        # แปลงเดือนภาษาไทย
        thai_months = {
            'ม.ค.': 1, 'ก.พ.': 2, 'มี.ค.': 3, 'เม.ย.': 4, 'พ.ค.': 5, 'มิ.ย.': 6,
            'ก.ค.': 7, 'ส.ค.': 8, 'ก.ย.': 9, 'ต.ค.': 10, 'พ.ย.': 11, 'ธ.ค.': 12
        }
        
        # ลอง pattern ภาษาไทยก่อน (22 ก.พ. 69)
        thai_pattern = re.compile(r'(\d{1,2})\s*(ม\.ค\.|ก\.พ\.|มี\.ค\.|เม\.ย\.|พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|ก\.ย\.|ต\.ค\.|พ\.ย\.|ธ\.ค\.)\s*(\d{2,4})')
        match = thai_pattern.search(text)
        if match:
            day = int(match.group(1))
            month = thai_months.get(match.group(2))
            year = int(match.group(3))
            
            # แปลง พ.ศ. เป็น ค.ศ.
            if year < 100:
                year += 2500  # 69 -> 2569
            if year > 2500:
                year -= 543  # 2569 -> 2026
            
            try:
                return f"{year:04d}-{month:02d}-{day:02d}"
            except (ValueError, TypeError):
                pass
        
        # Fallback: pattern ปกติ (DD/MM/YYYY)
        match = self.DATE_REGEX.search(text)
        if match:
            groups = match.groups()
            # ถ้า match แบบไทยแล้ว groups จะเป็น (day, month_abbr, year, None, None)
            # ถ้า match แบบปกติ groups จะเป็น (None, None, day, month, year)
            if groups[2] and groups[3]:  # แบบปกติ
                day, month, year = int(groups[2]), int(groups[3]), int(groups[4])
                
                # แปลง พ.ศ. เป็น ค.ศ.
                if year > 2500:
                    year -= 543
                elif year < 100:
                    year += 2000
                
                try:
                    return f"{year:04d}-{month:02d}-{day:02d}"
                except ValueError:
                    pass
        
        return None

    def _extract_time(self, text: str) -> str | None:
        match = self.TIME_REGEX.search(text)
        if match:
            hour, minute, second = match.groups()
            second = second or "00"
            return f"{int(hour):02d}:{int(minute):02d}:{int(second):02d}"
        return None

    def _extract_ref_no(self, text: str) -> str | None:
        match = self.REF_REGEX.search(text)
        return match.group(1) if match else None

    def _extract_sender_name(self, text: str) -> str | None:
        # กรอง noise ออก (คำอธิบายรูป, การ์ตูน, etc.)
        text_clean = text
        
        # ลบ figure tags และเนื้อหาข้างใน
        text_clean = re.sub(r'<figure>.*?</figure>', '', text_clean, flags=re.DOTALL)
        
        # ลบคำที่เป็น noise
        noise_words = ['one piece', 'ocean of fire', 'ยืนอยู่ทาง', 'ภาพประกอบ', 'qr code', 'เหรียญทอง', 'ตัวละคร']
        for noise in noise_words:
            text_clean = re.sub(noise, '', text_clean, flags=re.IGNORECASE)
        
        # ตัวอย่างเบื้องต้น: หาชื่อที่อยู่หลังคำว่า "จาก" หรือ "from" หรือชื่อบุคคลไทย
        patterns = [
            r'(?:จาก|from)[:\s]+((?:นาย|นาง|นางสาว|Mr\.|Mrs\.|Ms\.)\s+[\u0E00-\u0E7Fa-zA-Z\s]+)',
            r'(?:ผู้โอน|sender)[:\s]+((?:นาย|นาง|นางสาว|Mr\.|Mrs\.|Ms\.)\s+[\u0E00-\u0E7Fa-zA-Z\s]+)',
            r'^((?:นาย|นาง|นางสาว)\s+[\u0E00-\u0E7F]+(?:\s+[\u0E00-\u0E7F]+){1,2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_clean, re.IGNORECASE | re.MULTILINE)
            if match:
                name = match.group(1).strip()
                # ทำความสะอาด: ลบ newlines และช่องว่างซ้ำ
                name = ' '.join(name.split())
                # ตรวจสอบว่าชื่อไม่ใช่ noise
                if len(name) < 100 and not any(n in name.lower() for n in noise_words):
                    return name
        
        return None

    def _extract_receiver_name(self, text: str) -> str | None:
        # กรอง figure tags
        text_clean = re.sub(r'<figure>.*?</figure>', '', text, flags=re.DOTALL)
        
        patterns = [
            r'(?:ถึง|to|ผู้รับ|receiver)[:\s]+([\u0E00-\u0E7Fa-zA-Z\s&\-\.]+)',
            r'(?:บริษัท|ห้าง|ร้าน)\s+([\u0E00-\u0E7Fa-zA-Z\s&\-\.]+)',
            # สำหรับ K+ format: ชื่อร้านอยู่บรรทัดถัดจาก logo/brand
            r'(?:Tops|7-Eleven|Lotus|Big C|Central|Family Mart|Lawson|Makro)\s*(?:daily)?\n?([\u0E00-\u0E7Fa-zA-Z\s&\-\.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_clean, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # ทำความสะอาด: ลบ newlines และช่องว่างซ้ำ
                name = ' '.join(name.split())
                # กรองชื่อที่สั้นเกินไป หรือยาวเกินไป
                if 3 <= len(name) <= 100:
                    return name
        
        return None

    def _extract_sender_account(self, text: str) -> str | None:
        match = self.BANK_ACCOUNT_REGEX.search(text)
        return match.group(1) if match else None

    def _extract_receiver_account(self, text: str) -> str | None:

        # ลองหา merchant ID ก่อน (ตัวเลขยาวที่ไม่ใช่ ref no.)
        merchant_match = self.MERCHANT_ID_REGEX.search(text)
        if merchant_match:
            merchant_id = merchant_match.group(1)
            # ตรวจสอบว่าไม่ใช่เลขที่รายการ
            if merchant_id and len(merchant_id) >= 12:
                return merchant_id
        
        # ถ้าไม่เจอ merchant ID ให้หาบัญชีปกติ (แต่ไม่ใช่ของ sender)
        # สำหรับกรณีโอนให้บุคคลทั่วไป
        matches = self.BANK_ACCOUNT_REGEX.findall(text)
        if len(matches) > 1:
            # ถ้ามีหลายบัญชี เอาตัวที่ 2 (ตัวแรกมักเป็นของ sender)
            return matches[1]
        
        return None

    @staticmethod
    def export_json(data: dict[str, Any], output_path: str, indent: int = 2) -> None:

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        
        logger.info(f"✅ Exported JSON to: {output_path}")

    @staticmethod
    def pretty_print(data: dict[str, Any]) -> None:
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


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

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