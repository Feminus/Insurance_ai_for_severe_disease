"""OCR 模块入口。
"""
from src.ocr.ocr_runner import run_ocr
from src.ocr.schemas import OCRBlock, OCRPage, OCRResult

__all__ = ["run_ocr", "OCRResult", "OCRPage", "OCRBlock"]
