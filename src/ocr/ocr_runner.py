"""OCR 引擎主入口。

支持:
- 图片 (jpg/png/bmp/tiff):PaddleOCR
- PDF:pdfplumber 文本层(优先) + 必要时图片页走 PaddleOCR
- PDF 扫描件:每页转图片,逐页走 PaddleOCR

输出统一为 OCRResult。
"""
from __future__ import annotations

import io
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from typing import Union

from src.ocr.schemas import OCRBlock, OCRPage, OCRResult

log = logging.getLogger(__name__)

# 外部 OCR 引擎所在目录(可选)
LEGACY_OCR_SRC = os.getenv("INSURANCE_LEGACY_OCR_SRC", "")

# 支持的扩展
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
PDF_EXTS = {".pdf"}

# 文本层低于此阈值(字符)时,判定为扫描件,降级到图片 OCR
PDF_TEXT_LAYER_THRESHOLD = 50


# 性能优化: PaddleOCR 模型加载慢(~10-30s), 全进程只加载一次
@lru_cache(maxsize=1)
def _get_paddleocr_engine():
    """获取 PaddleOCR 引擎单例。

    PaddleOCR 3.x 模型首次加载慢,实例化和模型下载成本高。
    跨 case 复用同一个 engine 实例,避免重复加载。
    """
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "PaddleOCR 未安装。请先 pip install paddlepaddle paddleocr"
        ) from e

    log.info("[OCR] 首次加载 PaddleOCR 引擎(单例,后续 case 复用)...")
    engine = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="ch",
    )
    log.info("[OCR] PaddleOCR 引擎就绪")
    return engine


# =============================================================
# 图片 OCR
# =============================================================
def _ocr_image_with_paddleocr(image_path: Union[Path, bytes, bytearray]) -> OCRResult:
    """用 PaddleOCR 识别图片(支持本地路径或图片字节流)。

    Args:
        image_path: 文件路径,或传入 bytes 时用 io.BytesIO 包装
    """
    engine = _get_paddleocr_engine()  # 用单例,避免每张图都重新加载

    if isinstance(image_path, (bytes, bytearray)):
        # 字节流
        import numpy as np
        from PIL import Image
        img = Image.open(io.BytesIO(image_path)).convert("RGB")
        arr = np.array(img)
        raw = engine.predict(arr)
    else:
        raw = engine.predict(str(image_path))

    if not raw:
        return OCRResult(source_path=str(image_path), pages=[], engine="paddleocr")

    # PaddleOCR 3.x 返回: list[OCRResult] (每页一个),其中 OCRResult.json 是 dict
    # 结构: result.json["res"] = {"rec_texts": [...], "rec_scores": [...], "dt_polys": [[[x,y]...]]}
    blocks: list[OCRBlock] = []
    full_text_lines: list[str] = []
    confidences: list[float] = []
    pages: list[OCRPage] = []

    for page_idx, page_result in enumerate(raw, 1):
        try:
            res = page_result.json.get("res", {}) if hasattr(page_result, "json") else {}
        except Exception:
            res = {}
        texts = res.get("rec_texts", []) or []
        scores = res.get("rec_scores", []) or []
        polys = res.get("dt_polys", []) or []
        page_blocks: list[OCRBlock] = []
        for i, text in enumerate(texts):
            conf = float(scores[i]) if i < len(scores) else 1.0
            bbox = None
            if i < len(polys):
                poly = polys[i]
                try:
                    if hasattr(poly, "tolist"):
                        poly = poly.tolist()
                    bbox = [(float(p[0]), float(p[1])) for p in poly]
                except (TypeError, IndexError):
                    bbox = None
            block = OCRBlock(text=str(text), bbox=bbox, confidence=conf)
            page_blocks.append(block)
            blocks.append(block)
            full_text_lines.append(str(text))
            confidences.append(conf)
        pages.append(OCRPage(page_no=page_idx, blocks=page_blocks,
                              full_text="\n".join(str(t) for t in texts)))

    full_text = "\n".join(full_text_lines)
    avg_conf = sum(confidences) / len(confidences) if confidences else None
    return OCRResult(
        source_path=str(image_path) if not isinstance(image_path, bytes) else "<bytes>",
        pages=pages,
        full_text=full_text,
        engine="paddleocr",
        confidence_avg=avg_conf,
    )


# =============================================================
# PDF OCR
# =============================================================
def _ocr_pdf_text_layer(path: Path) -> OCRResult | None:
    """先用 pdfplumber 抽 PDF 文本层。返回 None 表示抽不到(可能是扫描件)。"""
    try:
        import pdfplumber  # type: ignore
    except ImportError as e:
        raise RuntimeError("pdfplumber 未安装。请先 pip install pdfplumber") from e

    pages: list[OCRPage] = []
    all_lines: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            blocks = [
                OCRBlock(text=ln, block_type="text")
                for ln in text.split("\n")
                if ln.strip()
            ]
            pages.append(OCRPage(page_no=i, blocks=blocks, full_text=text))
            all_lines.append(text)
    full_text = "\n\n".join(all_lines)
    return OCRResult(
        source_path=str(path),
        pages=pages,
        full_text=full_text,
        engine="pdfplumber",
    )


def _ocr_pdf_scanned(path: Path, dpi: int = 200) -> OCRResult:
    """扫描件 PDF:逐页转图片,逐页走 PaddleOCR。"""
    try:
        from pdf2image import convert_from_path  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "pdf2image 未安装。请先 pip install pdf2image(并安装 poppler-utils)"
        ) from e

    log.info(f"PDF 文本层过薄或缺失,降级为图片 OCR: {path}")
    images = convert_from_path(str(path), dpi=dpi)
    pages: list[OCRPage] = []
    all_text: list[str] = []
    all_conf: list[float] = []

    for i, img in enumerate(images, 1):
        # 转字节流给 _ocr_image_with_paddleocr
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        single = _ocr_image_with_paddleocr(buf.getvalue())
        # 调整 page_no
        for p in single.pages:
            p.page_no = i
        pages.extend(single.pages)
        all_text.append(single.full_text)
        if single.confidence_avg:
            all_conf.append(single.confidence_avg)

    full_text = "\n\n".join(all_text)
    avg = sum(all_conf) / len(all_conf) if all_conf else None
    return OCRResult(
        source_path=str(path),
        pages=pages,
        full_text=full_text,
        engine="paddleocr(pdf-scanned)",
        confidence_avg=avg,
    )


def _ocr_pdf(path: Path) -> OCRResult:
    """PDF 入口:先试文本层,过薄则降级到图片 OCR。"""
    text_result = _ocr_pdf_text_layer(path)
    if text_result is None:
        return _ocr_pdf_scanned(path)
    if len(text_result.full_text.strip()) < PDF_TEXT_LAYER_THRESHOLD:
        log.info(
            f"PDF 文本层仅 {len(text_result.full_text.strip())} 字符,降级为图片 OCR: {path}"
        )
        return _ocr_pdf_scanned(path)
    return text_result


# =============================================================
# 主入口
# =============================================================
def run_ocr(input_path: str | Path, **kwargs: Any) -> OCRResult:
    """主入口:根据文件类型分发到对应引擎。

    Args:
        input_path: 文件路径(.jpg/.png/.pdf 等)
        **kwargs: 透传给具体引擎(如 dpi)

    Returns:
        OCRResult
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"OCR 输入文件不存在: {p}")

    if _is_image(p):
        return _ocr_image_with_paddleocr(p)
    elif _is_pdf(p):
        return _ocr_pdf(p)
    else:
        raise ValueError(
            f"不支持的文件类型: {p.suffix}。仅支持图片 {IMAGE_EXTS} 和 PDF {PDF_EXTS}"
        )


# =============================================================
# 工具函数(供上层聚合器使用)
# =============================================================
def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def _is_pdf(path: Path) -> bool:
    return path.suffix.lower() in PDF_EXTS
