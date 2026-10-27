"""OCR 输出统一 schema。

任何 OCR 引擎(PaddleOCR / Tesseract / 商业 API)的输出都标准化为
`OCRResult`,便于下游聚合 / 抽取。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class OCRBlock(BaseModel):
    """OCR 识别出的一个文本块。"""
    text: str = Field(..., description="块内文本")
    bbox: list[tuple[float, float]] | None = Field(
        None, description="文本块四边形坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]"
    )
    confidence: float | None = Field(None, ge=0.0, le=1.0, description="识别置信度")
    block_type: str | None = Field(
        None, description="块类型:text/title/table/figure"
    )


class OCRPage(BaseModel):
    """单页识别结果。"""
    page_no: int = Field(..., ge=1, description="页码(从 1 开始)")
    blocks: list[OCRBlock] = Field(default_factory=list)
    full_text: str = Field("", description="整页拼接的纯文本(按阅读顺序)")


class OCRResult(BaseModel):
    """单份文档的 OCR 结果。"""
    source_path: str = Field(..., description="原始文件路径")
    doc_type: str | None = Field(
        None, description="文档类型:病理报告/诊断证明/出院小结/检查报告/其他"
    )
    pages: list[OCRPage] = Field(default_factory=list)
    full_text: str = Field("", description="整份文档拼接的纯文本")
    engine: str = Field("unknown", description="使用的 OCR 引擎标识")
    engine_version: str | None = None
    confidence_avg: float | None = Field(None, ge=0.0, le=1.0)

    def all_text(self) -> str:
        """全文(优先用 full_text,否则拼接 pages)。"""
        if self.full_text:
            return self.full_text
        return "\n\n".join(p.full_text for p in self.pages)
