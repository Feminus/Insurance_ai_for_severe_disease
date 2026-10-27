"""聚合器输出 schema。

把多份 OCR 结果归并为一份,保留每份文件的源信息,
并生成带文件名前缀的 merged_text(供 LLM 抽取阶段使用)。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.ocr.schemas import OCRResult


class MergedDocument(BaseModel):
    """聚合后的单份文件信息(保留溯源信息)。"""
    source_path: str = Field(..., description="原始文件路径/文件名")
    doc_type: str | None = Field(
        None, description="文档类型:病理报告/诊断证明/出院小结/检查报告/其他"
    )
    engine: str = Field(..., description="OCR 引擎")
    engine_version: str | None = None
    page_count: int = Field(0, ge=0, description="页数")
    confidence_avg: float | None = Field(None, ge=0.0, le=1.0)
    full_text: str = Field("", description="本份文件的全文(已含分隔)")
    char_count: int = Field(0, ge=0, description="字符数")


class MergedCase(BaseModel):
    """聚合后的单条案例(供 LLM 抽取阶段使用)。"""
    case_id: str = Field(..., description="案例 ID")
    documents: list[MergedDocument] = Field(
        default_factory=list, description="每份文件的聚合信息"
    )
    merged_text: str = Field("", description="带文件名前缀的全文(供 LLM 抽取)")
    total_chars: int = Field(0, ge=0, description="总字符数")
    has_pathology: bool = Field(False, description="是否含病理报告")
    has_diagnosis_cert: bool = Field(False, description="是否含诊断证明")
    has_discharge_summary: bool = Field(False, description="是否含出院小结")
    has_exam_report: bool = Field(False, description="是否含检查报告")

    def document_types(self) -> list[str]:
        """返回去重的文件类型列表(保留顺序)。"""
        seen: set[str] = set()
        out: list[str] = []
        for d in self.documents:
            if d.doc_type and d.doc_type not in seen:
                seen.add(d.doc_type)
                out.append(d.doc_type)
        return out
