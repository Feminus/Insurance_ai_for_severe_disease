"""评分器输入 schema。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class JudgeInput(BaseModel):
    """评分器输入 = LLM 抽取层的输出。"""

    case_id: str = Field(..., description="案例唯一标识")
    merged_text: str = Field("", description="聚合后的全文")
    doc_types: list[str] = Field(
        default_factory=list,
        description="材料类型列表,如 ['病理报告','诊断证明','出院小结','检查报告']",
    )

    # LLM 抽取的关键字段(字段待用户拍板,此处先放核心几个)
    diagnosis: str | None = Field(None, description="本次诊断名称")
    icd10: str | None = Field(None, description="本次诊断 ICD-10 编码")
    pathology: str | None = Field(None, description="病理所见")
    pathology_diagnosis: str | None = Field(None, description="病理诊断")
    explicit_negations: list[str] = Field(
        default_factory=list, description="显性否定表述(排除项)"
    )

    # 既往信息(作为佐证 / 风险提示,不再驱动"非首次"判定)
    historical_records: list[str] = Field(
        default_factory=list, description="既往就诊记录摘要"
    )
