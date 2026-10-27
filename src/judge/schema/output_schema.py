"""判定器输出 schema。

结论三分类:{是, 否, 无法认定}
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Conclusion = Literal["是", "否", "无法认定"]


class EvidenceQuote(BaseModel):
    """依据引用:从原始材料中引用的关键片段。"""
    field: str = Field(..., description="被引用的字段名,如'诊断' / '病理'")
    quote: str = Field(..., description="原文片段")
    source: str = Field(..., description="来源文件 / 文档类型")


class RiskWarning(BaseModel):
    """风险提示:可能导致判定错误的因素。"""
    code: str = Field(
        ...,
        description="风险代码:MISSING_MATERIAL / MISSING_INDICATOR / LOW_CONFIDENCE / MID_CONFIDENCE / VOTE_DIVERGENCE / DISEASE_NOT_IN_RULES / RULE_PARTIAL_MATCH ...",
    )
    message: str = Field(..., description="人类可读描述")


class JudgeOutput(BaseModel):
    """判定器最终输出。"""
    case_id: str = Field(..., description="案例 ID")
    conclusion: Conclusion = Field(..., description="最终结论:是 / 否 / 无法认定")

    # 置信度(LLM 投票 3 次中位数);无法认定时可能为 None
    confidence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="置信度评分;缺材料 / 缺指标时为 None(对应'无法认定')",
    )

    # 命中规则(若有)
    matched_rule_id: str | None = Field(
        None, description="命中的 R 规则 ID(如 R001);无规则命中时为 None"
    )
    matched_rule_name: str | None = Field(None, description="命中规则名")

    # 诊断信息(从抽取层透传)
    diagnosis: str | None = Field(None, description="本次诊断名称")
    icd10: str | None = Field(None, description="本次诊断 ICD-10 编码")

    # 判定依据
    evidence_quotes: list[EvidenceQuote] = Field(
        default_factory=list, description="依据引用列表"
    )

    # 风险提示
    risk_warnings: list[RiskWarning] = Field(
        default_factory=list, description="风险提示列表"
    )

    # 投票细节(供调试 / 审计)
    vote_rounds: int = Field(0, ge=0, description="LLM 投票轮数(通常 3)")
    vote_confidences: list[float] = Field(
        default_factory=list, description="每轮的 confidence(用于审计)"
    )
    vote_agree: bool = Field(
        True, description="3 轮 confidence 是否一致(差 ≤ 0.05)"
    )

    # 内部追溯
    hit_hard_rules: list[str] = Field(
        default_factory=list, description="命中的硬规则 ID(用于审计)"
    )
    judge_reasoning: str = Field("", description="LLM 推理过程摘要")
