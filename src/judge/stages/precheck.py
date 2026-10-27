"""Layer 1: 硬规则 gate(纯算法,无 LLM)。

规则:
- 至少 2 类诊断性材料(病理报告 / 诊断证明 / 出院小结)
- 且必须含 诊断证明 或 病理报告 至少一种
- 诊断名不能为空

否则 → "无法认定"。
"""
from __future__ import annotations

from dataclasses import dataclass

from src.extraction.schemas import CaseExtraction


@dataclass
class GateResult:
    passed: bool
    code: str  # OK / MISSING_DIAGNOSIS / MISSING_MATERIAL
    message: str
    missing_doc_types: list[str] = None


DIAG_TYPES = {"病理报告", "诊断证明", "出院小结", "检查报告"}
KEY_TYPES = {"诊断证明", "病理报告", "检查报告"}  # 至少一种(关键材料)


def precheck_materials(extraction: CaseExtraction) -> GateResult:
    """Layer 1.1: 关键材料是否齐全(至少 2 类诊断性材料)。"""
    doc_types = set(extraction.doc_types or [])

    # 1. 至少 2 类诊断材料
    # 修复(2026-10):"其他"也算入总数(LLM 经常把"知情同意书"归到"其他",
    #   但实际有材料);只要总数 ≥ 2 且至少一种是 4 类核心之一即可
    diag_present = doc_types & DIAG_TYPES
    if len(doc_types) < 2 or len(diag_present) < 1:
        return GateResult(
            passed=False,
            code="MISSING_MATERIAL",
            message=(
                f"诊断性材料不足 2 类,无法认定。需至少 2 类(病理报告/诊断证明/出院小结/检查报告),"
                f"实际: {sorted(doc_types) or '无'}"
            ),
            missing_doc_types=sorted(DIAG_TYPES - doc_types),
        )

    # 2. 诊断证明 或 病理报告 至少一种(放宽到包含检查报告)
    if not (doc_types & (KEY_TYPES | {"检查报告"})):
        return GateResult(
            passed=False,
            code="MISSING_MATERIAL",
            message=(
                f"缺少关键材料(诊断证明 / 病理报告 / 检查报告 至少一种),无法认定。实际: {sorted(doc_types)}"
            ),
            missing_doc_types=sorted(KEY_TYPES - doc_types),
        )

    # 3. 诊断名为空
    if not (extraction.diagnosis.diagnosis or "").strip():
        return GateResult(
            passed=False,
            code="MISSING_DIAGNOSIS",
            message="抽取阶段未识别出疾病诊断名称",
            missing_doc_types=[],
        )

    return GateResult(passed=True, code="OK", message="")
