"""判定器模块入口。"""
from src.judge.engine import judge
from src.judge.llm.assessor import fallback_match, self_assess_with_voting
from src.judge.rules.matcher import match_candidate_rules
from src.judge.schema.output_schema import (
    Conclusion,
    EvidenceQuote,
    JudgeOutput,
    RiskWarning,
)
from src.judge.stages.postcheck import (
    HIGH_THRESHOLD,
    check_required_indicators,
    finalize_decision,
)
from src.judge.stages.precheck import precheck_materials

__all__ = [
    "judge",
    "self_assess_with_voting",
    "fallback_match",
    "match_candidate_rules",
    "precheck_materials",
    "check_required_indicators",
    "finalize_decision",
    "HIGH_THRESHOLD",
    "JudgeOutput",
    "Conclusion",
    "EvidenceQuote",
    "RiskWarning",
]
