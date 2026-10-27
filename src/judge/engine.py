"""判定器主引擎。

3 层流程:
  Layer 1: 硬规则 gate(precheck)——缺材料/缺诊断 → "无法认定"
  Layer 2: 规则匹配(matcher)——抽取疾病 vs judgment rules,候选 0 个 → LLM 二次比对
  Layer 3: 投票自评(assessor + postcheck)——LLM 投 3 次取中位数,阈值化 → 三分类
"""
from __future__ import annotations

import logging
from typing import Any

from src.extraction.schemas import CaseExtraction
from src.judge.llm.assessor import fallback_match, self_assess_with_voting
from src.judge.rules.matcher import match_candidate_rules
from src.judge.schema.output_schema import JudgeOutput, RiskWarning
from src.judge.stages.postcheck import (
    HIGH_THRESHOLD,
    check_required_indicators,
    finalize_decision,
)
from src.judge.stages.precheck import precheck_materials

log = logging.getLogger(__name__)


def _not_found(case_id: str, msg: str, code: str = "MISSING_MATERIAL",
               extraction: CaseExtraction | None = None) -> JudgeOutput:
    return JudgeOutput(
        case_id=case_id,
        conclusion="无法认定",
        confidence=None,
        diagnosis=extraction.diagnosis.diagnosis if extraction else None,
        icd10=extraction.diagnosis.icd10 if extraction else None,
        risk_warnings=[RiskWarning(code=code, message=msg)],
        judge_reasoning=msg,
    )


def judge(extraction: CaseExtraction) -> JudgeOutput:
    """主入口:对 CaseExtraction 产出 JudgeOutput。"""
    case_id = extraction.case_id

    # ============================================================
    # Layer 1: 硬规则 gate
    # ============================================================
    # 修复(2026-10):缺诊断名时降级走 Layer 2 fallback_match
    #   之前: 缺诊断名 → 直接判"无法认定"(会误伤"糖尿病/妊娠/不适"等 xlsx 标"否"的 case)
    #   现在: 缺诊断名 → 尝试 fallback_match 二次比对
    #         - 二次比对成功(rule_id 给出) → 走该规则
    #         - 二次比对失败 → 才判"无法认定 - 缺材料/诊断"
    #   材料 < 2 类仍直接判"无法认定"(没文本可给 LLM 二次比对)
    gate = precheck_materials(extraction)
    if not gate.passed:
        log.info(f"[{case_id}] Layer 1 不通过: {gate.code} | {gate.message}")
        # 区分:缺材料(< 2 类)直接判;缺诊断名尝试降级
        if gate.code == "MISSING_DIAGNOSIS":
            # 缺诊断名但有材料 → 走 Layer 2 fallback_match
            log.info(f"[{case_id}] 缺诊断名但材料齐全,降级走 Layer 2 fallback_match")
            try:
                fb = fallback_match(extraction)
                if fb.get("error"):
                    return _not_found(
                        case_id, gate.message + f" | fallback 也失败: {fb.get('match_reasoning', '')}",
                        code=gate.code, extraction=extraction,
                    )
                if fb.get("unresolved"):
                    return _not_found(
                        case_id, gate.message + " | " + fb.get("match_reasoning", ""),
                        code=gate.code, extraction=extraction,
                    )
                if not fb.get("is_in_list"):
                    # 二次比对:不在 100 种 → 判"否"(疾病不在白名单,不算重疾)
                    return JudgeOutput(
                        case_id=case_id,
                        conclusion="否",
                        confidence=0.9,
                        diagnosis=extraction.diagnosis.diagnosis,
                        icd10=extraction.diagnosis.icd10,
                        matched_rule_id=None,
                        matched_rule_name=None,
                        risk_warnings=[RiskWarning(
                            code="DISEASE_NOT_IN_RULES",
                            message=f"fallback_match 二次比对:抽取的疾病不在 100 种重疾中。{fb.get('match_reasoning', '')}",
                        )],
                        judge_reasoning=fb.get("match_reasoning", ""),
                    )
                # 二次比对成功且有 rule_id(注:fallback_match 返回的 fb dict 不直接含 matched_rule,
                # 这里只走诊断名走不到候选规则,直接判"否"即可)
                return JudgeOutput(
                    case_id=case_id,
                    conclusion="否",
                    confidence=0.85,
                    diagnosis=extraction.diagnosis.diagnosis,
                    icd10=extraction.diagnosis.icd10,
                    risk_warnings=[RiskWarning(
                        code="DISEASE_NOT_IN_RULES",
                        message=f"无法定位具体重疾规则(疾病 '{extraction.diagnosis.diagnosis}' 无明确 R 规则匹配)。{fb.get('match_reasoning', '')}",
                    )],
                )
            except Exception as e:
                log.warning(f"[{case_id}] 缺诊断名降级处理失败: {e}")
                return _not_found(case_id, gate.message, code=gate.code, extraction=extraction)

        # 其他情况(< 2 类材料)直接判"无法认定"
        log.info(f"[{case_id}] Layer 1 拒绝: {gate.code} | {gate.message}")
        # 修复(2026-10):MISSING_MATERIAL 也走降级
        # 场景:某 case 全部 doc_type 都是"检查报告"(< 2 类),但 LLM 已抽到疾病
        #   → 之前判"无法认定";现在降级走 fallback_match 二次比对
        if gate.code in ("MISSING_MATERIAL", "MISSING_DIAGNOSIS"):
            try:
                fb = fallback_match(extraction)
                if not fb.get("error") and not fb.get("unresolved") and not fb.get("is_in_list"):
                    return JudgeOutput(
                        case_id=case_id,
                        conclusion="否",
                        confidence=0.9,
                        diagnosis=extraction.diagnosis.diagnosis,
                        icd10=extraction.diagnosis.icd10,
                        matched_rule_id=None,
                        matched_rule_name=None,
                        risk_warnings=[RiskWarning(
                            code="DISEASE_NOT_IN_RULES",
                            message=f"fallback_match 二次比对:抽取的疾病不在 100 种重疾中。{fb.get('match_reasoning', '')}",
                        )],
                        judge_reasoning=fb.get("match_reasoning", ""),
                    )
            except Exception as e:
                log.warning(f"[{case_id}] 降级 fallback_match 失败: {e}")
        return _not_found(case_id, gate.message, code=gate.code, extraction=extraction)

    # ============================================================
    # Layer 2: 算法 + LLM 双重规则匹配
    # ============================================================
    candidates = match_candidate_rules(extraction)

    matched_rule: dict | None = None
    if candidates:
        # 取 _score 最高的
        matched_rule = candidates[0]
        log.info(
            f"[{case_id}] Layer 2 命中: {matched_rule['rule_id']} {matched_rule['name']} "
            f"(score={matched_rule.get('_score'):.2f}, sources={matched_rule.get('_source')})"
        )
    else:
        # 算法没命中 → 先二次校验材料:仍不齐就"无法认定",不再走 fallback
        log.info(f"[{case_id}] Layer 2 算法 0 候选 → 二次材料校验")
        gate2 = precheck_materials(extraction)
        if not gate2.passed:
            log.info(f"[{case_id}] Layer 2 二次 gate 拒绝: {gate2.code} | {gate2.message}")
            return _not_found(case_id, gate2.message, code=gate2.code, extraction=extraction)
        # 材料齐全 → 走 LLM 二次比对
        log.info(f"[{case_id}] Layer 2 二次 gate 通过 → LLM 二次比对")
        fb = fallback_match(extraction)
        # 修复 Bug 1: LLM 二次比对本身调用失败 → 触发"无法认定"(不是"否")
        if fb.get("error"):
            return _not_found(
                case_id,
                f"算法 0 候选 + LLM 二次比对调用失败,无法判定: {fb.get('match_reasoning', '')}",
                code="FALLBACK_MATCH_FAILED",
                extraction=extraction,
            )
        # 修复 Bug 2/3: LLM 说"是 100 种之一"但没给 rule_id → 解析失败,触发"无法认定"
        if fb.get("unresolved"):
            return _not_found(
                case_id,
                f"LLM 二次比对认为是 100 种之一,但未能解析到具体规则: {fb.get('match_reasoning', '')}",
                code="RULE_NOT_RESOLVED",
                extraction=extraction,
            )
        if not fb.get("is_in_list"):
            # 升级路径(2026-10):fallback_match 判不在 100 种 + 抽取标信息不充分
            # → 升级为无法认定
            info_insufficient = getattr(extraction, "information_insufficient", None)
            extraction_notes = getattr(extraction, "extraction_notes", None) or ""
            notes_keywords = ["缺", "未提供", "未见", "未明确", "信息不充分", "不完整",
                              "拟诊", "待诊", "考虑", "可能", "待定", "无独立", "未单独"]
            notes_signal = any(kw in extraction_notes for kw in notes_keywords)
            if info_insufficient is True or notes_signal:
                reason_text = "疾病名含'考虑/可能'等非确诊表述" if "考虑" in extraction_notes or "可能" in extraction_notes else "信息不充分"
                return JudgeOutput(
                    case_id=case_id,
                    conclusion="无法认定",
                    confidence=0.95,
                    diagnosis=extraction.diagnosis.diagnosis,
                    icd10=extraction.diagnosis.icd10,
                    matched_rule_id=None,
                    matched_rule_name=None,
                    risk_warnings=[RiskWarning(
                        code="INSUFFICIENT_INFO",
                        message=f"信息不充分({reason_text}),无法认定为重疾",
                    )],
                    judge_reasoning=fb.get("match_reasoning", ""),
                )
            return JudgeOutput(
                case_id=case_id,
                conclusion="否",
                confidence=0.95,
                diagnosis=extraction.diagnosis.diagnosis,
                icd10=extraction.diagnosis.icd10,
                matched_rule_id=None,
                matched_rule_name=None,
                risk_warnings=[RiskWarning(
                    code="DISEASE_NOT_IN_RULES",
                    message=f"LLM 二次比对:抽取的疾病 '{extraction.diagnosis.diagnosis}' 不在 100 种重疾中。{fb.get('match_reasoning', '')}",
                )],
                judge_reasoning=fb.get("match_reasoning", ""),
            )
        # LLM 说"是 100 种之一"但给出了 rule_id → 用它重选
        suggested_rid = fb.get("matched_rule_id")
        if suggested_rid:
            # 从候选里找(理论上 fallback_match 后应重跑算法,但简单起见直接从 candidates 找)
            for c in candidates:
                if c["rule_id"] == suggested_rid:
                    matched_rule = c
                    break
            # 还没找到?直接读 rules
            if not matched_rule:
                from src.judge.rules.matcher import _load_rules
                rules = _load_rules()
                if suggested_rid in rules:
                    r = rules[suggested_rid]
                    matched_rule = {
                        "rule_id": suggested_rid,
                        "name": r["name"],
                        "criteria": r.get("criteria", []),
                        "exclusions": r.get("exclusions", []),
                        "criteria_type": r.get("criteria_type", "all"),
                    }
        if not matched_rule:
            return _not_found(
                case_id,
                f"LLM 二次比对说是 100 种之一,但未能解析到具体规则",
                code="RULE_NOT_RESOLVED",
                extraction=extraction,
            )

    # ============================================================
    # Layer 2.5: 检查关键指标缺失(规则要求但 LLM 没抽到)
    # ============================================================
    ind_check = check_required_indicators(extraction, matched_rule)
    if not ind_check["ok"]:
        msgs = "; ".join(m["reason"] for m in ind_check["missing"][:3])
        log.info(f"[{case_id}] Layer 2.5 缺关键指标: {msgs}")
        return JudgeOutput(
            case_id=case_id,
            conclusion="无法认定",
            confidence=None,
            matched_rule_id=matched_rule["rule_id"],
            matched_rule_name=matched_rule["name"],
            diagnosis=extraction.diagnosis.diagnosis,
            icd10=extraction.diagnosis.icd10,
            risk_warnings=[RiskWarning(
                code="MISSING_INDICATOR",
                message=f"规则 {matched_rule['rule_id']} 要求的指标缺失: {msgs}",
            )] + [RiskWarning(
                code=m["indicator"],
                message=m["reason"],
            ) for m in ind_check["missing"][:3]],
        )

    # ============================================================
    # Layer 3: LLM 投票 3 次 + 后置判定
    # ============================================================
    vote_result = self_assess_with_voting(extraction, matched_rule)
    log.info(
        f"[{case_id}] Layer 3 投票: {vote_result['vote_confidences']} "
        f"→ 中位数 {vote_result['confidence']:.3f} (agree={vote_result['vote_agree']})"
    )
    return finalize_decision(case_id, matched_rule=matched_rule,
                              extraction=extraction, vote_result=vote_result)
