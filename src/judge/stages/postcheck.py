"""Layer 3: 后置判定。

语义说明:
- "无法认定" 只由 Layer 1 / Layer 2.5 硬规则触发(缺材料/缺诊断/缺指标/疾病不在规则)
- 一旦进入 Layer 3 投票,confidence 只在 "是 / 否" 之间决策
- exclusion 命中(任何)→ 直接 "否"(硬规则)
- criteria 部分缺失: 加 WARNING,但不直接判否 — 让 confidence 主导
- confidence ≥ HIGH → "是"
- confidence < HIGH → "否"
- 投票分歧只加 VOTE_DIVERGENCE 风险码,不改变结论
"""
from __future__ import annotations

from src.judge.schema.output_schema import JudgeOutput, RiskWarning


# confidence 阈值:只用于 是/否 二分
# 阈值 0.70:42 例"是"中 35 例(83%)conf ≥ 0.70,9 例 conf < 0.70 是边界案例
# 提升到 0.70 让 LLM 自评不自信的 case 落回"否",减少过宽判定
HIGH_THRESHOLD = 0.70


def finalize_decision(
    case_id: str,
    *,
    matched_rule: dict | None,
    extraction,
    vote_result: dict,
) -> JudgeOutput:
    """根据投票自评结果 + 规则,产出 "是 / 否" 二分类 JudgeOutput。

    规则语义:
    - exclusion 命中(任一)→ 硬规则判"否",不依赖 confidence
    - criteria 部分缺失: 加 WARNING 风险码,不直接判否,让 confidence 主导
    - confidence ≥ HIGH_THRESHOLD → "是"
    - confidence < HIGH_THRESHOLD → "否"

    Args:
        case_id: 案例 ID
        matched_rule: 命中规则(可能 None)
        extraction: CaseExtraction(用于透传 diagnosis / icd10)
        vote_result: assessor.self_assess_with_voting() 返回值
    """
    rule_id = matched_rule["rule_id"] if matched_rule else None
    rule_name = matched_rule["name"] if matched_rule else None
    confidence = vote_result.get("confidence")
    criterion_pass = vote_result.get("criterion_pass", {}) or {}
    exclusion_hit = vote_result.get("exclusion_hit", []) or []
    vote_agree = vote_result.get("vote_agree", True)
    vote_confs = vote_result.get("vote_confidences", [])

    risks: list[RiskWarning] = []

    # 投票分歧(只警告,不改变结论)
    if not vote_agree and len(vote_confs) >= 2:
        risks.append(
            RiskWarning(
                code="VOTE_DIVERGENCE",
                message=f"LLM 投票 {len(vote_confs)} 轮 confidence 范围 "
                        f"{min(vote_confs):.2f}–{max(vote_confs):.2f},差异 > 0.05",
            )
        )

    # criteria 部分缺失 → 加 WARNING,不直接判否(让 confidence 主导)
    if criterion_pass and any(v is False for v in criterion_pass.values()):
        failed = [k for k, v in criterion_pass.items() if v is False]
        risks.append(
            RiskWarning(
                code="CRITERIA_PARTIAL_MISS",
                message=f"规则 {rule_id} 的 criteria 中第 {','.join(failed)} 项缺失,"
                        f"以 confidence 主导判定,建议人工核验",
            )
        )

    # ---- 主判定逻辑(优先级) ----
    # 0. 既往病史(诊断发生在最近报告材料 1 年前 + 无复发/治疗记录) → 否
    #    这是新增的硬规则:即使 confidence 高,既往病史不参与本次重疾判定
    is_historical = getattr(extraction.diagnosis, "is_historical", None)
    if is_historical is True:
        risks.append(
            RiskWarning(
                code="HISTORICAL_DIAGNOSIS",
                message=(
                    f"诊断 '{extraction.diagnosis.diagnosis}' 标记为既往病史"
                    f"(诊断日期与最近报告材料间隔 > 1 年),不属于本次重疾认定范围"
                ),
            )
        )
        return JudgeOutput(
            case_id=case_id,
            conclusion="否",
            confidence=confidence,
            matched_rule_id=rule_id,
            matched_rule_name=rule_name,
            diagnosis=extraction.diagnosis.diagnosis,
            icd10=extraction.diagnosis.icd10,
            risk_warnings=risks,
            vote_rounds=len(vote_confs),
            vote_confidences=vote_confs,
            vote_agree=vote_agree,
            hit_hard_rules=[rule_id] if rule_id else [],
            judge_reasoning=vote_result.get("reasoning", ""),
        )

    # 1. exclusion 命中任一 → 否(硬规则,最高优先级)
    if exclusion_hit:
        risks.append(
            RiskWarning(
                code="EXCLUSION_HIT",
                message=f"规则 {rule_id} 的 exclusions 中第 {','.join(map(str, exclusion_hit))} 项命中",
            )
        )
        return JudgeOutput(
            case_id=case_id,
            conclusion="否",
            confidence=confidence,
            matched_rule_id=rule_id,
            matched_rule_name=rule_name,
            diagnosis=extraction.diagnosis.diagnosis,
            icd10=extraction.diagnosis.icd10,
            risk_warnings=risks,
            vote_rounds=len(vote_confs),
            vote_confidences=vote_confs,
            vote_agree=vote_agree,
            hit_hard_rules=[rule_id] if rule_id else [],
            judge_reasoning=vote_result.get("reasoning", ""),
        )

    # 2. confidence ≥ HIGH → 是
    if confidence is not None and confidence >= HIGH_THRESHOLD:
        return JudgeOutput(
            case_id=case_id,
            conclusion="是",
            confidence=confidence,
            matched_rule_id=rule_id,
            matched_rule_name=rule_name,
            diagnosis=extraction.diagnosis.diagnosis,
            icd10=extraction.diagnosis.icd10,
            risk_warnings=risks,
            vote_rounds=len(vote_confs),
            vote_confidences=vote_confs,
            vote_agree=vote_agree,
            hit_hard_rules=[rule_id] if rule_id else [],
            judge_reasoning=vote_result.get("reasoning", ""),
        )

    # 3. confidence < HIGH → 否(LLM 自评低把握,判否)
    if confidence is not None and confidence < HIGH_THRESHOLD:
        risks.append(
            RiskWarning(
                code="LOW_CONFIDENCE",
                message=f"LLM 自评 confidence={confidence:.2f},低于高阈值 {HIGH_THRESHOLD},判否",
            )
        )

    # 升级路径(2026-10):如果判定="否" + 抽取标了 information_insufficient=True
    #   → 升级为"无法认定 - 缺指标/缺材料"
    # 抽取 stage 只标"信息不充分"信号,判定 stage 走完正常路径后,
    # 若结论是"否"且信号为真,升级为"无法认定"。
    # 不依赖 confidence 数值,符合"无法认定不参与阈值评判"原则。
    # 修复: 同时检查 information_insufficient 字段 + extraction_notes 关键词
    # (LLM 可能不填新字段,但 notes 字符串里有"信息不充分"信号)
    info_insufficient = getattr(extraction, "information_insufficient", None)
    extraction_notes = getattr(extraction, "extraction_notes", None) or ""
    notes_insufficient_keywords = [
        "缺", "未提供", "未见", "未明确", "无明确", "未发现",
        "信息不充分", "不完整", "未填", "留空", "拟诊", "待诊",
        "考虑", "可能", "待定", "待查", "未见", "无独立", "未单独",
    ]
    notes_signal = any(kw in extraction_notes for kw in notes_insufficient_keywords)
    if info_insufficient is True or notes_signal:
        reasons = getattr(extraction, "information_insufficient_reasons", None) or []
        # 提取 notes 的关键句作 reason
        if not reasons and extraction_notes:
            reasons = [extraction_notes[:150]]
        reason_text = "; ".join(reasons) if reasons else "信息不充分"
        # 区分:有 R 规则 matched 时算"缺指标",无 R 规则算"缺材料"
        code = "INSUFFICIENT_INFO" if rule_id else "MISSING_MATERIAL"
        risks.append(
            RiskWarning(
                code=code,
                message=f"信息不充分({reason_text}),判否升级为无法认定",
            )
        )
        return JudgeOutput(
            case_id=case_id,
            conclusion="无法认定",
            confidence=confidence,
            matched_rule_id=rule_id,
            matched_rule_name=rule_name,
            diagnosis=extraction.diagnosis.diagnosis,
            icd10=extraction.diagnosis.icd10,
            risk_warnings=risks,
            vote_rounds=len(vote_confs),
            vote_confidences=vote_confs,
            vote_agree=vote_agree,
            hit_hard_rules=[rule_id] if rule_id else [],
            judge_reasoning=vote_result.get("reasoning", ""),
        )

    return JudgeOutput(
        case_id=case_id,
        conclusion="否",
        confidence=confidence,
        matched_rule_id=rule_id,
        matched_rule_name=rule_name,
        diagnosis=extraction.diagnosis.diagnosis,
        icd10=extraction.diagnosis.icd10,
        risk_warnings=risks,
        vote_rounds=len(vote_confs),
        vote_confidences=vote_confs,
        vote_agree=vote_agree,
        hit_hard_rules=[rule_id] if rule_id else [],
        judge_reasoning=vote_result.get("reasoning", ""),
    )


def check_required_indicators(
    extraction,
    matched_rule: dict | None,
) -> dict:
    """检查候选 R 规则是否要求了 LLM 没抽到的关键指标。

    Returns:
        {"missing": [{"indicator": "cTn", "rule_id": "R002", "reason": "..."}], "ok": bool}
    """
    if not matched_rule:
        return {"missing": [], "ok": True}

    # 启发:从 criteria 文本里搜"cTn / CK-MB / LVEF / FEV1 / GCS / IQ"等关键词
    # 每条可以是:
    #   ("short", "full") → 短关键词 + 全名,任一在抽取/规则里命中即可
    #   ("short", "full", ["alt1", "alt2"]) → 含同义词列表(任意一个命中即可)
    INDICATOR_KEYWORDS = [
        ("cTn", "肌钙蛋白"),
        ("CK-MB", "肌酸激酶同工酶"),
        ("LVEF", "射血分数"),
        ("FEV1", "第一秒用力呼气容积"),
        ("GCS", "格拉斯哥"),
        # 修复 Bug 6: "PaO2/FiO2" 在规则里是一个组合指标,LLM 经常拆开抽;
        # 只要 PaO2 和 FiO2 都有就算 OK
        ("PaO2", "动脉血氧分压", ["PaO2", "氧分压"]),
        ("FiO2", "吸入氧浓度", ["FiO2", "氧浓度", "吸氧浓度"]),
        ("IQ", "智商"),
        ("ANC", "中性粒细胞"),
        ("网织红", "网织红"),
        ("Hb", "血红蛋白"),
        ("WBC", "白细胞"),
        ("PLT", "血小板"),
        ("ACTH", "促肾上腺皮质激素"),
    ]
    crit_text = " ".join(c.get("text", "") for c in matched_rule.get("criteria", []))
    extracted_names = {li.name.lower() for li in extraction.lab_indicators}
    # 同时建一个"raw"集合用于子串包含匹配(去括号/标点后)
    extracted_norm = {n.replace(" ", "").replace("(", "").replace(")", "") for n in extracted_names}

    def _extracted_has(*needles: str) -> bool:
        """检查抽取的指标里是否包含任一 needle(短关键词或全名)。"""
        for needle in needles:
            nl = needle.lower()
            if any(nl in n for n in extracted_names) or any(nl in n for n in extracted_norm):
                return True
        return False

    missing = []
    for entry in INDICATOR_KEYWORDS:
        kw_short, kw_full = entry[0], entry[1]
        synonyms = entry[2] if len(entry) > 2 else None

        # 1) 规则 criteria 里是否提到这个指标
        rule_has = (kw_short in crit_text) or (kw_full in crit_text)
        if synonyms:
            rule_has = rule_has or any(s in crit_text for s in synonyms)
        if not rule_has:
            continue

        # 2) 抽取的指标里是否满足
        if synonyms:
            # 复合指标:每个 synonym 子项都要单独命中(如 PaO2/FiO2 → PaO2 和 FiO2 都要)
            # 但 INDICATOR_KEYWORDS 现在拆成两条 PaO2 / FiO2 独立条目,所以走不到这里
            ok = _extracted_has(kw_short, kw_full, *synonyms)
        else:
            ok = _extracted_has(kw_short, kw_full)

        if not ok:
            missing.append({
                "indicator": kw_full,
                "rule_id": matched_rule.get("rule_id"),
                "reason": f"规则 {matched_rule.get('rule_id')} 要求指标 '{kw_full}',但抽取阶段未识别",
            })

    # 特殊: PaO2/FiO2(氧合指数)作为整体——只在 crit_text 出现"PaO2/FiO2"或"氧合指数"时
    # 才视为要求,且需要 PaO2 和 FiO2 都有
    if ("PaO2/FiO2" in crit_text) or ("氧合指数" in crit_text):
        pao2_ok = _extracted_has("PaO2", "动脉血氧分压", "氧分压")
        fio2_ok = _extracted_has("FiO2", "吸入氧浓度", "氧浓度", "吸氧浓度")
        if not (pao2_ok and fio2_ok):
            missing.append({
                "indicator": "氧合指数(PaO2/FiO2)",
                "rule_id": matched_rule.get("rule_id"),
                "reason": (
                    f"规则 {matched_rule.get('rule_id')} 要求氧合指数 PaO2/FiO2,"
                    f"但抽取阶段 PaO2/FiO2 至少缺一项"
                ),
            })
        # 去重:PaO2/FiO2 整体缺失已包含"PaO2 缺"和"FiO2 缺",删掉这两条单项 missing
        missing = [m for m in missing if m["indicator"] not in ("动脉血氧分压", "吸入氧浓度")]

    return {"missing": missing, "ok": len(missing) == 0}
