"""评析模块:对上游 4 个 stage 的结果做综合解读,生成一段话 + 关键风险点。

输入:executor.judge_one_case() 拼好的 result dict
  {
    "extraction": { ... 抽取字段 ... },
    "judge": { ... 判定结果 ... },
    "step_log": { ... 4 阶段 step logger 数据 ... }
  }
输出:dict {"summary": str, "risk_callouts": list[str]}

特性:
- 失败兜底: LLM 不可用时返回默认文本,不阻塞主流程
- response_format={"type":"json_object"} 强制 LLM 返 JSON
- 单次调用, 不重试(评析模块非关键路径)
"""
from __future__ import annotations

import json
import logging
from typing import Any

from src.llm.client import chat_json, load_config

log = logging.getLogger(__name__)

# 评析 prompt: 角色 + 任务 + 输出格式
EVALUATE_SYSTEM = """你是理赔审核资深审核员,任务是根据"抽取证据 + 判定结果"整合一段简明评析。

# 任务
对给定的【抽取证据】和【判定结果】,整合成一段 80-200 字的评析,覆盖:
1. **关键依据**: 本案例为什么判"是/否/无法认定"?(1-2 句)
2. **风险点**: 本次判定可能存在的不确定性 / 边界争议 (1-2 句)
3. **后续建议**: 人工复核时该重点核验什么(1 句)

# 输出要求
- 简明扼要,不超过 200 字
- 引用具体证据(诊断名、关键指标、风险码等),不要空泛
- 不要重复结论本身("是/否/无法认定"),那已经显示在结果里
- 风格: 客观、像内部审核意见
- 中文输出

# 输出格式 (严格 JSON)
{
  "summary": "<一段话 80-200 字,涵盖依据/风险/建议>",
  "risk_callouts": ["<关键要点 1>", "<关键要点 2>"]
}
"""

# 失败兜底
DEFAULT_EVALUATION: dict[str, Any] = {
    "summary": "评析生成失败,请人工核验本次判定。",
    "risk_callouts": ["LLM 评析调用失败,需要人工介入"],
}


def _format_inputs(result: dict) -> tuple[list[dict], list[str]]:
    """把 result dict 转成 LLM 友好的精简输入。

    Returns:
        (messages, risk_codes): LLM messages + 风险码列表(供兜底用)
    """
    ext = result.get("extraction") or {}
    diag = ext.get("diagnosis") or {}
    judge = result.get("judge") or {}

    # 精简抽取信息
    extraction_brief = {
        "diagnosis": diag.get("diagnosis"),
        "icd10": diag.get("icd10"),
        "is_historical": diag.get("is_historical"),
        "diagnosis_date": diag.get("diagnosis_date"),
        "histology_type": diag.get("histology_type"),
        "doc_types": ext.get("doc_types", []),
        "lab_indicators": [li.get("name") for li in (ext.get("lab_indicators") or [])],
        "imaging_findings_count": len(ext.get("imaging_findings") or []),
    }

    # 精简判定信息
    risk_codes = [w.get("code") for w in (judge.get("risk_warnings") or [])]
    judge_brief = {
        "conclusion": judge.get("conclusion"),
        "confidence": judge.get("confidence"),
        "matched_rule_id": judge.get("matched_rule_id"),
        "matched_rule_name": judge.get("matched_rule_name"),
        "risk_codes": risk_codes,
        "vote_rounds": judge.get("vote_rounds"),
        "vote_confidences": judge.get("vote_confidences"),
    }

    user_prompt = (
        "# 抽取的关键证据\n"
        + json.dumps(extraction_brief, ensure_ascii=False, indent=2)
        + "\n\n# 判定结果\n"
        + json.dumps(judge_brief, ensure_ascii=False, indent=2)
        + "\n\n# 请输出 JSON(严格按 schema):"
    )

    messages = [
        {"role": "system", "content": EVALUATE_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    return messages, risk_codes


def evaluate(result: dict) -> dict[str, Any]:
    """对 result dict 做评析,返回 {summary, risk_callouts}。

    失败兜底: LLM 不可用时返回 DEFAULT_EVALUATION,不抛错。
    """
    messages, risk_codes = _format_inputs(result)
    try:
        cfg = load_config()
        raw = chat_json(messages, config=cfg, max_retries=1)
        # 校验返回结构
        summary = raw.get("summary", "").strip()
        callouts = raw.get("risk_callouts", [])
        if not isinstance(callouts, list):
            callouts = [str(callouts)]
        callouts = [str(c).strip() for c in callouts if c]
        if not summary:
            # LLM 返回了空 summary
            log.warning("evaluate: LLM 返回空 summary,使用兜底")
            return {**DEFAULT_EVALUATION, "risk_codes_used": risk_codes}
        return {
            "summary": summary[:500],  # 限长,防 LLM 啰嗦
            "risk_callouts": callouts[:5],  # 最多 5 个要点
            "risk_codes_used": risk_codes,
        }
    except Exception as e:
        log.warning(f"evaluate 失败(已用兜底): {type(e).__name__}: {e}")
        return {**DEFAULT_EVALUATION, "risk_codes_used": risk_codes, "error": str(e)[:200]}
