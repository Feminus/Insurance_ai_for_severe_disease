"""判定器阶段的 LLM 评估器。

提供:
1. self_assess_with_voting:对候选 R 规则,LLM 投票 3 次,取中位数
2. fallback_match:疾病不在 rules → LLM 二次比对
"""
from __future__ import annotations

import concurrent.futures
import logging
import statistics
from typing import Any

from src.extraction.schemas import CaseExtraction
from src.judge.llm.prompt_builder import (
    FALLBACK_MATCH_SYSTEM,
    SELF_ASSESS_SYSTEM,
    build_fallback_match_user,
    build_self_assess_user,
    summarize_extraction,
)
from src.llm.client import chat_json, load_config

log = logging.getLogger(__name__)

DEFAULT_VOTE_ROUNDS = 3
# 方差低于此阈值 → 取平均;高于此阈值 → 退化为中位数(抗极端值)
VOTE_VAR_THRESHOLD = 0.01  # 0.01 对应标准差 ~0.1,经验值
# 3 轮 spread ≤ 0.05 视为一致(用于 VOTE_DIVERGENCE 风险码)
VOTE_AGREE_THRESHOLD = 0.05


def _do_single_vote(messages_base: list[dict], cfg, round_idx: int) -> dict:
    """单轮投票:调一次 LLM,返回 {conf, criterion_pass, exclusion_hit, reasoning}。

    失败时返回 conf=NaN + 错误信息;不抛错(让 caller 决定)。
    """
    try:
        raw = chat_json(messages_base, config=cfg)
        conf = _parse_float(raw.get("confidence", 0.5))
        cp = raw.get("criterion_pass", {}) or {}
        eh = raw.get("exclusion_hit", []) or []
        return {
            "conf": conf,
            "criterion_pass": {str(k): bool(v) for k, v in cp.items()} if isinstance(cp, dict) else {},
            "exclusion_hit": [int(x) for x in eh if str(x).isdigit()] if isinstance(eh, list) else [],
            "reasoning": str(raw.get("reasoning", "")),
        }
    except Exception as e:
        log.warning(f"vote {round_idx+1} 失败: {type(e).__name__}: {e}")
        return {
            "conf": float("nan"),
            "criterion_pass": {},
            "exclusion_hit": [],
            "reasoning": f"(本轮失败: {e})",
        }


def _parse_float(v: Any, default: float = 0.5) -> float:
    """鲁棒解析 LLM 返回的 confidence 字段。"""
    try:
        f = float(v)
        if 0.0 <= f <= 1.0:
            return f
    except (TypeError, ValueError):
        pass
    return default


def self_assess_with_voting(
    extraction: CaseExtraction,
    candidate_rule: dict,
    *,
    rounds: int = DEFAULT_VOTE_ROUNDS,
    config=None,
) -> dict:
    """对候选 R 规则做 LLM 投票自评,返回中位数 confidence + 详情。

    性能优化: 3 轮投票并发执行(同一 case 内 IO 密集),总耗时从 ~75s 降到 ~25s。
    并发安全: 每轮调 chat_json 各自独立(messages / LLM client 是 thread-safe 的
    requests.post);只共享 messages_base (只读)。

    Args:
        extraction: CaseExtraction 实例
        candidate_rule: {rule_id, name, criteria, exclusions}
        rounds: 投票轮数(默认 3)
        config: LLMConfig

    Returns:
        dict {
            "confidence": <中位数>,
            "vote_confidences": [c1, c2, c3],
            "vote_agree": bool,
            "criterion_pass": {idx: bool},
            "exclusion_hit": [int],
            "reasoning": str,
        }
    """
    cfg = config or load_config()
    summary = summarize_extraction(extraction)
    messages_base = [
        {"role": "system", "content": SELF_ASSESS_SYSTEM},
        {
            "role": "user",
            "content": build_self_assess_user(
                extraction.case_id, summary, candidate_rule
            ),
        },
    ]

    # 性能优化: 3 轮投票并发执行
    # 使用 ThreadPoolExecutor:LLM 调用是 IO 密集,线程足够;不阻塞主线程
    # 第 1 轮失败仍要抛错(Bug 4);并发场景下 round_idx 0 必须成功
    vote_results: list[dict] = [None] * rounds  # type: ignore
    with concurrent.futures.ThreadPoolExecutor(max_workers=rounds) as pool:
        future_to_idx = {
            pool.submit(_do_single_vote, messages_base, cfg, r): r
            for r in range(rounds)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                vote_results[idx] = future.result()
            except Exception as e:
                # 兜底(其实 _do_single_vote 内部已 catch,这里防意外)
                vote_results[idx] = {
                    "conf": float("nan"),
                    "criterion_pass": {},
                    "exclusion_hit": [],
                    "reasoning": f"(本轮异常: {e})",
                }

    # Bug 4: 第 1 轮失败 → 投票不能继续(硬规则 criteria/exclusion 没法从失败轮拿)
    # 兼容并发场景:vote_results[0] 是 NaN 也算失败
    if vote_results[0]["conf"] != vote_results[0]["conf"]:  # NaN check
        raise RuntimeError(
            f"投票第 1 轮 LLM 调用失败,无法继续: {vote_results[0]['reasoning']}"
        )

    # 提取第 1 轮的 criterion_pass / exclusion_hit(锁定)
    criterion_pass: dict[str, bool] = vote_results[0]["criterion_pass"]
    exclusion_hit: list[int] = vote_results[0]["exclusion_hit"]
    confidences: list[float] = [v["conf"] for v in vote_results]
    reasonings: list[str] = [v["reasoning"] for v in vote_results]

    if not confidences:
        confidences = [0.5]

    # 过滤掉 NaN(失败的轮次),只剩有效值
    valid_confs = [c for c in confidences if not (isinstance(c, float) and c != c)]  # NaN != NaN
    if not valid_confs:
        # 全失败 → 兜底 0.5(配合 Bug 4:第 1 轮失败已抛错,这里只防第 2/3 全失败)
        valid_confs = [0.5]
        confidences = valid_confs

    # 聚合策略:方差 < 阈值 → 取平均(更稳);否则 → 取中位数(抗极端值)
    if len(valid_confs) >= 2:
        spread = max(valid_confs) - min(valid_confs)
        variance = statistics.variance(valid_confs)
        if variance < VOTE_VAR_THRESHOLD:
            final_conf = statistics.mean(valid_confs)
        else:
            final_conf = statistics.median(valid_confs)
        vote_agree = spread <= VOTE_AGREE_THRESHOLD
    else:
        final_conf = valid_confs[0]
        vote_agree = True

    return {
        "confidence": final_conf,
        "vote_confidences": confidences,
        "vote_agree": vote_agree,
        "criterion_pass": criterion_pass,
        "exclusion_hit": exclusion_hit,
        "reasoning": " | ".join(r for r in reasonings if r)[:500],
    }


def fallback_match(
    extraction: CaseExtraction,
    *,
    config=None,
) -> dict:
    """二次比对:疾病不在白名单 → LLM 判断"是否 100 种重疾之一"。

    Returns:
        dict {"is_in_list": bool, "matched_rule_id": str|None, "match_reasoning": str}
    """
    cfg = config or load_config()
    diagnosis = extraction.diagnosis.diagnosis or ""
    icd10 = extraction.diagnosis.icd10
    histology = extraction.diagnosis.histology_type

    messages = [
        {"role": "system", "content": FALLBACK_MATCH_SYSTEM},
        {
            "role": "user",
            "content": build_fallback_match_user(
                extraction.case_id, diagnosis, icd10, histology
            ),
        },
    ]
    try:
        raw = chat_json(messages, config=cfg)
    except Exception as e:
        # 修复 Bug 1:LLM 调用失败 ≠ "疾病不在白名单",应是"无法认定"
        # 失败时返回 error=True 标记,engine 据此触发"无法认定"
        log.warning(f"fallback_match 失败: {type(e).__name__}: {e}")
        return {"is_in_list": False, "matched_rule_id": None,
                "match_reasoning": f"LLM 调用失败: {e}",
                "error": True}

    # LLM 返回了,但没给出 rule_id 也算"无法解析" → 让 engine 走无法认定
    is_in_list = bool(raw.get("is_in_list", False))
    matched_rid = raw.get("matched_rule_id")
    if is_in_list and not matched_rid:
        # Bug 2: LLM 说"是 100 种之一"但没给 rule_id → 解析失败
        return {"is_in_list": True, "matched_rule_id": None,
                "match_reasoning": str(raw.get("match_reasoning", ""))
                + " [LLM 未给出具体 rule_id]",
                "unresolved": True}

    return {
        "is_in_list": is_in_list,
        "matched_rule_id": matched_rid,
        "match_reasoning": str(raw.get("match_reasoning", "")),
    }
