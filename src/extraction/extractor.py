"""LLM 抽取器主入口。

流水线:
1. 接收 OCR 文本(已聚合)
2. 构造 prompt(system + user)
3. 调 LLM 拿 JSON(内部对 SSL/Timeout 等网络瞬时错误做指数退避重试)
4. Pydantic 严格校验
5. 校验失败 → 把错误反馈到 prompt,重试(最多 3 次)
6. 返回 CaseExtraction

支持两种输入:
- 纯文本(merged_text)
- 已聚合的 OCRResult 列表(从 src.aggregation.aggregator 来的)
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests
from pydantic import ValidationError

from src.extraction.prompt_builder import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from src.extraction.schemas import CaseExtraction
from src.llm.client import chat_json, load_config

log = logging.getLogger(__name__)

# LLM 调用重试:服务端瞬时错误(SSL/Timeout/Connection)重试
LLM_MAX_RETRIES = 3
LLM_RETRY_BACKOFF = 2.0  # 1s, 2s, 4s

# Pydantic 校验重试:schema 错误反馈给 LLM 修正
SCHEMA_MAX_RETRIES = 3


def _format_validation_error(err: ValidationError) -> str:
    """把 Pydantic 错误压缩成可读文本反馈给 LLM。"""
    lines: list[str] = []
    for e in err.errors()[:8]:  # 限 8 条,避免 prompt 爆炸
        loc = ".".join(str(x) for x in e.get("loc", []))
        msg = e.get("msg", "")
        typ = e.get("type", "")
        lines.append(f"  - {loc}: {msg} (type={typ})")
    return "JSON 字段校验失败:\n" + "\n".join(lines) + "\n请修正后重新输出。"


# 文档类型归一化映射:LLM 经常返的近似名 → 5 个合法值之一
# 减少 Pydantic Literal 校验失败的概率,降低重试次数
DOC_TYPE_NORMALIZE_MAP: dict[str, str] = {
    # 病理
    "病理": "病理报告",
    "病理诊断": "病理报告",
    "病理诊断报告": "病理报告",
    "病理报告单": "病理报告",
    "活检报告": "病理报告",
    "冰冻": "病理报告",
    "冰冻病理": "病理报告",
    # 诊断证明
    "诊断书": "诊断证明",
    "诊断单": "诊断证明",
    "诊断证明书": "诊断证明",
    "出院诊断证明": "诊断证明",
    # 出院小结
    "出院": "出院小结",
    "出院记录": "出院小结",
    "出院证明": "出院小结",
    "出院医嘱": "出院小结",
    # 检查报告
    "检查": "检查报告",
    "检查单": "检查报告",
    "化验": "检查报告",
    "化验单": "检查报告",
    "检验": "检查报告",
    "检验报告": "检查报告",
    "检验报告单": "检查报告",
    "影像": "检查报告",
    "影像报告": "检查报告",
    "影像学报告": "检查报告",
    "心电图": "检查报告",
    "ct": "检查报告",
    "mri": "检查报告",
    "超声": "检查报告",
    # 其他
    "医嘱": "其他",
    "医嘱单": "其他",
    "临时医嘱": "其他",
    "长期医嘱": "其他",
    "知情同意": "其他",
    "输血": "其他",
    "病案": "其他",
}


def _normalize_doc_types(raw_doc_types: list) -> list[str]:
    """把 LLM 返回的 doc_types 归一到 5 个合法值。

    - 去除 None / 空字符串
    - 严格匹配(已合法)保留
    - 近似名通过 DOC_TYPE_NORMALIZE_MAP 归一
    - 仍无法归一的 → "其他"(不抛错,给 LLM 兜底)
    - 去重但保留顺序
    """
    seen: set[str] = set()
    result: list[str] = []
    for item in raw_doc_types or []:
        if item is None:
            continue
        s = str(item).strip()
        if not s:
            continue
        # 已经在合法值里
        if s in ("病理报告", "诊断证明", "出院小结", "检查报告", "其他"):
            norm = s
        # 近似名归一
        elif s in DOC_TYPE_NORMALIZE_MAP:
            norm = DOC_TYPE_NORMALIZE_MAP[s]
        # 不识别 → "其他"兜底
        else:
            norm = "其他"
        if norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result


def _normalize_extraction(raw: dict) -> dict:
    """对 LLM 返回的整个 dict 做归一化处理。

    处理字段:
    - 顶层 doc_types (list of doc type strings)
    - 嵌套 list 字段里每个 dict 的 source_doc: lab_indicators / imaging_findings /
      historical_records / explicit_negations / source_quotes / surgery.source_doc

    任何非法的 source_doc 都被归一到 5 个合法值之一,减少 Pydantic Literal 校验失败。
    """
    if not isinstance(raw, dict):
        return raw

    normalized = dict(raw)

    # 顶层 doc_types
    if "doc_types" in normalized:
        normalized["doc_types"] = _normalize_doc_types(normalized["doc_types"])

    # 嵌套 list 字段里每个 dict 的 source_doc
    LIST_FIELDS_WITH_SOURCE_DOC = [
        "lab_indicators",
        "imaging_findings",
        "historical_records",
        "explicit_negations",
        "source_quotes",
    ]
    for field_name in LIST_FIELDS_WITH_SOURCE_DOC:
        if field_name not in normalized or not isinstance(normalized[field_name], list):
            continue
        for item in normalized[field_name]:
            if isinstance(item, dict) and "source_doc" in item:
                normalized_list = _normalize_doc_types([item["source_doc"]])
                if normalized_list:
                    item["source_doc"] = normalized_list[0]
                else:
                    item["source_doc"] = "其他"

    # 嵌套 dict 字段: surgery.source_doc
    if isinstance(normalized.get("surgery"), dict) and "source_doc" in normalized["surgery"]:
        normalized_list = _normalize_doc_types([normalized["surgery"]["source_doc"]])
        if normalized_list:
            normalized["surgery"]["source_doc"] = normalized_list[0]
        else:
            normalized["surgery"]["source_doc"] = "其他"

    return normalized


def _call_llm_with_retry(messages: list[dict], cfg, case_id: str) -> dict:
    """调 LLM,对服务端瞬时错误做指数退避重试。

    返回解析后的 dict。空 {} / 非 dict 也算失败 → 抛 ValueError 让上层 schema 重试。
    """
    last_err: Exception | None = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            raw = chat_json(messages, config=cfg, max_retries=1)  # 单次由本函数控重试
            if not isinstance(raw, dict) or not raw:
                raise ValueError(f"LLM 返回为空或非 dict: {str(raw)[:200]}")
            return raw
        except (
            requests.exceptions.SSLError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ) as e:
            last_err = e
            log.warning(
                f"抽取 LLM 网络瞬时错误 attempt={attempt}/{LLM_MAX_RETRIES}: "
                f"{type(e).__name__}: {e}"
            )
            if attempt < LLM_MAX_RETRIES:
                time.sleep(LLM_RETRY_BACKOFF * (2 ** (attempt - 2)))
                continue
            # 重试耗尽
            raise
        except (ValueError, json.JSONDecodeError) as e:
            # JSON 解析失败 → 立即让 schema 重试
            raise
    raise last_err  # 不会到这里


def extract(
    case_id: str,
    merged_text: str,
    *,
    max_retries: int = SCHEMA_MAX_RETRIES,
    config=None,
) -> CaseExtraction:
    """抽取单条案例。

    Args:
        case_id: 案例 ID
        merged_text: 聚合后的 OCR 文本
        max_retries: schema 校验失败时最大重试次数
        config: LLMConfig(默认从环境变量加载)

    Returns:
        CaseExtraction(通过 Pydantic 校验的实例)
    """
    cfg = config or load_config()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(case_id, merged_text)},
    ]
    prev_error: str | None = None

    for attempt in range(1, max_retries + 1):
        try:
            if prev_error:
                # 替换最后一条 user,把错误追加进去
                messages[-1] = {
                    "role": "user",
                    "content": build_user_prompt(case_id, merged_text, prev_error),
                }
            raw = _call_llm_with_retry(messages, cfg, case_id)
            # 归一化:LLM 经常返"诊断书"/"化验单"等近似值,先归一到 5 个合法值
            # 再做 Pydantic Literal 校验,减少重试次数
            normalized_raw = _normalize_extraction(raw)
            result = CaseExtraction.model_validate(normalized_raw)
            log.info(f"抽取成功: case={case_id} attempt={attempt}")
            return result
        except ValidationError as ve:
            prev_error = _format_validation_error(ve)
            log.warning(f"校验失败 attempt={attempt}: {prev_error[:200]}")
        except (ValueError, json.JSONDecodeError) as e:
            prev_error = f"LLM 返回无法解析: {str(e)[:300]}\n请确保输出严格符合 JSON schema,不要包含 markdown。"
            log.warning(f"JSON 解析失败 attempt={attempt}: {e}")
        except (
            requests.exceptions.SSLError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ) as e:
            # LLM 网络重试也耗尽 → 走 schema 重试(可能是临时性)
            prev_error = f"LLM 调用网络错误: {type(e).__name__}: {e}\n请重新输出。"
            log.warning(f"LLM 网络错误 attempt={attempt}: {e}")
        except Exception as e:
            prev_error = f"调用失败: {e}\n请确保输出严格符合 JSON schema。"
            log.warning(f"抽取异常 attempt={attempt}: {e}")

    raise RuntimeError(
        f"案例 {case_id} 经 {max_retries} 次重试仍无法抽取成功。请人工检查 OCR 文本质量。"
    )
