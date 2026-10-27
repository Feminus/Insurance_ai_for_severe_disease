"""规则匹配器(Layer 2)。

把抽取的疾病名 / ICD 映射到 judgment_rules.json 的候选 R 规则。
算法 + LLM 二次比对(疾病词条不在白名单时)。

匹配策略:
1. 用 diseases.json 的 ICD ranges 做主匹配(优先级高)
2. 用疾病名/亚型做模糊匹配(次优)
3. 都没匹配 → 返回空,移交 LLM 二次比对
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.extraction.schemas import CaseExtraction

log = logging.getLogger(__name__)

KB_DIR = Path(__file__).resolve().parents[3] / "data" / "knowledge_base"


# 性能优化:全进程只读一次 JSON,避免每个 case 重复读盘+解析
@lru_cache(maxsize=1)
def _load_diseases() -> list[dict]:
    p = KB_DIR / "diseases.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("diseases", [])


@lru_cache(maxsize=1)
def _load_rules() -> dict[str, dict]:
    """读 judgment_rules.json,按 rule_id 索引。"""
    p = KB_DIR / "judgment_rules.json"
    if not p.exists():
        return {}
    rules_list = json.loads(p.read_text(encoding="utf-8")).get("rules", [])
    return {r["id"]: r for r in rules_list}


# 调试/单测用:清缓存让 reload
def _clear_cache() -> None:
    _load_diseases.cache_clear()
    _load_rules.cache_clear()


def _expand_icd_range(prefix: str) -> set[str]:
    """把 ICD 前缀如 'C00-C97' / 'I20-I25' / 'C50.9' 展开为具体码的近似集合。

    国临版字典里码格式如 'C50.900' / 'C50.900x001',按主类(字母+2位)粗匹配。
    """
    out: set[str] = set()
    if "-" in prefix:
        # 范围:左2位数字
        a, b = prefix.split("-", 1)
        if len(a) >= 3 and len(b) >= 3 and a[0] == b[0]:
            try:
                start = int(a[1:3])
                end = int(b[1:3])
                for i in range(start, end + 1):
                    out.add(f"{a[0]}{i:02d}")
            except ValueError:
                pass
    else:
        # 单码:取字母+2位
        if len(prefix) >= 3:
            out.add(prefix[:3].upper())
    return out


def _icd_prefixes(ranges: list[str]) -> set[str]:
    """把规则的 icd_ranges 展开为 ICD 主类集合。"""
    out: set[str] = set()
    for r in ranges:
        out.update(_expand_icd_range(r))
    return out


def _icd_main_class(icd10: str) -> str | None:
    """从完整 ICD-10 码提取主类(字母+2位)。"""
    if not icd10 or len(icd10) < 3:
        return None
    return icd10[:3].upper()


def _normalize(s: str) -> str:
    """轻量归一:去括号/标点/空格。

    把 "非 ST 段抬高型心肌梗死(NSTEMI)" / "左 肺 上 叶 腺 癌" 这类带
    空格/标点的子病名归一为连续字符串,避免 substring 匹配失败。
    """
    import re
    s = re.sub(r"[（(\[【].*?[）)\]】]", " ", s)
    s = re.sub(r"[,，。；;、]", " ", s)
    # 去所有空白(空格/tab/换行)→ 连续字符串
    s = re.sub(r"\s+", "", s)
    return s.strip().lower()


def match_candidate_rules(extraction: CaseExtraction) -> list[dict]:
    """用算法找候选 R 规则。

    优先级:
    1. ICD 范围匹配(若 extraction.diagnosis.icd10 给出)
    2. 疾病名/亚型名匹配(sub_diseases 列表里)

    Returns:
        候选 R 规则列表(只含 {rule_id, name, criteria, exclusions, _match_source}),
        按匹配强度排序
    """
    diseases = _load_diseases()
    rules = _load_rules()
    if not diseases or not rules:
        return []

    candidates: dict[str, dict] = {}  # rule_id -> entry,带 _score

    diagnosis_name = (extraction.diagnosis.diagnosis or "").strip()
    diagnosis_norm = _normalize(diagnosis_name)
    histology = (extraction.diagnosis.histology_type or "").strip()
    histology_norm = _normalize(histology)
    icd10 = (extraction.diagnosis.icd10 or "").strip()
    icd_class = _icd_main_class(icd10)

    # Pass 1: ICD 主类匹配
    if icd_class:
        for d in diseases:
            ranges = d.get("icd_ranges", [])
            range_classes = _icd_prefixes(ranges)
            if icd_class in range_classes:
                rid = d["rule_id"]
                if rid in rules:
                    r = rules[rid]
                    if rid not in candidates:
                        candidates[rid] = {
                            "rule_id": rid,
                            "name": r["name"],
                            "criteria": r.get("criteria", []),
                            "exclusions": r.get("exclusions", []),
                            "criteria_type": r.get("criteria_type", "all"),
                            "_score": 0.0,
                            "_source": [],
                        }
                    candidates[rid]["_score"] += 1.0
                    candidates[rid]["_source"].append(f"ICD:{icd10}∈{ranges[0] if ranges else ''}")

    # Pass 2: 疾病名/亚型名匹配
    if diagnosis_norm:
        for d in diseases:
            subs = d.get("sub_diseases", [])
            for sd in subs:
                sd_name = _normalize(sd.get("name", ""))
                if not sd_name:
                    continue
                # 子类名包含 或 被包含
                if sd_name in diagnosis_norm or diagnosis_norm in sd_name:
                    rid = d["rule_id"]
                    if rid in rules:
                        if rid not in candidates:
                            r = rules[rid]
                            candidates[rid] = {
                                "rule_id": rid,
                                "name": r["name"],
                                "criteria": r.get("criteria", []),
                                "exclusions": r.get("exclusions", []),
                                "criteria_type": r.get("criteria_type", "all"),
                                "_score": 0.0,
                                "_source": [],
                            }
                        # 名字越短越精确 → 加分高
                        candidates[rid]["_score"] += 0.5 + max(0, 1 - len(sd_name) / 20)
                        candidates[rid]["_source"].append(f"名:{sd['name']}")
                    break  # 同一规则不重复加分
        # 主病名匹配
        for d in diseases:
            if _normalize(d.get("canonical", "")) == diagnosis_norm:
                rid = d["rule_id"]
                if rid in rules:
                    if rid not in candidates:
                        r = rules[rid]
                        candidates[rid] = {
                            "rule_id": rid,
                            "name": r["name"],
                            "criteria": r.get("criteria", []),
                            "exclusions": r.get("exclusions", []),
                            "criteria_type": r.get("criteria_type", "all"),
                            "_score": 0.0,
                            "_source": [],
                        }
                    candidates[rid]["_score"] += 1.0
                    candidates[rid]["_source"].append(f"主名:{d['canonical']}")

    # Pass 3: 组织学类型模糊匹配
    if histology_norm:
        for d in diseases:
            subs_text = " ".join(_normalize(sd.get("name", "")) for sd in d.get("sub_diseases", []))
            canonical_text = _normalize(d.get("canonical", ""))
            corpus = subs_text + " " + canonical_text
            if histology_norm in corpus:
                rid = d["rule_id"]
                if rid in rules and rid not in candidates:
                    r = rules[rid]
                    candidates[rid] = {
                        "rule_id": rid,
                        "name": r["name"],
                        "criteria": r.get("criteria", []),
                        "exclusions": r.get("exclusions", []),
                        "criteria_type": r.get("criteria_type", "all"),
                        "_score": 0.0,
                        "_source": [],
                    }
                    candidates[rid]["_score"] += 0.3
                    candidates[rid]["_source"].append(f"组织学:{histology}")

    # 按 _score 降序
    ranked = sorted(candidates.values(), key=lambda x: -x["_score"])
    log.info(
        f"match_candidate_rules: diagnosis='{diagnosis_name}' icd={icd10} → "
        f"{len(ranked)} 候选: {[c['rule_id'] for c in ranked[:5]]}"
    )
    return ranked
