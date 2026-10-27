"""判定阶段 LLM prompt 构造器。

提供 3 类 prompt:
1. self_assess(主投票):对候选 R 规则给出 confidence 0-1
2. fallback_match(二次比对):疾病词条不在 rules → LLM 兜底判断"是否 100 种重疾之一"
3. criteria_check(单次评估):逐条核对 criteria/exclusion(投票的内部细节)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

KB_DIR = Path(__file__).resolve().parents[3] / "data" / "knowledge_base"


def _load_diseases_summary() -> str:
    """diseases.json 的精简列表(给 LLM 当锚定)。"""
    p = KB_DIR / "diseases.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    lines = []
    for r in d.get("diseases", []):
        ranges = "/".join(r.get("icd_ranges", [])[:2])
        lines.append(f"- {r['rule_id']} {r['canonical']} ({ranges})")
    return "\n".join(lines)


DISEASES_SUMMARY = _load_diseases_summary()


# =============================================================
# 主 prompt:对候选 R 规则自评 confidence
# =============================================================
SELF_ASSESS_SYSTEM = f"""你是保险理赔审核员,任务是对一组医学证据 + 候选重疾规则,评估本案例"是否满足该重疾定义"。

# 任务
对给定的【抽取证据】和【候选重疾规则】,逐条核对规则定义的 criteria(满足条件)与 exclusions(排除条款),然后:
1. 给出 confidence ∈ [0, 1],表示本案例"满足该重疾"的把握
   - 1.0 = 完全满足,所有 criteria 都通过且无 exclusion 命中
   - 0.0 = 明显不满足
   - 0.5 = 部分证据支持,关键信息缺失
2. 列出每条 criterion 是否通过(pass: true/false)
3. 列出每条 exclusion 是否被命中(hit: true/false)
4. 简短 reasoning

# 自评口径(关键)
- 本系统的判定原则是"按业务打标口径贴近评估",即对符合 ICD 范围 + 抽取到疾病 + 关键功能/影像证据齐备的 case,即使某条 criteria 在文字上不完全字面满足,也应给出较高 confidence(≥ 0.70)。
- criteria 文本中如注明"【口径放宽:...】",请严格按放宽后的口径判断(如不强制要求"180 天后""永久性"等条件)。
- 急性期/住院期间的神经功能缺损(如肌力 1-2 级)若满足标准即应判满足;不要因"未观察到 180 天后情况"而直接判否。
- 只有当 exclusion 命中、关键指标严重缺失、或证据明显矛盾时,才给出低 confidence。
- **既往病史口径**:如果抽取结果中 `is_historical=true` 标记了诊断是既往的(>1 年前,且无本次复发/治疗证据),该诊断不参与本次重疾认定 → 应给出低 confidence(< 0.40)。本规则优先级最高,系统会单独判否。

# 已知重疾范围(只列名称+ICD大类,用于锚定;具体规则以传入的为准)
{DISEASES_SUMMARY}

# 输出格式
请输出严格符合下述 schema 的 JSON,不要任何解释性文字或 markdown 标记:
{{
  "confidence": <0-1 浮点>,
  "criterion_pass": {{"1": true, "2": false, "3": true, ...}},
  "exclusion_hit": [1, 3],
  "reasoning": "<中文 50-200 字>"
}}
"""


def build_self_assess_user(
    case_id: str,
    extraction_summary: dict,
    candidate_rule: dict,
) -> str:
    """构造自评 user prompt。

    Args:
        case_id: 案例 ID
        extraction_summary: CaseExtraction 关键字段精简版
        candidate_rule: 候选 R 规则 {rule_id, name, criteria: [...], exclusions: [...]}
    """
    parts = [f"# 案例 ID\n{case_id}\n", "# 抽取的关键证据\n"]
    parts.append(json.dumps(extraction_summary, ensure_ascii=False, indent=2))
    parts.append("\n# 候选重疾规则\n")
    parts.append(f"**{candidate_rule['rule_id']} {candidate_rule['name']}**\n")
    parts.append("判定条件(全部满足才算'是'):\n")
    for c in candidate_rule.get("criteria", []):
        parts.append(f"  [{c['idx']}] ({c['category']}) {c['text']}\n")
    excl = candidate_rule.get("exclusions", [])
    if excl:
        parts.append("\n排除条款(命中任一则判'否'):\n")
        for e in excl:
            parts.append(f"  [{e['idx']}] ({e['category']}) {e['text']}\n")
    parts.append("\n# 请输出 JSON(严格按 schema):")
    return "".join(parts)


# =============================================================
# 兜底 prompt:疾病不在 rules → LLM 二次比对"是否 100 种重疾之一"
# =============================================================
FALLBACK_MATCH_SYSTEM = f"""你是保险理赔审核员,任务是对一份抽取出的疾病信息,判断它是否属于【人身险重疾险种 100 种标准】中的任一重疾。

# 已知重疾范围(100 种)
{DISEASES_SUMMARY}

# 任务
对给定的【抽取的疾病名 / ICD】,判断:
1. 它是否对应该列表中**任一**重疾规则?
2. 如果对,返回最相似的 rule_id
3. 如果不对应(非重疾 / 在白名单外),is_in_list = false

# 输出格式
{{
  "is_in_list": <true|false>,
  "matched_rule_id": "<R001 等;若 is_in_list=false 留 null>",
  "match_reasoning": "<中文 30-100 字,说明匹配或不匹配的理由>"
}}
"""


def build_fallback_match_user(
    case_id: str,
    diagnosis_name: str,
    icd10: str | None,
    histology: str | None = None,
) -> str:
    parts = [
        f"# 案例 ID\n{case_id}\n",
        f"# 抽取的疾病\n诊断名:{diagnosis_name}\n",
        f"ICD-10:{icd10 or '(未填)'}\n",
    ]
    if histology:
        parts.append(f"组织学类型:{histology}\n")
    parts.append("\n# 请判断并输出 JSON:")
    return "".join(parts)


# =============================================================
# 抽取阶段 CaseExtraction 精简(给判定 prompt 用,避免 token 爆炸)
# =============================================================

# 抽取阶段容易漏抽的关键证据关键词 → 用来从原始 OCR 文本里捞关键句
# 漏抽原因:LLM 抽取时只关注典型定量指标,漏掉肌力等级、神经功能评估等关键描述
# 覆盖范围:100 条 R 规则中占训练集 80% 的 R001/R002/R003/R009 等高发规则
KEY_EVIDENCE_KEYWORDS = [
    # ---- 神经功能 / 意识 ----
    "肌力", "肌张力", "NIHSS", "洼田", "GCS", "昏迷", "意识", "言语", "失语",
    "偏瘫", "瘫痪", "癫痫", "抽搐", "痴呆", "认知", "记忆",
    # ---- 心血管 / 心肌酶 ----
    "肌钙蛋白", "cTn", "CK-MB", "LVEF", "射血分数", "EF", "NT-proBNP",
    "心电图", "ST段", "心律失常", "室壁瘤", "冠脉", "支架", "搭桥", "开胸",
    "血压", "心率", "NYHA", "心功能", "心衰",
    # ---- 呼吸 ----
    "FEV1", "PaO2", "PaO2/FiO2", "氧合指数", "SpO2", "氧饱和度", "呼吸衰竭",
    # ---- 肾 / 血液 ----
    "透析", "eGFR", "肌酐", "尿素氮", "肾衰", "尿毒症",
    "ANC", "中性粒", "网织红", "PLT", "血小板", "白细胞", "血红蛋白",
    "再障", "骨髓",
    # ---- 肝 / 消化 ----
    "肝硬化", "腹水", "黄疸", "脾大", "食管静脉", "腹水", "Child",
    # ---- 内分泌 ----
    "糖化", "HbA1c", "空腹血糖", "胰岛素",
    # ---- 肿瘤分期 ----
    "TNM", "WHO", "FIGO", "Ann Arbor", "Binet", "分期", "分级", "Ki-67",
    # ---- 病理 / 组织学 ----
    "病理", "腺癌", "鳞癌", "小细胞", "胶质瘤", "淋巴瘤", "白血病", "骨髓瘤",
    "低分化", "中分化", "高分化", "未分化", "低级别", "高级别", "G1", "G2", "G3", "G4",
    "原位癌", "浸润", "转移", "复发", "免疫组化", "ER", "PR", "HER2",
    # ---- 治疗 / 手术 ----
    "开颅", "切除", "搭桥", "支架", "移植", "透析", "放疗", "化疗", "靶向", "免疫治疗",
    "微创", "腹腔镜", "胸腔镜",
    # ---- 烧伤 / 外伤 ----
    "烧伤", "Ⅲ度", "三度", "面积", "体表", "TBSA", "截肢", "肢体缺失",
    # ---- 五官 ----
    "听力", "听阈", "dB", "HL", "视力", "视野", "光感", "失明", "失聪",
    # ---- 智力 ----
    "IQ", "智商", "智力", "发育",
    # ---- 时间口径 ----
    "180", "天后", "半年", "永久性", "3月", "6月", "12月",
]


def _extract_key_evidence_lines(merged_text: str, max_lines: int = 30) -> list[str]:
    """从原始 OCR 文本里捞含关键证据关键词的行。

    抽取阶段常漏掉的功能性描述(如"右上肢肌力 1-2 级")用这个兜底捞出来。
    限定 max_lines 避免 token 爆炸。
    """
    if not merged_text:
        return []
    seen = set()
    matched = []
    for line in merged_text.splitlines():
        line = line.strip()
        if not line or len(line) < 4:
            continue
        if line in seen:
            continue
        for kw in KEY_EVIDENCE_KEYWORDS:
            if kw in line:
                matched.append(line[:200])  # 单行限长
                seen.add(line)
                break
        if len(matched) >= max_lines:
            break
    return matched


def summarize_extraction(extraction) -> dict[str, Any]:
    """把 CaseExtraction 转成判定 prompt 友好的精简 dict。"""
    summary = {
        "case_id": extraction.case_id,
        "doc_types": extraction.doc_types,
        "has_pathology": extraction.has_pathology,
        "has_diagnosis_cert": extraction.has_diagnosis_cert,
        "has_discharge_summary": extraction.has_discharge_summary,
        "has_exam_report": extraction.has_exam_report,
        "patient": {
            "age": extraction.patient.age,
            "gender": extraction.patient.gender,
        },
        "diagnosis": {
            "name": extraction.diagnosis.diagnosis,
            "icd10": extraction.diagnosis.icd10,
            "pathology": extraction.diagnosis.pathology,
            "pathology_diagnosis": extraction.diagnosis.pathology_diagnosis,
            "histology_type": extraction.diagnosis.histology_type,
            "differentiation": extraction.diagnosis.differentiation,
            "diagnosis_date": extraction.diagnosis.diagnosis_date,
        },
        "staging": {
            "tnm_stage": extraction.staging.tnm_stage,
            "who_grade": extraction.staging.who_grade,
            "figo_stage": extraction.staging.figo_stage,
            "ann_arbor_stage": extraction.staging.ann_arbor_stage,
            "binet_stage": extraction.staging.binet_stage,
            "nyha_class": extraction.staging.nyha_class,
        },
        "lab_indicators": [
            {
                "name": li.name,
                "value": li.value,
                "unit": li.unit,
                "abnormal": li.abnormal,
                "source_doc": li.source_doc,
            }
            for li in extraction.lab_indicators
        ],
        "imaging_findings": [
            {
                "modality": img.modality,
                "location": img.location,
                "size": img.size,
                "conclusion": img.conclusion,
            }
            for img in extraction.imaging_findings
        ],
        "surgery": {
            "performed": extraction.surgery.performed,
            "type": extraction.surgery.type,
            "approach": extraction.surgery.approach,
            "date": extraction.surgery.date,
        },
        "explicit_negations": [
            {"text": n.text, "target": n.target}
            for n in extraction.explicit_negations
        ],
    }
    # 关键证据兜底:从原始 OCR 文本里捞含关键词的行
    # 用于补充 LLM 抽取漏掉的关键描述(如"肌力 1-2 级"、"NIHSS 评分")
    if getattr(extraction, "merged_text", None):
        key_lines = _extract_key_evidence_lines(extraction.merged_text)
        if key_lines:
            summary["key_evidence_from_ocr"] = key_lines
    return summary
