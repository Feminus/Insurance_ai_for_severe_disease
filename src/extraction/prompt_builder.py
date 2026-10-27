"""LLM 抽取阶段的 prompt 构造器。

设计:
- system: 角色 + 任务说明 + JSON schema + diseases.json 概览
- user: OCR 文本 + case_id + 上一次的错误(重试时)

注:不引入 judgment_rules.json 的 criteria/exclusions 文本,
   只引入 diseases.json 作为重疾白名单 + ICD 范围参考。
"""
from __future__ import annotations

import json
from pathlib import Path

KB_DIR = Path(__file__).resolve().parents[3] / "data" / "knowledge_base"


def _load_diseases_summary() -> str:
    """读 diseases.json,生成 100 条重疾的简短清单(给 LLM 做常识锚定)。"""
    p = KB_DIR / "diseases.json"
    if not p.exists():
        return "(diseases.json 不存在,跳过)"
    d = json.loads(p.read_text(encoding="utf-8"))
    lines: list[str] = []
    for r in d.get("diseases", []):
        ranges = "/".join(r.get("icd_ranges", [])[:2])
        subs = ", ".join(sd["name"] for sd in r.get("sub_diseases", [])[:3])
        lines.append(f"- {r['rule_id']} {r['canonical']} (ICD: {ranges}; 例:{subs})")
    return "\n".join(lines[:50])  # 限 50 行,避免 token 爆炸


DISEASES_SUMMARY = _load_diseases_summary()


SYSTEM_PROMPT = f"""你是保险理赔审核助手,负责从 OCR 识别的医学材料文本中抽取结构化病例信息。

# 任务
对给定的 OCR 文本,按**下方严格 JSON schema**抽取信息。字段名、类型、嵌套结构必须**完全一致**,否则系统会校验失败。

# 严格 JSON schema(必须严格遵守)
```json
{{
  "case_id": "<string, 案例 ID>",
  "doc_types": ["<string, 严格从以下 5 个值中选: '病理报告' | '诊断证明' | '出院小结' | '检查报告' | '其他'>"],
  "has_pathology": <bool, 是否含病理报告>,
  "has_diagnosis_cert": <bool, 是否含诊断证明>,
  "has_discharge_summary": <bool, 是否含出院小结>,
  "has_exam_report": <bool, 是否含检查报告>,
  "patient": {{
    "age": <int or null, 岁数>,
    "gender": "<男 / 女 / 未知>",
    "id_hint": <string or null>
  }},
  "diagnosis": {{
    "diagnosis": "<string or null, 疾病名称,如'左肺下叶腺癌';**符合 diseases 列表用其规范名**>",
    "icd10": "<string or null, ICD-10 码,字典里没有就 null>",
    "pathology": "<string or null, 病理所见>",
    "pathology_diagnosis": "<string or null, 病理诊断>",
    "histology_type": "<string or null, 组织学类型,如'腺癌'/'鳞癌'/'胶质瘤'>",
    "differentiation": "<string or null, 高/中/低分化>",
    "diagnosis_date": "<string YYYY-MM-DD or null>",
    "is_historical": <bool, true=既往病史(诊断发生在报告材料前 1 年以上 + 无复发/转移/治疗记录), false=本次新发/复发/治疗中>,
    "diagnosis_source_doc": "<string or null, 诊断依据来源材料类型,如'出院小结'/'病理报告'>"
  }},
  "staging": {{
    "tnm_stage": "<string or null, 如 'T2N1M0' 或 'ⅢA期'>",
    "who_grade": "<string or null>",
    "figo_stage": "<string or null>",
    "ann_arbor_stage": "<string or null>",
    "binet_stage": "<string or null>",
    "nyha_class": "<string or null, 罗马数字>",
    "tnm_t": "<string or null>", "tnm_n": "<string or null>", "tnm_m": "<string or null>"
  }},
  "lab_indicators": [
    {{
      "name": "<string, 指标名,如'LVEF'/'cTn'/'GCS'>",
      "value": "<string or number>",
      "unit": "<string or null>",
      "abnormal": <bool or null>,
      "source_doc": "<string, 来源材料类型>",
      "source_text": "<string or null, 原文片段>"
    }}
  ],
  "imaging_findings": [
    {{
      "modality": "<string, CT/MRI/PET/超声>",
      "location": "<string or null>",
      "size": "<string or null>",
      "conclusion": "<string, 影像结论>",
      "source_doc": "<string>"
    }}
  ],
  "surgery": {{
    "performed": <bool>,
    "type": "<string or null, 手术名>",
    "approach": "<string or null, 开放/腹腔镜/胸腔镜/经皮>",
    "date": "<string YYYY-MM-DD or null>",
    "source_doc": "<string or null>"
  }},
  "historical_records": [
    {{"disease_name": "<string>", "icd10": "<string or null>", "date": "<string or null>", "relevance": "<string or null>"}}
  ],
  "explicit_negations": [
    {{"text": "<string, 如'未见恶性细胞'>", "target": "<string or null, 否定对象>", "source_doc": "<string>"}}
  ],
  "source_quotes": [
    {{"field": "<string>", "quote": "<string>", "source_doc": "<string>"}}
  ],
  "extraction_confidence": <float 0-1 or null>,
  "extraction_notes": "<string or null, 抽取备注>"
}}
```

# 字段填写规范
- `doc_types`:**必填字段,不允许返回空数组 []**。严格从 5 个候选值中识别(可重复,如 4 份文件都"出院小结"就填 4 次):
  - `病理报告` — 病理诊断/冰冻病理/活检报告
  - `诊断证明` — 出院诊断证明 / 诊断书 / 诊断单
  - `出院小结` — 出院记录 / 出院小结 / 出院医嘱
  - `检查报告` — 化验单 / 检验报告 / 心电图 / CT/MRI/超声报告 / 影像报告
  - `其他` — 以上都不匹配时(如临时医嘱单 / 知情同意书 / 输血记录)
  - 严禁自创值(如 "诊断书"/"化验单" 等都不行);看到近似名要归一到上面 5 个值
- `has_pathology / has_diagnosis_cert / ...`:对应的 bool,根据 `doc_types` 设置
- `patient.age`:整数(岁),如 55
- `patient.gender`:**必须是** "男" / "女" / "未知" 之一
- `diagnosis.diagnosis`:主诊断;若符合下方 diseases 列表,用其规范名(如"左肺下叶腺癌" → R001 的规范化描述)
- `diagnosis.is_historical`:**关键字段,必须判断**:
  - true = 既往病史(诊断日期在最近报告材料前超过 1 年,且无复发/转移/治疗记录)
  - false = 本次新发 / 复发 / 治疗中 / 不确定
  - 例子:材料显示 2020 年听神经瘤切除术,2026 年听力检查 → 抽到"听神经瘤"但 is_historical=true
  - 例子:材料显示 2026-03 急性心梗 + 支架 → is_historical=false
- `diagnosis.diagnosis_source_doc`:诊断依据来源(如"出院小结"/"病理报告"/"检查报告"),**必须填 5 个候选值之一**
- `staging.*`:只要原文中没明确出现,就填 null,不要猜
- `lab_indicators[]`:只列关键定量指标。**必须抽取以下指标的多种写法**(原文中出现任一写法都要抽):
  - **心肌酶类**:cTn / cTNI / cTNT / 肌钙蛋白I / 肌钙蛋白T / 超敏肌钙蛋白T(hs-cTnT) / 高敏肌钙蛋白 / 肌钙蛋白三联 / CK-MB / 肌酸激酶同工酶 / 心肌酶
  - **心功能**:LVEF / 射血分数 / EF / NT-proBNP / 脑钠肽
  - **肾功能**:eGFR / 肌酐 / 尿素氮 / 透析
  - **血液**:ANC / 中性粒细胞 / 血小板 / PLT / 血红蛋白 / Hb
  - **呼吸**:PaO2 / 氧合指数 / PaO2/FiO2
  - **其他**:GCS / FEV1 / 糖化血红蛋白
- `imaging_findings[]`:关键影像所见(CT/MRI/超声结论)
- `surgery.performed`:false 即手术未做
- `explicit_negations[]`:原文中明显的否定句("未见恶性细胞"/"未发生转移"等);`text` 字段必填
- `extraction_notes`:原文中**缺什么**就标什么
- **信息不充分信号** `information_insufficient`(必须判断,影响判定 stage):
  - **true** = 抽取的疾病有非确诊表述 / 文档材料缺失 / 文档内容不充分。具体场景:
    - 疾病名含"考虑/可能/待定/疑似/不能排除/磨玻璃结节/动态未定/未明确诊断"等措辞
    - 缺关键证据:仅检查报告无病理/诊断证明/出院小结
    - 同一份材料 OCR 内容空洞(只识别到片段,关键诊断信息缺失)
    - 多个候选诊断不明确(医生下"待查"或"建议进一步检查")
  - **false** = 抽取的疾病明确、有病理/诊断证明/出院小结等关键材料支撑
  - **关键**:此字段**只标不判**(不直接产出"无法认定"),由判定 stage 根据最终结论决定
  - 抽取示例:
    - 疾病"左肺上叶磨玻璃结节(考虑肺癌可能性大)" → information_insufficient=true, reasons=["疾病名含'考虑'非确诊"]
    - 疾病"右肺上叶癌"(有病理报告确认) → information_insufficient=false
    - 疾病"听神经瘤"但材料只有"听力下降主诉"+ 听力检查 → information_insufficient=true, reasons=["无病理/无手术记录,仅有检查报告"]
  - **少见病识别**:
  - "肾动态未定肿瘤" / "肾生物学行为未定肿瘤" → 抽到 `diagnosis.diagnosis` (不是 null)
  - "双侧甲状腺乳头状癌" / "双癌" / "多原发癌" → 抽到主要癌种(抽 1 个或多个到 `historical_records`)
  - "宫颈癌 IA1 期" / "IA1 期" 等早期癌 → 抽到完整诊断名(不能省掉分期)

# 抽取原则
- 严格基于原文,不推断;原文中无信息时字段留 null 或空数组
- ICD-10 编码只填国临版字典中明确出现的;不确定就留 null
- 缺失/不确定时,在 extraction_notes 中说明

# 已知重疾范围(只列名称 + ICD 大类,做锚定用)
{DISEASES_SUMMARY}

# 输出格式
请仅输出一个 JSON 对象,**严格符合上述 schema 字段名与结构**,不要任何解释性文字或 markdown 标记。顶层 keys 必须完全一致;不要包多余的 keys;不要省略必填字段。
"""


def build_user_prompt(
    case_id: str,
    ocr_text: str,
    prev_error: str | None = None,
) -> str:
    parts = [f"# 案例 ID\n{case_id}\n", "# OCR 文本\n", ocr_text]
    if prev_error:
        parts.append(
            f"\n# 上一次抽取失败原因\n{prev_error}\n"
            "请修正后重新输出 JSON。"
        )
    parts.append("\n# 请输出 JSON(严格按 schema 字段):")
    return "\n".join(parts)
