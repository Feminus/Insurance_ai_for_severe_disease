# 判定规则介绍

> 判定器的核心契约是 **"3 层流水线 + LLM 投票"**。本文档讲清:
> 1. 100 条规则的来源与编码
> 2. 判定器分 3 层做什么
> 3. 置信度怎么计算、怎么用
> 4. 投票怎么聚合
> 5. "无法认定" 在哪里、什么情况下出现

---

## 1. 规则来源

### 1.1 数据源

| 文件 | 内容 |
|---|---|
| `data/knowledge_base/judgment_rules.json` | 程序可读的 100 条 R 规则(每条含 criteria / exclusions) |
| `data/knowledge_base/diseases.json` | 100 条 R ↔ ICD-10 范围 + 171 个子类映射 |
| `data/knowledge_base/icd_official.json` | 国临版 ICD-10 字典(35587 条) |

### 1.2 规则的 1:1 派生关系

| 编号 | 范围 | 数量 | 来源 |
|---|---|---|---|
| R001 ~ R028 | 行业标准 28 种 | 28 | 《重大疾病保险的疾病定义使用规范(2020 年修订版)》 |
| R029 ~ R100 | 扩展 72 种 | 72 | 业务自定义扩展 |
| **合计** | | **100** | |

`judgment_rules.json` 的每条规则结构:

```json
{
  "id": "R001",
  "no": 1,
  "name": "恶性肿瘤——重度",
  "source": "行业标准2020",
  "criteria_type": "all",
  "criteria": [
    {"idx": 1, "text": "...", "category": "病理"},
    {"idx": 2, "text": "...", "category": "ICD"},
    ...
  ],
  "exclusions": [
    {"idx": 1, "text": "...", "category": "ICD-O-3"},
    ...
  ]
}
```

---

## 2. 判定器 3 层流水线

### Layer 1: 硬规则 gate(`precheck_materials`)

**职责**:材料是否齐全?诊断名是否抽取到?

```python
def precheck_materials(extraction) -> GateResult:
    doc_types = set(extraction.doc_types or [])
    has_any_diag = bool(doc_types & {"病理报告", "诊断证明", "出院小结"})
    if not has_any_diag:
        return GateResult(passed=False, code="MISSING_MATERIAL", ...)
    if not extraction.diagnosis.diagnosis:
        return GateResult(passed=False, code="MISSING_DIAGNOSIS", ...)
    return GateResult(passed=True, code="OK", ...)
```

**触发"无法认定"**:

| code | 含义 |
|---|---|
| `MISSING_MATERIAL` | 4 类医学材料(病理/诊断/出院/检查)全缺,无任何诊断依据 |
| `MISSING_DIAGNOSIS` | 有材料但 LLM 没识别出疾病名 |

### Layer 2: 规则匹配(`matcher.py` + `fallback_match`)

**职责**:把抽取的疾病映射到 100 条 R 规则,确定候选规则。

**算法匹配 3 轮**:

1. **ICD 主类匹配** — 抽取的 ICD 落在 R 的 `icd_ranges` → 加 1.0 分
2. **疾病名/亚型名匹配** — 抽取名包含 R 的子病名或被包含 → 加 0.5~1.5 分
3. **组织学类型匹配** — 抽取的组织学类型在 R 子病名里出现 → 加 0.3 分

返回 `_score` 最高的候选。

**LLM 二次比对**(`fallback_match`):算法 0 候选时,LLM 判断"抽取的疾病是否属于 100 种重疾之一":
- `is_in_list=true` + 给定 `matched_rule_id` → 用该 rule 重进 Layer 3
- `is_in_list=false` → 直接判"否" + `DISEASE_NOT_IN_RULES`

### Layer 2.5: 关键指标检查(`check_required_indicators`)

**职责**:规则要求的指标 LLM 没抽到 → 触发"无法认定 - 缺指标"。

启发式:在 criteria 文本中找"cTn/CK-MB/LVEF/FEV1/GCS/PaO2/IQ/ANC/网织红/Hb/ACTH"等关键词 → 必须在 `extraction.lab_indicators` 里有对应名称。

⚠️ **这是硬规则**——R002 急性心梗要求 cTn/CK-MB,缺一即"无法认定"。

### Layer 3: 投票 + 后置(`assessor.py` + `postcheck.py`)

**职责**:LLM 投票 3 次,按置信度 + 硬规则产出三分类。

---

## 3. 置信度(CaseConfidence)的语义

### 3.1 它是什么

**CaseConfidence** = LLM 对"本案例是否满足该重疾规则"的把握度,**∈ [0, 1]**。

- **0.95** = LLM 自我评估 95% 把握
- **0.5** = 半信半疑
- **0.1** = 几乎不满足

### 3.2 它**不影响"无法认定"**

> 🧠 **设计原则**:confidence 只在"是 / 否"之间决策;**"无法认定"只由硬规则触发**。

| confidence 区间 | 结论 |
|---|---|
| ≥ 0.80 | **是** |
| < 0.80 | **否**(永远不是"无法认定") |
| `None` / 异常 | **否** + LOW_CONFIDENCE 风险码 |

### 3.3 当前阈值

```python
HIGH_THRESHOLD = 0.80
# LOW_THRESHOLD 已废弃 — confidence 不再产生"无法认定"
```

阈值**可调**,未来用全量案例数据调优。

---

## 4. 投票机制

### 4.1 流程

```python
def self_assess_with_voting(extraction, candidate_rule, rounds=3):
    confidences = []
    for r in range(rounds):
        raw = chat_json([system, user_prompt])
        confidences.append(parse_float(raw["confidence"]))
    # 聚合
    if variance < 0.01:
        final = mean(confidences)   # 稳定 → 平均
    else:
        final = median(confidences)  # 极端值 → 中位数
```

### 4.2 聚合策略

| 方差 | 策略 | 理由 |
|---|---|---|
| < 0.01 | **mean** | 3 轮意见一致,平均更准 |
| ≥ 0.01 | **median** | 有一轮明显异常,中位数抗干扰 |

### 4.3 `criterion_pass` / `exclusion_hit` 怎么用

**只在第 1 轮锁定**——后续 2 轮只用来投票 confidence。

```python
if r == 0:
    criterion_pass = raw["criterion_pass"]  # 锁定
    exclusion_hit = raw["exclusion_hit"]      # 锁定
```

LLM 给的是 `{idx: bool}` 字典,判定器硬规则:
- `criterion_pass` 任一 `false` → **否** + `CRITERIA_NOT_MET`
- `exclusion_hit` 非空 → **否** + `EXCLUSION_HIT`

这两个**优先级最高**,不看 confidence。

---

## 5. "无法认定" 的 4 个触发点

| code | 触发层 | 含义 |
|---|---|---|
| `MISSING_MATERIAL` | Layer 1 | 4 类医学材料全缺 |
| `MISSING_DIAGNOSIS` | Layer 1 | 有材料但 LLM 没识别出疾病名 |
| `MISSING_INDICATOR` | Layer 2.5 | 命中规则要求的关键指标但抽取没拿到 |
| `DISEASE_NOT_IN_RULES` | Layer 2 | 算法 0 候选 + LLM 二次比对也确认不在 100 种中 |

**只有这 4 个 code 产生"无法认定"**。其他 code(LOW_CONFIDENCE / VOTE_DIVERGENCE / CRITERIA_NOT_MET / EXCLUSION_HIT)都是"否" 的风险码或备注,不改结论。

---

## 6. 风险码全表

| code | 结论 | 说明 |
|---|---|---|
| **MISSING_MATERIAL** | 无法认定 | 4 类医学材料全缺 |
| **MISSING_DIAGNOSIS** | 无法认定 | 没识别出疾病名 |
| **MISSING_INDICATOR** | 无法认定 | 规则要求的关键指标 LLM 没抽到 |
| **DISEASE_NOT_IN_RULES** | 否 | 抽取的疾病不在 100 种白名单 |
| **RULE_NOT_RESOLVED** | 否 | LLM 二次比对说是 100 种之一但 rule_id 解析不到 |
| **CRITERIA_NOT_MET** | 否 | 规则 criteria 中有未满足项(硬规则) |
| **EXCLUSION_HIT** | 否 | 规则 exclusion 中有命中项(硬规则) |
| **LOW_CONFIDENCE** | 否 | LLM 投票 confidence < 0.80 |
| **VOTE_DIVERGENCE** | (不改结论) | 3 轮 spread > 0.05,标记一致性低 |

---

## 7. 一个完整的判定流程示例

**输入**(`CaseExtraction`):
```python
{
  "case_id": "CL0000000000000001",
  "doc_types": ["病理报告", "诊断证明", "出院小结"],
  "diagnosis": {"diagnosis": "左肺下叶腺癌", "icd10": null, ...},
  "staging": {"tnm_stage": "T2N1M0"},
  "lab_indicators": [],
  ...
}
```

**Layer 1**:`doc_types` 含 3 类诊断材料 → ✅ PASS

**Layer 2**:
- ICD=null → 第 1 轮跳过
- "左肺下叶腺癌" 匹配 R001 sub_diseases "肺恶性肿瘤" → **R001 score=0.7**
- 候选非空 → 进入 Layer 3

**Layer 2.5**:`R001 恶性肿瘤` criteria 不要求特定指标(cTn/LVEF 等)→ ✅

**Layer 3 投票**:
```
[0.95, 0.90, 0.95]  variance=0.0006 → mean=0.933
criterion_pass = {1: true, 2: true, 3: true}
exclusion_hit = []
```

**结论**: confidence 0.933 ≥ 0.80 → **是** + R001

---

## 8. 调优建议

| 调整 | 触发条件 | 改什么 |
|---|---|---|
| HIGH_THRESHOLD 调高(0.85) | 误判"是"太多 | `src/judge/stages/postcheck.py` |
| 投票改成 5 轮 | 稳定性不够 | `assessor.DEFAULT_VOTE_ROUNDS` |
| 加 key indicator 白名单 | 误判"无法认定-缺指标" | `check_required_indicators` |
| 算法匹配的 ICD 范围细化 | R001 这种大伞误伤边界 | `diseases.json` 加细分 rule |

未来调优直接跑全量案例 + 看 `eval_official_label.py` 的 by_label 分布,定位具体哪个规则过严/过松。
