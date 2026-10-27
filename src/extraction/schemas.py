"""LLM 抽取阶段的输出 schema(Pydantic 严格校验)。

设计原则:
- 一个案例一份 CaseExtraction
- 包含疾病信息 + 文件类型 + 患者信息 + 关键临床指标
- 每个字段都可追溯到 source_quote
- 抽取失败时,由 schema 校验报错反馈给 LLM 重试(最多 3 次)
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---- 文档类型(强制 5 个候选值之一,LLM 不可自创) ----
# 之前注释 "抽取时不强求,统一在聚合/判定阶段归一" 实际效果差:
# - LLM 经常返回 [] 空数组
# - LLM 经常返回自创值("诊断书"/"化验单"),聚合层归一会丢信息
# 现在改成 Literal 强校验:LLM 返非候选值 → Pydantic ValidationError → 反馈给 LLM 重试
# 后处理归一化(在 extract() 内):碰到近似值先归一再校验,减少重试次数
DocType = Literal["病理报告", "诊断证明", "出院小结", "检查报告", "其他"]


# ---- 患者基本信息 ----
class PatientInfo(BaseModel):
    age: int | None = Field(None, ge=0, le=150, description="年龄(岁)")
    gender: Literal["男", "女", "未知"] = Field("未知", description="性别")
    id_hint: str | None = Field(None, description="患者标识(可只给 case_id 末 4 位作去标识)")


# ---- 疾病抽取 ----
class DiagnosisResult(BaseModel):
    """主要疾病抽取结果。"""

    diagnosis: str | None = Field(
        None, description="疾病诊断名称,如'左肺下叶腺癌'"
    )
    icd10: str | None = Field(
        None, description="ICD-10 编码,如'C34.300x003';无法确定时留空"
    )
    pathology: str | None = Field(
        None, description="病理所见(显微镜下描述)"
    )
    pathology_diagnosis: str | None = Field(
        None, description="病理诊断(病理科出具)"
    )
    is_historical: bool | None = Field(
        None, description=(
            "本次确诊 vs 既往病史区分。"
            "true = 既往病史(诊断在报告材料前 1 年以上 + 没有复发/转移/治疗记录),"
            "false/null = 本次新发/复发/治疗中。"
            "判定 stage 看到 is_historical=true 会判否 + HISTORICAL_DIAGNOSIS 风险码。"
        )
    )
    diagnosis_source_doc: str | None = Field(
        None, description=(
            "诊断依据来源材料类型(病理报告/诊断证明/出院小结/检查报告/其他),"
            "用于判定 stage 追溯证据出处"
        )
    )
    histology_type: str | None = Field(
        None,
        description="组织学类型,如'腺癌'/'鳞癌'/'小细胞癌'/'胶质瘤'/'腺瘤'等",
    )
    differentiation: str | None = Field(
        None,
        description="分化程度:高分化/中分化/低分化/未分化",
    )
    diagnosis_date: str | None = Field(
        None, description="确诊日期(YYYY-MM-DD);不确定时留空"
    )


# ---- 分期 / 分级(关键边界信号) ----
class StagingInfo(BaseModel):
    """分期 / 分级(支持 R001 排除条款、R021/R024/R033 等关键规则)。"""

    tnm_stage: str | None = Field(
        None, description="TNM 分期,如'T2N1M0'或'ⅢA期'"
    )
    tnm_t: str | None = Field(None, description="T(原发肿瘤)")
    tnm_n: str | None = Field(None, description="N(区域淋巴结)")
    tnm_m: str | None = Field(None, description="M(远处转移)")

    who_grade: str | None = Field(
        None, description="WHO 分级(神经内分泌瘤/胶质瘤需要)"
    )
    figo_stage: str | None = Field(
        None, description="FIGO 分期(妇科肿瘤)"
    )
    ann_arbor_stage: str | None = Field(
        None, description="Ann Arbor 分期(淋巴瘤)"
    )
    binet_stage: str | None = Field(
        None, description="Binet 分期(CLL)"
    )
    child_pugh: str | None = Field(
        None, description="Child-Pugh 分级(肝病)"
    )
    nyha_class: str | None = Field(
        None, description="NYHA 心功能分级(罗马数字 Ⅰ-Ⅳ)"
    )


# ---- 实验室 / 检查指标 ----
class LabIndicator(BaseModel):
    name: str = Field(..., description="指标名,如'LVEF'/'FEV1'/'GCS'/'PaO2'/'cTn'")
    value: float | str = Field(..., description="数值或文字描述")
    unit: str | None = Field(None, description="单位,如'%'/'mmHg'/'×10^9/L'")
    ref_range: str | None = Field(None, description="正常参考值范围")
    abnormal: bool | None = Field(None, description="是否异常")
    source_doc: DocType = Field(..., description="来源文档类型")
    source_text: str | None = Field(
        None, description="原文片段(便于追溯)"
    )


class ImagingFinding(BaseModel):
    modality: str = Field(..., description="成像方式:CT/MRI/PET/X线/超声/内镜")
    location: str | None = Field(None, description="部位,如'左肺下叶'/'肝脏'")
    size: str | None = Field(None, description="尺寸描述,如'3.2cm×2.5cm'")
    conclusion: str = Field(..., description="影像结论/所见关键描述")
    source_doc: DocType = Field(..., description="来源文档类型")


# ---- 手术信息 ----
class SurgeryInfo(BaseModel):
    performed: bool = Field(False, description="是否实施了相关手术")
    type: str | None = Field(None, description="手术名称")
    approach: str | None = Field(
        None, description="手术入路:开胸/开腹/腹腔镜/胸腔镜/经皮/介入"
    )
    date: str | None = Field(None, description="手术日期")
    source_doc: DocType | None = Field(None, description="来源文档")


# ---- 既往信息(佐证,不驱动"非首次") ----
class HistoricalRecord(BaseModel):
    disease_name: str = Field(..., description="既往疾病名")
    icd10: str | None = Field(None, description="ICD-10 码")
    date: str | None = Field(None, description="确诊/发生日期")
    relevance: str | None = Field(
        None, description="与本次诊断的关联(佐证/无关/矛盾)"
    )


# ---- 显性否定(防止 LLM 误判) ----
class ExplicitNegation(BaseModel):
    text: str = Field(..., description="显性否定的原文,如'未见恶性细胞'")
    target: str | None = Field(None, description="否定针对的对象,如'恶性'/'转移'")
    source_doc: DocType = Field(..., description="来源文档")


# ---- 原文引用(可追溯) ----
class SourceQuote(BaseModel):
    field: str = Field(..., description="被引用的字段名")
    quote: str = Field(..., description="OCR 原文片段")
    source_doc: DocType = Field(..., description="来源文档类型")


# ---- 整体抽取结果 ----
class CaseExtraction(BaseModel):
    """LLM 抽取阶段的最终输出(单条案例)。"""

    case_id: str = Field(..., description="案例 ID")

    # 1. 元信息
    doc_types: list[DocType] = Field(
        default_factory=list,
        description="已识别的文档类型列表(支持多类型同时存在)",
    )
    has_pathology: bool = Field(False, description="是否有病理报告")
    has_diagnosis_cert: bool = Field(False, description="是否有诊断证明")
    has_discharge_summary: bool = Field(False, description="是否有出院小结")
    has_exam_report: bool = Field(False, description="是否有检查报告")

    # 2. 患者信息
    patient: PatientInfo = Field(default_factory=PatientInfo)

    # 3. 核心:疾病抽取
    diagnosis: DiagnosisResult = Field(default_factory=DiagnosisResult)

    # 4. 分期 / 分级
    staging: StagingInfo = Field(default_factory=StagingInfo)

    # 5. 关键指标
    lab_indicators: list[LabIndicator] = Field(default_factory=list)
    imaging_findings: list[ImagingFinding] = Field(default_factory=list)

    # 6. 手术
    surgery: SurgeryInfo = Field(default_factory=SurgeryInfo)

    # 7. 既往(佐证)
    historical_records: list[HistoricalRecord] = Field(default_factory=list)

    # 8. 显性否定
    explicit_negations: list[ExplicitNegation] = Field(default_factory=list)

    # 9. 原文引用(可追溯)
    source_quotes: list[SourceQuote] = Field(default_factory=list)

    # 10. 抽取质量自评
    extraction_confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="LLM 自评抽取置信度"
    )
    extraction_notes: str | None = Field(
        None, description="抽取备注:信息不全/字段缺失/不确定等"
    )

    # 11. 原始 OCR 文本(由 executor 注入,非 LLM 抽取)
    # 判定阶段 LLM 需要看到原始文本以发现抽取漏掉的关键证据(如"肌力 1-2 级")
    merged_text: str | None = Field(
        None, description="原始 OCR 聚合文本(executor 注入,判定阶段用于补充抽取漏掉的关键证据)"
    )

    # 12. 信息不充分信号(LLM 抽取时主动识别)
    # 抽取 stage 暴露此信号,但**不直接**判无法认定;由判定 stage 根据最终结论决定
    # 升级路径:结论="否" + information_insufficient=True → "无法认定 - 缺材料/缺指标"
    information_insufficient: bool | None = Field(
        None, description=(
            "信息不充分信号:抽取的疾病存在'考虑/可能/待定/磨玻璃结节/动态未定'等"
            "非确诊表述,或文档材料缺失/质量不足。**抽取 stage 只标不判**,"
            "由判定 stage 在 conclusion='否' 时升级为'无法认定 - 缺指标/缺材料'。"
        )
    )
    information_insufficient_reasons: list[str] | None = Field(
        default=None, description=(
            "信息不充分的具体原因列表(供判定 stage 写风险码用),如:"
            "['疾病名含\"考虑\"非确诊', '缺病理报告', '仅有检查报告无诊断证明']"
        )
    )
