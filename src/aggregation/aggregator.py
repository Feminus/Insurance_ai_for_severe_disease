"""聚合器主实现。

输入:多条 OCRResult(同一案例的若干材料)
输出:MergedCase(保留源信息 + 生成带文件名前缀的 merged_text)

doc_type 自动检测策略(优先级降序):
1. 用户显式传入(通过 doc_types 参数)
2. 文件名关键词匹配(中文关键词)
3. 修复 Bug 5: OCR 文本标题嗅探(应对 hash 文件名场景)
   - 搜索范围 500 字(原 200 字不够,医院抬头常占 100+ 字)
   - 长 hint 优先匹配(避免"出院"被"出院诊断证明"覆盖时丢失具体性)
4. 留空(由 LLM 抽取阶段判断)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

from src.aggregation.schemas import MergedCase, MergedDocument
from src.ocr.schemas import OCRResult

log = logging.getLogger(__name__)

# 文件名 → 文档类型 的关键词映射(关键词可出现在文件名任意位置)
# 扩了"出院证"/"结算单"/"医嘱"等常见场景
DOC_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "病理报告": ("病理", "病理报告", "病理诊断", "pathology", "活检", "冰冻"),
    "诊断证明": ("诊断证明", "诊断书", "诊断单", "certificate", "diagnosis", "出院证"),
    "出院小结": ("出院", "出院小结", "出院记录", "出院证明", "discharge"),
    "检查报告": ("检查", "检查报告", "化验", "检验", "影像", "test", "exam", "report"),
}

# OCR 文本标题嗅探的关键词(用于 hash 文件名场景)
# 排序原则:同一类目下,**长 hint 排前面**(优先匹配最具体的)
# 例如 "出院诊断证明" 比 "出院" 更具体,应优先匹配
DOC_TYPE_TITLE_HINTS: list[tuple[str, str]] = [
    # ---- 病理报告(最具体在前) ----
    ("病理报告", "病理检查报告单"),
    ("病理报告", "病理诊断报告书"),
    ("病理报告", "病理诊断报告"),
    ("病理报告", "术中冰冻病理"),
    ("病理报告", "冰冻病理报告"),
    ("病理报告", "病理报告"),
    # ---- 诊断证明(最具体在前) ----
    ("诊断证明", "出院诊断证明"),
    ("诊断证明", "出院诊断证明书"),
    ("诊断证明", "诊断证明书"),
    ("诊断证明", "诊断证明"),
    ("诊断证明", "诊断书"),
    ("诊断证明", "出院证"),
    # ---- 出院小结(最具体在前) ----
    ("出院小结", "出院小结"),
    ("出院小结", "出院记录"),
    ("出院小结", "出院证明"),
    # ---- 检查报告(按子类型) ----
    ("检查报告", "检验报告单"),
    ("检查报告", "检查报告单"),
    ("检查报告", "影像检查报告"),
    ("检查报告", "化验报告单"),
    ("检查报告", "CT检查报告"),
    ("检查报告", "MRI检查报告"),
    ("检查报告", "超声检查报告"),
    ("检查报告", "心电图报告"),
    ("检查报告", "心电图"),
    ("检查报告", "听性脑干反应"),
    ("检查报告", "纯音测听"),
    ("检查报告", "化验报告"),
    ("检查报告", "检验报告"),
    ("检查报告", "检查报告"),
    ("检查报告", "影像报告"),
    # ---- 其他(医嘱、结算单、同意书等) ----
    ("其他", "临时医嘱单"),
    ("其他", "长期医嘱单"),
    ("其他", "临时医嘱"),
    ("其他", "长期医嘱"),
    ("其他", "医嘱单"),
    ("其他", "医保结算单"),
    ("其他", "费用结算单"),
    ("其他", "费用清单"),
    ("其他", "知情同意书"),
    ("其他", "输血治疗同意书"),
    ("其他", "手术同意书"),
    ("其他", "麻醉同意书"),
]

# 标题嗅探搜索范围(字符数)
TITLE_HINT_SEARCH_RANGE = 500


def detect_doc_type(
    file_path: str,
    ocr_doc_type: str | None = None,
    full_text: str | None = None,
) -> str | None:
    """根据文件名 + OCR 自身标识 + OCR 文本标题 嗅探文档类型。

    修复 Bug 5: 72 case 的文件名都是 hash,文件名匹配永远失败。
    在 hash 文件名场景下,扫 OCR 文本前 N 字里的文档标题关键词。

    匹配策略:长 hint 优先(避免"出院"被"出院诊断证明"抢先匹配丢具体性)。
    同一 hint 多次出现取第一次。

    Args:
        file_path: 文件名/路径
        ocr_doc_type: OCR 时已识别的 doc_type(可空)
        full_text: OCR 识别的完整文本(用于标题嗅探)

    Returns:
        文档类型字符串,无法判断返回 None
    """
    if ocr_doc_type and ocr_doc_type != "其他":
        return ocr_doc_type

    # 1. 文件名匹配
    name = Path(file_path).stem.lower()
    for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in name:
                return doc_type

    # 2. OCR 文本标题嗅探
    # 长 hint 优先(列表里已经按长度倒序,见 DOC_TYPE_TITLE_HINTS 定义)
    if full_text:
        head = full_text[:TITLE_HINT_SEARCH_RANGE]
        for doc_type, hint in DOC_TYPE_TITLE_HINTS:
            if hint in head:
                return doc_type

    return None


def _format_doc_block(doc: MergedDocument) -> str:
    """把单份文件格式化为带文件名前缀的文本块。"""
    meta = [f"引擎={doc.engine}"]
    if doc.doc_type:
        meta.append(f"类型={doc.doc_type}")
    if doc.page_count:
        meta.append(f"页数={doc.page_count}")
    if doc.confidence_avg is not None:
        meta.append(f"置信度={doc.confidence_avg:.2f}")
    header = f"【文件:{doc.source_path} | {' | '.join(meta)}】"
    body = doc.full_text.strip() if doc.full_text else "(空)"
    return f"{header}\n{body}"


def aggregate(
    case_id: str,
    ocr_results: Iterable[OCRResult],
    doc_type_overrides: dict[str, str] | None = None,
) -> MergedCase:
    """把多份 OCR 结果聚合成一条案例。

    Args:
        case_id: 案例 ID
        ocr_results: 同一案例的若干 OCR 结果
        doc_type_overrides: 文件路径 → 文档类型 的显式覆盖(优先级最高)

    Returns:
        MergedCase
    """
    overrides = doc_type_overrides or {}
    documents: list[MergedDocument] = []
    blocks: list[str] = []

    for ocr in ocr_results:
        # 1. 确定 doc_type
        override = overrides.get(ocr.source_path)
        # 修复 Bug 5: 传 full_text 让 detect_doc_type 在 hash 文件名场景下嗅探标题
        doc_type = override or detect_doc_type(ocr.source_path, ocr.doc_type, ocr.full_text)

        # 2. 构造 MergedDocument
        page_count = len(ocr.pages)
        char_count = len(ocr.full_text)
        doc = MergedDocument(
            source_path=ocr.source_path,
            doc_type=doc_type,
            engine=ocr.engine,
            engine_version=ocr.engine_version,
            page_count=page_count,
            confidence_avg=ocr.confidence_avg,
            full_text=ocr.full_text,
            char_count=char_count,
        )
        documents.append(doc)
        blocks.append(_format_doc_block(doc))

    merged_text = "\n\n".join(blocks) if blocks else ""

    # 3. 统计 4 类材料是否齐全
    types = {d.doc_type for d in documents if d.doc_type}
    return MergedCase(
        case_id=case_id,
        documents=documents,
        merged_text=merged_text,
        total_chars=sum(d.char_count for d in documents),
        has_pathology="病理报告" in types,
        has_diagnosis_cert="诊断证明" in types,
        has_discharge_summary="出院小结" in types,
        has_exam_report="检查报告" in types,
    )
