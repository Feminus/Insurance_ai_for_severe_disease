"""端到端执行器。

流水线:OCR (per file) → 聚合 → LLM 抽取 → LLM 判定 (投票 3 次) → JudgeOutput

提供两个主入口:
- judge_one_case(case_id, case_dir) -> dict
  给定一个案例目录,跑完整流水线,返回 JSON 可序列化的结果字典
- run_batch(case_dirs) -> list[dict]
  批跑多个案例

输入/输出目录:
- 输入: case_dir 是包含若干图片/PDF 的目录(文件名即 case_id 也行)
- 输出: 默认写到 data/output/case_<id>_<timestamp>.json
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.aggregation import aggregate
from src.aggregation.schemas import MergedCase
from src.extraction import extract
from src.extraction.schemas import CaseExtraction
from src.judge import judge
from src.judge.schema.output_schema import JudgeOutput
from src.ocr import run_ocr
from src.ocr.schemas import OCRResult
from src.utils.step_logger import StepLogger

log = logging.getLogger(__name__)

# 路径约定(全部相对项目根,不用绝对路径)
# executor.py 位于 src/, 项目根 = parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTS = {".pdf"}
SUPPORTED_EXTS = IMAGE_EXTS | PDF_EXTS


def list_files(case_dir: str | Path) -> list[Path]:
    """列出案例目录下的所有支持文件(图片/PDF)。"""
    p = Path(case_dir)
    if not p.exists():
        raise FileNotFoundError(f"案例目录不存在: {p}")
    if not p.is_dir():
        # 把它当作单个文件
        if p.suffix.lower() in SUPPORTED_EXTS:
            return [p]
        raise ValueError(f"不是目录也不是支持的文件: {p}")

    files: list[Path] = []
    for f in sorted(p.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
            # 跳过 Windows Zone.Identifier 元数据
            if f.name.endswith(":Zone.Identifier"):
                continue
            files.append(f)
    return files


def run_ocr_for_files(files: list[Path]) -> list[OCRResult]:
    """对每个文件跑 OCR,失败的跳过但记日志。"""
    results: list[OCRResult] = []
    for f in files:
        t0 = time.time()
        try:
            r = run_ocr(str(f))
            log.info(f"OCR 成功: {f.name} ({time.time()-t0:.1f}s)")
            results.append(r)
        except Exception as e:
            log.warning(f"OCR 失败: {f.name} | {type(e).__name__}: {e}")
    return results


def _infer_doc_type_from_filename(name: str) -> str | None:
    """从文件名猜文档类型(4 类医学材料)。"""
    n = name.lower()
    if "病理" in n:
        return "病理报告"
    if "诊断" in n:
        return "诊断证明"
    if "出院" in n:
        return "出院小结"
    if "检查" in n or "化验" in n or "检验" in n or "影像" in n:
        return "检查报告"
    return None


def judge_one_case(
    case_id: str,
    case_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    step_logger: StepLogger | None = None,
    save_step_log: bool = True,
) -> dict:
    """跑单条案例全流程,返回 JSON 可序列化的结果字典。

    每个 stage 都有结构化的输入/输出日志(StepLogger):
      - 1_ocr: 输入文件列表 → 输出每张图的字符数/置信度
      - 2_aggregate: 输入 OCR 结果数 → 输出合并后 doc_types / 总字符数
      - 3_llm_extract: 输入 merged_text 长度 → 输出诊断/ICD/lab_indicators 数
      - 4_judge: 输入 extraction 摘要 → 输出 conclusion/confidence/命中规则/风险码

    Args:
        case_id: 案例 ID(用于标注)
        case_dir: 案例目录路径
        output_path: 可选,落盘的 JSON 文件路径
        step_logger: 可选,外部传入的 StepLogger(供 WebUI 实时拿 step 状态)
        save_step_log: 是否落盘 logs/case_<id>.log.json(默认 True)

    Returns:
        {
            "case_id": ...,
            "input_dir": ...,
            "file_count": N,
            "ocr_results": [OCR 结果数],
            "merged": {MergedCase dict},
            "extraction": {CaseExtraction dict or None},
            "judge": {JudgeOutput dict},
            "elapsed_sec": float,
        }
    """
    t_start = time.time()
    case_dir = Path(case_dir)
    log.info(f"[{case_id}] 开始处理: {case_dir}")

    # 初始化 StepLogger(每个 case 一个)
    if step_logger is None:
        step_logger = StepLogger(case_id=case_id, verbose=True)

    # ============================================================
    # Stage 1: OCR
    # ============================================================
    files = list_files(case_dir)
    step_logger.start_step("1_ocr", inputs={
        "case_dir": str(case_dir.absolute()),
        "file_count": len(files),
        "files": [f.name for f in files[:10]] + (["..."] if len(files) > 10 else []),
    })
    if not files:
        step_logger.end_step(error=f"案例目录 {case_dir} 下没有任何支持的文件")
        raise ValueError(f"案例目录 {case_dir} 下没有任何支持的文件")

    try:
        ocr_results = run_ocr_for_files(files)
        if not ocr_results:
            step_logger.end_step(error="全部 OCR 失败")
            raise RuntimeError(f"[{case_id}] 全部 OCR 失败")
        # 统计 OCR 输出
        total_chars = sum(len(r.full_text or "") for r in ocr_results)
        confidences = [r.confidence_avg for r in ocr_results if r.confidence_avg is not None]
        avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else None
        min_conf = round(min(confidences), 4) if confidences else None
        per_file = [
            {
                "file": Path(r.source_path).name,
                "chars": len(r.full_text or ""),
                "confidence": round(r.confidence_avg, 4) if r.confidence_avg is not None else None,
            }
            for r in ocr_results
        ]
        step_logger.end_step(outputs={
            "valid_count": len(ocr_results),
            "total_files": len(files),
            "total_chars": total_chars,
            "avg_confidence": avg_conf,
            "min_confidence": min_conf,
            "per_file": per_file[:5] + (["...(省略)"] if len(per_file) > 5 else []),
            "elapsed_sec": round(time.time() - t_start, 2),
        })
    except Exception as e:
        step_logger.end_step(error=f"{type(e).__name__}: {e}")
        raise

    # ============================================================
    # Stage 2: 聚合
    # ============================================================
    t_stage = time.time()
    step_logger.start_step("2_aggregate", inputs={
        "ocr_count": len(ocr_results),
        "inferred_overrides": sum(
            1 for f in files if _infer_doc_type_from_filename(f.name)
        ),
    })
    try:
        overrides: dict[str, str] = {}
        for f in files:
            guessed = _infer_doc_type_from_filename(f.name)
            if guessed:
                overrides[str(f.absolute())] = guessed
        merged: MergedCase = aggregate(case_id, ocr_results, doc_type_overrides=overrides)

        # 统计 doc_type 分布
        doc_type_counts: dict[str, int] = {}
        for doc in merged.documents:
            dt = doc.doc_type or "未识别"
            doc_type_counts[dt] = doc_type_counts.get(dt, 0) + 1

        step_logger.end_step(outputs={
            "document_count": len(merged.documents),
            "total_chars": merged.total_chars,
            "merged_text_len": len(merged.merged_text),
            "has_pathology": merged.has_pathology,
            "has_diagnosis_cert": merged.has_diagnosis_cert,
            "has_discharge_summary": merged.has_discharge_summary,
            "has_exam_report": merged.has_exam_report,
            "doc_type_counts": doc_type_counts,
            "elapsed_sec": round(time.time() - t_stage, 2),
        })
    except Exception as e:
        step_logger.end_step(error=f"{type(e).__name__}: {e}")
        raise

    # ============================================================
    # Stage 3: LLM 抽取
    # ============================================================
    t_stage = time.time()
    step_logger.start_step("3_llm_extract", inputs={
        "case_id": case_id,
        "merged_text_len": len(merged.merged_text),
        "doc_types": list(doc_type_counts.keys()),
    })
    extraction: CaseExtraction | None = None
    try:
        extraction = extract(case_id, merged.merged_text)
        # 抽取成功:统计输出
        ext_diag = extraction.diagnosis.diagnosis if extraction else None
        ext_icd = extraction.diagnosis.icd10 if extraction else None
        ext_hist = extraction.diagnosis.histology_type if extraction else None
        ext_lab_count = len(extraction.lab_indicators) if extraction else 0
        ext_img_count = len(extraction.imaging_findings) if extraction else 0
        ext_doc_types = extraction.doc_types if extraction else []
        ext_conf = extraction.extraction_confidence if extraction else None
        step_logger.end_step(outputs={
            "diagnosis": ext_diag,
            "icd10": ext_icd,
            "histology_type": ext_hist,
            "extraction_confidence": ext_conf,
            "doc_types": ext_doc_types,
            "lab_indicators_count": ext_lab_count,
            "imaging_findings_count": ext_img_count,
            "elapsed_sec": round(time.time() - t_stage, 2),
        })
    except Exception as e:
        step_logger.end_step(error=f"{type(e).__name__}: {e}")
        log.warning(f"[{case_id}] 抽取失败: {type(e).__name__}: {e}")

    # 抽取兜底
    if extraction is None:
        extraction = CaseExtraction(
            case_id=case_id,
            doc_types=[d.doc_type for d in merged.documents if d.doc_type],
        )

    # 把原始 OCR 文本塞进 extraction,让判定阶段 LLM 能看到抽取漏掉的关键证据
    extraction.merged_text = merged.merged_text

    # ============================================================
    # Stage 4: 判定(3 层流水线)
    # ============================================================
    t_stage = time.time()
    step_logger.start_step("4_judge", inputs={
        "case_id": case_id,
        "extraction_diagnosis": extraction.diagnosis.diagnosis,
        "extraction_doc_types": extraction.doc_types,
        "lab_indicators_count": len(extraction.lab_indicators),
    })
    try:
        judge_output: JudgeOutput = judge(extraction)
        elapsed_judge = round(time.time() - t_stage, 2)

        # 风险码汇总
        risk_codes = [w.code for w in (judge_output.risk_warnings or [])]
        # 投票详情
        vote_confs = judge_output.vote_confidences or []

        step_logger.end_step(outputs={
            "conclusion": judge_output.conclusion,
            "confidence": judge_output.confidence,
            "matched_rule_id": judge_output.matched_rule_id,
            "matched_rule_name": judge_output.matched_rule_name,
            "diagnosis": judge_output.diagnosis,
            "icd10": judge_output.icd10,
            "risk_codes": risk_codes,
            "vote_rounds": judge_output.vote_rounds,
            "vote_confidences": vote_confs,
            "vote_agree": judge_output.vote_agree,
            "hit_hard_rules": judge_output.hit_hard_rules or [],
            "elapsed_sec": elapsed_judge,
        })
    except Exception as e:
        step_logger.end_step(error=f"{type(e).__name__}: {e}")
        raise

    # ============================================================
    # Stage 5: 评析(对前 4 阶段做综合解读)
    # ============================================================
    t_stage = time.time()
    step_logger.start_step("5_evaluate", inputs={
        "case_id": case_id,
        "judge_conclusion": judge_output.conclusion,
        "matched_rule_id": judge_output.matched_rule_id,
        "extraction_diagnosis": extraction.diagnosis.diagnosis,
    })
    evaluation: dict = {}
    try:
        from src.judge.evaluator import evaluate
        # 构造 evaluator 需要的 input(result 形式)
        eval_input = {
            "extraction": _to_dict(extraction),
            "judge": _to_dict(judge_output),
        }
        evaluation = evaluate(eval_input)
        step_logger.end_step(outputs={
            "summary": evaluation.get("summary", ""),
            "risk_callouts": evaluation.get("risk_callouts", []),
            "risk_codes_used": evaluation.get("risk_codes_used", []),
            "error": evaluation.get("error", ""),
            "elapsed_sec": round(time.time() - t_stage, 2),
        })
    except Exception as e:
        # 评析失败不阻塞主流程, 只记日志
        log.warning(f"[{case_id}] 评析 stage 失败(已用兜底): {type(e).__name__}: {e}")
        step_logger.end_step(error=f"{type(e).__name__}: {e}", outputs={
            "summary": "评析生成失败,请人工核验本次判定。",
        })

    elapsed = time.time() - t_start
    log.info(
        f"[{case_id}] 判定: {judge_output.conclusion} "
        f"(conf={judge_output.confidence}, {elapsed:.1f}s)"
    )

    # 5. 组装结果
    result = {
        "case_id": case_id,
        "input_dir": str(case_dir.absolute()),
        "file_count": len(files),
        "files": [f.name for f in files],
        "ocr_count": len(ocr_results),
        "elapsed_sec": round(elapsed, 2),
        "merged": _to_dict(merged),
        "extraction": _to_dict(extraction) if extraction is not None else None,
        "judge": _to_dict(judge_output),
        "evaluation": evaluation,
    }

    # 6. 落盘主结果
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info(f"[{case_id}] 落盘: {p}")

    # 7. 落盘步骤日志(默认开启)
    if save_step_log:
        log_path = DEFAULT_LOG_DIR / f"case_{case_id}.log.json"
        try:
            step_logger.save(log_path)
        except Exception as e:
            log.warning(f"[{case_id}] step log 落盘失败: {e}")

    return result


def run_batch(
    case_specs: list[tuple[str, str | Path]],
    *,
    output_dir: str | Path | None = None,
    verbose_steps: bool = False,
    save_step_log: bool = True,
) -> list[dict]:
    """批跑多个案例。

    Args:
        case_specs: [(case_id, case_dir), ...]
        output_dir: 落盘目录,默认 data/output/
        verbose_steps: 是否打印每步详细日志(默认 False,只打汇总)
        save_step_log: 是否落盘 logs/case_<id>.log.json(默认 True)

    Returns:
        results: 每个案例的结果字典
    """
    out = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    results: list[dict] = []
    for case_id, case_dir in case_specs:
        out_path = out / f"case_{case_id}_{timestamp}.json"
        try:
            # 批跑时复用单个 StepLogger,但 verbose 受 verbose_steps 控制
            step_logger = StepLogger(case_id=case_id, verbose=verbose_steps)
            r = judge_one_case(
                case_id, case_dir,
                output_path=out_path,
                step_logger=step_logger,
                save_step_log=save_step_log,
            )
            results.append(r)
        except Exception as e:
            log.error(f"[{case_id}] 失败: {type(e).__name__}: {e}")
            results.append({
                "case_id": case_id,
                "input_dir": str(case_dir),
                "error": f"{type(e).__name__}: {e}",
            })
    return results


def _to_dict(obj: Any) -> dict:
    """统一把 Pydantic BaseModel 转 dict,避免 list[EvidenceQuote] 等嵌套序列化问题。"""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=False)
    if hasattr(obj, "dict") and callable(obj.dict):  # 兼容 pydantic v1
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj


if __name__ == "__main__":
    # 占位:CLI 入口在 scripts/batch_run.py
    import sys
    print("请通过 scripts/batch_run.py 调用")
    sys.exit(0)
