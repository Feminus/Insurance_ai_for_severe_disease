"""
WebUI 预置 demo 案例。

设计:
- 自动扫描 SAMPLES_BASE 下的所有 case 目录(每个子目录 = 1 个 case)
- 可选从打标 xlsx 读标签,作为下拉项的 description 补充
- 不再硬编码 4 个 demo,改用动态列表
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, NamedTuple, Dict, Optional


# 样本案例根目录(可通过环境变量覆盖)
SAMPLES_BASE = Path(
    os.getenv(
        "INSURANCE_SAMPLES_BASE",
        "resources/samples",
    )
)


# 打标 xlsx 路径(用于下拉项的 description,通过环境变量覆盖)
XLSX_PATH = Path(
    os.getenv(
        "INSURANCE_XLSX_PATH",
        "data/knowledge_base/labels.xlsx",
    )
)


class DemoCase(NamedTuple):
    """一个预置 demo 案例。"""

    case_id: str
    label: str                # 下拉项显示文本
    description: str          # 案例说明(xlsx 真实标签 + 疾病名)
    image_paths: List[Path]   # 图片路径列表


def _list_images(case_dir: Path) -> List[Path]:
    """列出一个 case 目录下的所有图片。"""
    if not case_dir.exists():
        return []
    images: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp"):
        images.extend(sorted(case_dir.glob(ext)))
    return images


def _load_xlsx_labels() -> Dict[str, Dict[str, str]]:
    """从 xlsx 读打标,返回 {case_id: {label, disease}}。

    失败时返回空 dict(降级到纯目录扫描)。
    """
    if not XLSX_PATH.exists():
        return {}
    try:
        import openpyxl
    except ImportError:
        return {}
    try:
        wb = openpyxl.load_workbook(XLSX_PATH, data_only=True)
        ws = wb.active
        labels: Dict[str, Dict[str, str]] = {}
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[1] and row[2] is not None:
                labels[str(row[1])] = {
                    "label": str(row[2]).strip(),
                    "disease": str(row[3] or "").strip(),
                }
        return labels
    except Exception:
        return {}


def _all_cases() -> List[DemoCase]:
    """扫描 SAMPLES_BASE,返回所有可用 case。

    从 xlsx 读打标补充 description;xlsx 没有时只显示 case_id。
    """
    if not SAMPLES_BASE.exists():
        return []

    xlsx_labels = _load_xlsx_labels()

    cases: List[DemoCase] = []
    for case_dir in sorted(SAMPLES_BASE.iterdir()):
        if not case_dir.is_dir():
            continue
        cid = case_dir.name
        images = _list_images(case_dir)
        if not images:
            continue  # 跳过空目录

        xlsx_info = xlsx_labels.get(cid, {})
        truth_label = xlsx_info.get("label", "未标注")
        disease = xlsx_info.get("disease", "")
        img_count = len(images)

        # 构造 label: "case_id - 疾病名" 或 "case_id"
        if disease:
            label = f"{cid} - {disease[:18]}"
        else:
            label = f"{cid} - (无疾病标注)"

        # 构造 description: 真实标签 + 图片数
        desc = f"打标:{truth_label} | {img_count} 张图"
        if disease:
            desc = f"{disease[:30]} | 标={truth_label} | {img_count} 张图"

        cases.append(DemoCase(
            case_id=cid,
            label=label,
            description=desc,
            image_paths=images,
        ))

    return cases


# 动态生成所有 demo 列表
# 注:在 webui 端 `st.selectbox` 渲染时是下拉项,不会一次性展示
DEMO_CASES: List[DemoCase] = _all_cases()


def get_demo(case_id: str) -> Optional[DemoCase]:
    """按 case_id 查找 demo。"""
    for d in DEMO_CASES:
        if d.case_id == case_id:
            return d
    return None


def case_count() -> int:
    """返回可用 case 数量(供 webui 显示)。"""
    return len(DEMO_CASES)
