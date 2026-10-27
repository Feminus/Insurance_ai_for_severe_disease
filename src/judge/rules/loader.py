"""规则库加载器。

读取 `data/knowledge_base/judgment_rules.json`,严格 JSON 解析。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_RULES_PATH = Path(__file__).resolve().parents[3] / "data" / "knowledge_base" / "judgment_rules.json"


def load_rules(path: str | Path = DEFAULT_RULES_PATH) -> dict[str, Any]:
    """严格加载规则 JSON。"""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
