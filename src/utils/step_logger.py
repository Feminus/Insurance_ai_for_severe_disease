"""步骤日志器：为每个 pipeline stage 记录输入/输出摘要，便于定位问题。

设计:
- 内存中累积 step_log(每步一条 StepRecord)
- 同步打印到 stdout(实时看,flush=True 不缓冲)
- 可选落盘到 JSON 文件(事后审计 / WebUI 渲染)

用法:
    logger = StepLogger(case_id="CASE-001")
    logger.start_step("1_ocr", inputs={"file_count": 4})
    # ... 跑 OCR ...
    logger.end_step(outputs={"valid_count": 3, "avg_confidence": 0.95})
    logger.save(Path("logs/case_001.log.json"))
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# 单个字段最大字符数(超过截断,防止 raw_text / merged_text 撑爆日志)
MAX_FIELD_CHARS = 500
# 列表最多保留项数
MAX_LIST_ITEMS = 20


def _truncate(obj: Any, max_chars: int = MAX_FIELD_CHARS) -> Any:
    """截断长字段(防止 raw_text / merged_text / ocr_results 撑爆日志)。"""
    if obj is None:
        return None
    if isinstance(obj, str):
        if len(obj) > max_chars:
            return f"{obj[:max_chars]}...(截断,共 {len(obj)} 字符)"
        return obj
    if isinstance(obj, (int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _truncate(v, max_chars) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        items = list(obj[:MAX_LIST_ITEMS])
        truncated = [_truncate(x, max_chars) for x in items]
        if len(obj) > MAX_LIST_ITEMS:
            truncated.append(f"...(省略 {len(obj) - MAX_LIST_ITEMS} 项)")
        return truncated
    # Pydantic 模型、dataclass、OCRResult 等 → 尝试 model_dump / dict
    if hasattr(obj, "model_dump"):
        try:
            return _truncate(obj.model_dump(exclude_none=True), max_chars)
        except Exception:
            return str(obj)[:max_chars]
    if hasattr(obj, "__dataclass_fields__"):
        try:
            from dataclasses import asdict
            return _truncate(asdict(obj), max_chars)
        except Exception:
            return str(obj)[:max_chars]
    # 兜底转字符串
    return str(obj)[:max_chars]


@dataclass
class StepRecord:
    """单步执行记录。"""

    name: str
    inputs: dict = field(default_factory=dict)
    outputs: dict | None = None
    error: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at: str | None = None
    duration_ms: float | None = None

    def finish(self, outputs: dict | None = None, error: str | None = None) -> None:
        self.outputs = outputs
        self.error = error
        self.finished_at = datetime.now().isoformat()
        self.duration_ms = (
            datetime.fromisoformat(self.finished_at)
            - datetime.fromisoformat(self.started_at)
        ).total_seconds() * 1000


class StepLogger:
    """累积步骤日志的 logger。"""

    def __init__(self, case_id: str, verbose: bool = True, stream=None,
                 on_step_end=None):
        """
        Args:
            case_id: 案例 ID
            verbose: 是否打印到 stdout
            stream: 打印流(测试用 io.StringIO 捕获)
            on_step_end: 每完成一步(end_step 时)调用的回调,签名 callback(step_record: StepRecord)
                         WebUI 用它来实时刷新 UI 状态
        """
        self.case_id = case_id
        self.verbose = verbose
        self.steps: list[StepRecord] = []
        self._current: StepRecord | None = None
        self._started_at = datetime.now()
        # 允许重定向流(测试时用 io.StringIO 捕获)
        self._stream = stream or sys.stdout
        # WebUI 实时刷新钩子(可选)
        self.on_step_end = on_step_end

    def start_step(self, name: str, inputs: dict | None = None) -> StepRecord:
        """开始一个 step。

        如果上一个 step 没显式 end_step,自动以 WARN 关闭(防止漏 end 导致计时不准)。
        """
        if self._current is not None:
            self.end_step(error="WARN: step not explicitly ended")
        rec = StepRecord(name=name, inputs=inputs or {})
        rec.inputs = _truncate(rec.inputs)
        self._current = rec
        self.steps.append(rec)
        if self.verbose:
            self._print(f"\n┌─ [{self.case_id}] STEP: {name}")
            if rec.inputs:
                self._print("│  INPUT:")
                for k, v in rec.inputs.items():
                    self._print(f"│    {k}: {_fmt_value(v)}")
        return rec

    def end_step(self, outputs: dict | None = None, error: str | None = None) -> None:
        """结束当前 step。

        outputs / error 至少一个;都空表示只是计时结束(没产出)。
        """
        if self._current is None:
            return
        self._current.finish(outputs=outputs, error=error)
        # 截断长字段
        if self._current.outputs is not None:
            self._current.outputs = _truncate(self._current.outputs)
        if self.verbose:
            status = "❌ ERROR" if error else "✅ OK"
            ms = self._current.duration_ms or 0
            self._print(f"└─ {status} ({ms:.0f}ms)")
            if self._current.outputs and not error:
                self._print("   OUTPUT:")
                for k, v in self._current.outputs.items():
                    self._print(f"     {k}: {_fmt_value(v)}")
            if error:
                self._print(f"   ERROR: {error}")
        # 通知回调(WebUI 用此实时刷新 UI 状态)
        if self.on_step_end is not None:
            try:
                self.on_step_end(self._current)
            except Exception:
                # 回调失败不影响主流程
                pass
        self._current = None

    def log_event(self, message: str) -> None:
        """在当前 step 中插入一行事件(不开始/结束 step,只追加到当前 step 的输出里)。"""
        if self._current is not None:
            self._current.outputs = self._current.outputs or {}
            self._current.outputs.setdefault("_events", []).append(message)
        if self.verbose:
            self._print(f"│  {message}")

    def to_dict(self) -> dict:
        """导出为可序列化的 dict(供 WebUI 渲染 / 落盘)。"""
        return {
            "case_id": self.case_id,
            "started_at": self._started_at.isoformat(),
            "finished_at": datetime.now().isoformat(),
            "total_duration_ms": (datetime.now() - self._started_at).total_seconds() * 1000,
            "steps": [
                {
                    "name": s.name,
                    "inputs": s.inputs,
                    "outputs": s.outputs,
                    "started_at": s.started_at,
                    "finished_at": s.finished_at,
                    "duration_ms": s.duration_ms,
                    "error": s.error,
                }
                for s in self.steps
            ],
        }

    def save(self, path: str | Path) -> None:
        """保存完整日志到 JSON 文件。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        if self.verbose:
            self._print(f"\n📝 步骤日志已保存: {p}")

    def summary(self) -> str:
        """生成简短摘要(适合 UI 展示)。"""
        lines = [f"案例 {self.case_id} 步骤摘要:"]
        total = sum(s.duration_ms or 0 for s in self.steps)
        lines.append(f"  总耗时: {total:.0f}ms ({total / 1000:.1f}s)")
        for s in self.steps:
            status = "❌" if s.error else "✅"
            lines.append(f"  {status} {s.name}: {s.duration_ms or 0:.0f}ms")
        return "\n".join(lines)

    def _print(self, msg: str) -> None:
        print(msg, flush=True, file=self._stream)


def _fmt_value(v: Any) -> str:
    """格式化字段值用于打印:None / dict / list 简略显示。"""
    if v is None:
        return "None"
    if isinstance(v, str):
        return v if len(v) <= 200 else v[:200] + "..."
    if isinstance(v, (int, float, bool)):
        return str(v)
    if isinstance(v, dict):
        keys = list(v.keys())[:5]
        s = "{" + ", ".join(f"{k}=..." for k in keys)
        if len(v) > 5:
            s += f", +{len(v) - 5} more"
        s += "}"
        return s
    if isinstance(v, (list, tuple)):
        return f"[{len(v)} items]"
    return str(v)[:200]
