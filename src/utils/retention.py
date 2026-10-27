"""上传文件保留期清理工具占位。"""
from __future__ import annotations

import os
from pathlib import Path


def cleanup_expired_uploads(upload_dir: str | Path, retention_days: int | None = None) -> int:
    """清理超过 retention_days 天的上传文件。返回删除数。"""
    days = retention_days if retention_days is not None else int(os.getenv("UPLOAD_RETENTION_DAYS", "7"))
    raise NotImplementedError("清理逻辑待填充。")
