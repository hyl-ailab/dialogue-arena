"""通用工具：输出清洗、文件锁、实验目录。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from filelock import FileLock


def _thinking_patterns() -> list[re.Pattern[str]]:
    """构建思考标签清洗正则（避免源码中标签被转义丢失）。"""
    tags = [
        ("redacted_thinking", "redacted_thinking"),
        ("think", "think"),
        ("thinking", "thinking"),
    ]
    flags = re.DOTALL | re.IGNORECASE
    return [
        re.compile(rf"<{open_tag}>.*?</{close_tag}>", flags)
        for open_tag, close_tag in tags
    ]


_THINKING_PATTERNS = _thinking_patterns()


def clean_model_output(text: str) -> str:
    """移除思考标签，保留最终回答正文。"""
    result = text or ""
    for pattern in _THINKING_PATTERNS:
        result = pattern.sub("", result)
    return result.strip()


def create_experiment_dir(base_dir: str | Path) -> Path:
    """创建带时间戳的实验目录。"""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base / f"exp_{stamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """线程/进程安全的 JSONL 追加写入。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with FileLock(str(lock_path)):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    """原子写入 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with FileLock(str(lock_path)):
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_finetune_record(messages: list[dict[str, str]]) -> dict[str, Any]:
    """构建标准 SFT 微调格式。"""
    return {"messages": messages}
