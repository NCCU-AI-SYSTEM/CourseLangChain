"""L3:輸出驗證。

設計成「驗證器登記制」:harness 本身只提供 `register_validator` 與 `validate_output`,
不寫具體驗證邏輯。各 tool 在自己的模組裡 `register_validator(name, fn)`,
fn 簽名 `(output_str) -> (ok: bool, msg: str)`,驗證失敗時 ok=False、msg 為人類可讀說明。

範例(在 tools/recommend_schedule.py 末尾):

    from harness import register_validator
    from .validators import has_time_conflict

    def _check_no_conflict(output: str) -> tuple[bool, str]:
        courses = _extract_courses_from_markdown(output)
        if courses and has_time_conflict(courses):
            return False, "輸出含衝堂課程"
        return True, ""

    register_validator("schedule_no_conflict", _check_no_conflict)

`validate_output(output)` 跑所有已註冊 validator,任何一個失敗就回 (False, "[name] msg")。
"""
from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

Validator = Callable[[str], tuple[bool, str]]
_REGISTRY: dict[str, Validator] = {}


def register_validator(name: str, fn: Validator) -> None:
    """登記一個輸出驗證器。`name` 是除錯用識別,須唯一。"""
    if name in _REGISTRY:
        logger.warning("validator %r already registered, overwriting", name)
    _REGISTRY[name] = fn


def validate_output(output: str) -> tuple[bool, str]:
    """跑所有驗證器。第一個失敗即返回 `(False, "[name] msg")`。"""
    if not isinstance(output, str):
        return False, "[type] output must be str"
    for name, fn in _REGISTRY.items():
        try:
            ok, msg = fn(output)
        except Exception:
            logger.exception("validator %r raised", name)
            return False, f"[{name}] validator crashed"
        if not ok:
            return False, f"[{name}] {msg}"
    return True, ""


def clear_validators() -> None:
    """測試用:清空登記表。"""
    _REGISTRY.clear()
