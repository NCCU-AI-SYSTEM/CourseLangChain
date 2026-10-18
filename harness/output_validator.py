"""L3:輸出驗證。

設計成「登記制」:harness 本身只提供登記與執行,不寫具體驗證邏輯。具體規則由擁有
那份知識的模組自行登記(排課規則在 tools/schedule_tool.py,agent 層的規則在
agents/output_checks.py)。

兩種規則:

- **驗證器**(`register_validator`):`fn(output) -> (ok, msg)`。任何一個失敗,整段回答
  就不給使用者看,改回該驗證器登記的 `user_message`(沒登記就用呼叫端的預設訊息)。
- **改寫器**(`register_rewriter`):`fn(output) -> output`。把可疑的部分拿掉後放行,
  例如移除指向外部網域的連結。所有驗證器都通過之後才執行。

驗證器需要回答以外的資訊(例如使用者這一輪問了什麼)時,呼叫 `current_context()`。
呼叫端用 `check_output(output, context={...})` 傳進來;沒傳就是空 dict。

範例(在 tools/schedule_tool.py 末尾):

    from harness import register_validator

    def _check_no_conflict(output: str) -> tuple[bool, str]:
        ...
        return False, "輸出含衝堂課程"

    register_validator("schedule_no_conflict", _check_no_conflict,
                       user_message="抱歉,系統產生的課表未通過正確性檢查…")

`validate_output(output)` 是舊介面,只跑驗證器、回 `(ok, "[name] msg")`,保留給既有測試用。
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

Validator = Callable[[str], tuple[bool, str]]
Rewriter = Callable[[str], str]


@dataclass(frozen=True)
class _Entry:
    fn: Validator
    user_message: str | None


@dataclass(frozen=True)
class Verdict:
    """`check_output` 的結果。

    ok=True 時 `output` 是(可能被改寫過的)可以給使用者看的回答;
    ok=False 時 `failed` 是失敗的驗證器名稱,`user_message` 是它登記的訊息(可能是 None)。
    """

    ok: bool
    output: str
    failed: str | None = None
    detail: str = ""
    user_message: str | None = None


_VALIDATORS: dict[str, _Entry] = {}
_REWRITERS: dict[str, Rewriter] = {}
_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("l3_context", default={})


def register_validator(name: str, fn: Validator, *, user_message: str | None = None) -> None:
    """登記一個驗證器。`name` 須唯一;`user_message` 是驗證失敗時給使用者看的話。"""
    if name in _VALIDATORS:
        logger.warning("validator %r already registered, overwriting", name)
    _VALIDATORS[name] = _Entry(fn, user_message)


def register_rewriter(name: str, fn: Rewriter) -> None:
    """登記一個改寫器。`name` 須唯一。"""
    if name in _REWRITERS:
        logger.warning("rewriter %r already registered, overwriting", name)
    _REWRITERS[name] = fn


def current_context() -> dict[str, Any]:
    """本次 `check_output` 呼叫端傳進來的 context(例如 `user_input`)。"""
    return _CONTEXT.get()


def check_output(output: str, context: dict[str, Any] | None = None) -> Verdict:
    """先跑所有驗證器(第一個失敗就返回),全部通過再依序跑改寫器。**不 raise。**

    驗證器或改寫器本身拋例外時一律視為失敗:L3 是最後一道關卡,寧可不回答,
    也不要讓沒檢查過的內容出去。
    """
    if not isinstance(output, str):
        return Verdict(False, "", "type", "output must be str")

    token = _CONTEXT.set(context or {})
    try:
        for name, entry in _VALIDATORS.items():
            try:
                ok, msg = entry.fn(output)
            except Exception:
                logger.exception("validator %r raised", name)
                return Verdict(False, output, name, "validator crashed", entry.user_message)
            if not ok:
                return Verdict(False, output, name, msg, entry.user_message)

        for name, fn in _REWRITERS.items():
            try:
                output = fn(output)
            except Exception:
                logger.exception("rewriter %r raised", name)
                return Verdict(False, output, name, "rewriter crashed")
        return Verdict(True, output)
    finally:
        _CONTEXT.reset(token)


def validate_output(output: str) -> tuple[bool, str]:
    """舊介面:只回驗證結果,不含改寫。失敗時 msg 為 `"[name] detail"`。"""
    v = check_output(output)
    return (True, "") if v.ok else (False, f"[{v.failed}] {v.detail}")


def clear_validators() -> None:
    """測試用:清空登記表。"""
    _VALIDATORS.clear()
    _REWRITERS.clear()
