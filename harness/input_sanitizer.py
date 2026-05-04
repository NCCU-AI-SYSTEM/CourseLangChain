"""L1:輸入清理。

把判定為不安全的輸入直接擋下,不要送進 LLM。當前規則保守(偵測常見的
prompt injection / 控制 token / 過長輸入),寧可誤拒也不放行。新增規則時請寫單元測試。

公開介面只有 `sanitize_input(raw)`,回傳 `(text, is_safe)`。FastAPI / CLI 入口呼叫這個函數,
`is_safe=False` 時直接給用戶友善訊息,不要再呼叫 agent。
"""
from __future__ import annotations

import re

MAX_INPUT_LENGTH = 500

INJECTION_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|above)", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|endoftext\|>", re.IGNORECASE),
    re.compile(r"<\|.*?\|>"),
    re.compile(r"^\s*system\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*assistant\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"</?\s*(system|user|assistant)\s*>", re.IGNORECASE),
)


def sanitize_input(raw: str) -> tuple[str, bool]:
    """檢查 + 截斷使用者輸入。

    Returns:
        (cleaned_text, is_safe)
        - is_safe=True:可送進 agent
        - is_safe=False:呼叫端應拒絕,並回友善訊息
    """
    if not isinstance(raw, str):
        return "", False

    text = raw.strip()
    if not text:
        return "", False

    if len(text) > MAX_INPUT_LENGTH:
        text = text[:MAX_INPUT_LENGTH]

    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            return "", False

    return text, True
