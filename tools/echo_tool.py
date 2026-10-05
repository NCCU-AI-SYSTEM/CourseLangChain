"""Dummy echo tool — 給協作者做 harness 端到端測試用。

協作者第一個真實 tool 進來後可以刪掉這個檔案 + 從 registry 拿掉。
"""
from __future__ import annotations

from langchain_core.tools import tool

from .registry import register_tool


@tool
def echo_tool(message: str) -> str:
    """當用戶想要把一段文字原樣回傳時呼叫此工具(僅供 harness 煙霧測試)。

    Args:
        message: 任何字串

    Returns:
        前綴 "echo: " 的字串。
    """
    try:
        return f"echo: {message}"
    except Exception as e:
        return f"ERROR: {e}"


register_tool(echo_tool)
