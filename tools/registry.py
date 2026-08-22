"""Tool registry — ReAct agent 可呼叫的工具集中清單。

協作流程(寫一個新 tool):

1. 在 `tools/` 底下新建 `your_tool_name.py`,內容看下方 [Tool 合約] 章節
2. 在本檔案的 `_TOOL_MODULES` 加一行 `from . import your_tool_name`
3. 在 `your_tool_name.py` 裡用 `register_tool(your_tool)` 自我註冊
4. (可選)若 tool 產出需要 L3 輸出驗證,在 tool 模組裡 `harness.register_validator(...)` 自我註冊

不要動 `agent/main_agent.py` 或 `harness/`,新 tool 應該完全在自己的檔案內完成。

[Tool 合約]
-----------
- 用 `@langchain_core.tools.tool` decorator
- 參數型別只用 **扁平** `int` / `float` / `str` / `bool` — 不要 nested dict / list of dict
  (1b/8b 模型對 nested 結構錯誤率極高)
- docstring 第一行是給 LLM 看的「何時呼叫此工具」說明,要短而具體
- 失敗時 **回字串** 而不是 raise,前綴用 `"ERROR: ..."`,讓 agent 能自行決策
- 內部任何 LLM 呼叫請另外起一個 helper,不要直接讓 tool 函數本身依賴 LLM
- pure 計算邏輯(solver / validator / ranker)放 `tools/` 同層獨立檔案,tool wrapper 只做 I/O + 編排

範例骨架:

    # tools/example_tool.py
    from langchain_core.tools import tool
    from .registry import register_tool

    @tool
    def example_tool(keyword: str, max_results: int = 10) -> str:
        \"\"\"當用戶想 X 時呼叫此工具。\"\"\"
        try:
            ...
            return formatted_result
        except Exception as e:
            return f"ERROR: {e}"

    register_tool(example_tool)
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_TOOLS: list[Any] = []
_NAMES: set[str] = set()


def register_tool(tool_obj: Any) -> None:
    """登記一個 LangChain tool。重複名稱會被忽略並 log warning。"""
    name = getattr(tool_obj, "name", None) or getattr(tool_obj, "__name__", repr(tool_obj))
    if name in _NAMES:
        logger.warning("tool %r already registered, skipping", name)
        return
    _TOOLS.append(tool_obj)
    _NAMES.add(name)


def all_tools() -> list[Any]:
    """回傳目前所有已註冊的工具(順序穩定,以註冊順序為準)。"""
    _import_tool_modules()
    return list(_TOOLS)


# ---- 在這裡列出所有 tool 模組,讓它們的 register_tool 副作用觸發 ----
_TOOL_MODULES: tuple[str, ...] = (
    "tools.echo_tool",               # ← dummy,協作者第一個真實 tool 上線後可移除
    "tools.text_to_sql",             # 段 2:自然語言時間限制 → SQL WHERE
    "tools.retrieve",                # 段 2:檢索候選課程(回 course_id)
    "tools.course_detail",           # 段 2:單門課詳情(課綱/評分/教科書)
    "tools.schedule_tool",           # 段 2:接候選 id 排課
    "tools.user_profile",            # 段 3:個人化——讀使用者成績單(可選)
    "tools.my_schedule",             # 段 3:讀寫 session 已排定課表(與前端面板同步)
)

_imported = False


def _import_tool_modules() -> None:
    global _imported
    if _imported:
        return
    import importlib

    for mod_name in _TOOL_MODULES:
        try:
            importlib.import_module(mod_name)
        except Exception:
            logger.exception("failed to import tool module %r", mod_name)
    _imported = True
