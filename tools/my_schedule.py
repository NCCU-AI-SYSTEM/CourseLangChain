"""my_schedule_tool —— 讓 agent 讀寫「使用者當前已排定的課表」(面板同一份資料)。

為什麼需要它:使用者可能在課表面板上手動加課/刪課,那些動作不經過 LLM,
對話歷史裡看不到。agent 要回答「我現在排幾學分了」「幫我把週五那門拿掉」,
就必須讀這份 session 狀態,而不是從對話裡猜。

session_id 從 `RunnableConfig` 的 `configurable.thread_id` 注入——
**LLM 看不到也填不了這個參數**,不會被模型亂編。

契約(見 tools/registry.py):扁平參數、回字串不 raise、無 LLM。
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from tools import session_schedule as store

from .registry import register_tool

_ACTIONS = ("view", "add", "remove", "clear")


def _session_of(config: RunnableConfig | None) -> str:
    return ((config or {}).get("configurable") or {}).get("thread_id") or ""


def _render(session_id: str, prefix: str = "") -> str:
    body = store.format_markdown(store.get_schedule(session_id))
    return f"{prefix}\n\n{body}" if prefix else body


@tool
def my_schedule_tool(
    action: str = "view",
    course_id: str = "",
    config: RunnableConfig = None,
) -> str:
    """查看或修改使用者「目前已排定的課表」(與畫面上的課表面板同步)。

    什麼時候用:使用者問「我現在排了哪些課 / 幾學分」、要求「把某門加進課表 /
    從課表移除」、或排課前要知道既有課表以避免衝堂時。

    參數:
    - action: "view" 查看 / "add" 加課 / "remove" 移除 / "clear" 清空
    - course_id: action 為 add 或 remove 時必填,13 位課程代碼(照抄 query_courses_tool 的結果)

    加課時若與課表現有課程衝堂、或加入的是同一門課的另一個班(一張課表只能有一門),
    會**自動移除舊的那幾門**再加入新課,並在回覆中說明移除原因。
    """
    session_id = _session_of(config)
    if not session_id:
        return "ERROR: 無法取得對話 session,課表功能暫時無法使用。"

    action = (action or "view").strip().lower()
    if action not in _ACTIONS:
        return f"ERROR: action 只能是 {' / '.join(_ACTIONS)},收到 {action!r}。"

    try:
        if action == "view":
            return _render(session_id)

        if action == "clear":
            store.clear_schedule(session_id)
            return "已清空課表。"

        if not course_id.strip():
            return f"ERROR: action={action} 需要提供 course_id(13 位課程代碼)。"

        if action == "remove":
            ok = store.remove_course(session_id, course_id)
            prefix = (
                f"已從課表移除 {course_id}。"
                if ok
                else f"課表中沒有 {course_id},未做變更。"
            )
            return _render(session_id, prefix)

        # add
        result = store.add_course(session_id, course_id)
        if not result["ok"]:
            return f"ERROR: {result['error']}"
        added = result["added"]
        if result["already"]:
            return _render(session_id, f"{added.name} 已經在課表裡了,未重複加入。")
        removed = result["removed"]
        if removed:
            # 移除原因有兩種,講錯會讓使用者困惑(同名換班時說「時間衝突」是假的)
            reasons = result.get("removed_reasons", {})
            names = "、".join(
                f"{c.name}({c.time_str}"
                + (",同一門課的另一班" if reasons.get(c.course_id) == "same_name" else ",時間衝突")
                + ")"
                for c in removed
            )
            prefix = f"已加入 {added.name}({added.time_str or '時間未定'})。已移除:{names}。"
        else:
            prefix = f"已加入 {added.name}({added.time_str or '時間未定'})。"
        return _render(session_id, prefix)
    except Exception as e:  # noqa: BLE001 — 契約:回字串而非 raise
        return f"ERROR: 課表操作失敗({e})。"


register_tool(my_schedule_tool)
