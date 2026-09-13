"""外部 MCP server 提供的工具(目前只有一個來源:另一組維護的校內網站檢索)。

和 `registry.py` 登記的工具不同,這裡的工具**不在本行程裡執行**:開機時連到 MCP server
問它有哪些工具,逐一包成 LangChain tool 交給 `brain_agent`。所以它們不進 registry
(`registry.all_tools()` 只列本地工具,main.py 開機時的名稱斷言也只管本地工具)。

為什麼只有外部服務走 MCP、自家課程工具留在行程內:課程工具依賴行程內的狀態 ——
`schedule_tool._PLAN_CACHE`(側通道)、LangGraph 注入 `RunnableConfig` 的 thread_id、
與 `app.py` 共用的 session 課表 —— 搬到別的行程會靜默失效。

包裝時做的三件事:
1. **外部服務掛了不能拖垮選課功能。** 開機載入失敗 → log 警告、回空清單,agent 照常啟動;
   執行期呼叫失敗 → 回 `"ERROR: ..."` 字串(照 registry.py 的工具合約),不 raise。
2. **同步橋接。** langchain-mcp-adapters 轉出的工具只有 coroutine,但
   `SafeAgentExecutor.run` 與 `app.py` 的非串流端點走同步 `graph.invoke`,
   不補同步版本的話,那兩條路徑一呼叫就 NotImplementedError。
3. **回傳統一成純文字。** 轉接器回的是 content block 清單;本專案的工具一律回 str,
   main.py 取工具輸出、ChatOllama 組訊息都假設是字串。
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool, ToolException

from paths import CAMPUS_WEB_MCP_URL

logger = logging.getLogger(__name__)

CAMPUS_WEB_SERVER = "campus_web"

# 前端進度提示。外部工具的名字由對方決定、隨時可能改,所以不放進 main.py 的
# `_TOOL_STATUS`(那張表開機時會斷言每個 key 都是已註冊的本地工具),改依來源給一句通用的。
CAMPUS_WEB_STATUS = "正在查詢校內網站資料…"

# 開機時取工具清單的上限:對方服務沒開時要快點放棄,不能讓後端卡在啟動。
_LOAD_TIMEOUT_SEC = 10.0
# 單次工具呼叫的 HTTP 逾時。對方若是「檢索 + LLM 生成」會比純檢索慢很多,抓寬一點。
_CALL_TIMEOUT_SEC = 60.0

# 已成功載入的外部工具名,給 tool_status() 判斷來源
_loaded_names: set[str] = set()


def _run_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """在獨立執行緒的新 event loop 跑完 coroutine。

    不直接 `asyncio.run`:呼叫端可能已經在 event loop 裡(uvicorn 匯入 app 時就是),
    那樣會 RuntimeError。換 loop 是安全的 —— 轉接器每次呼叫都開新的 MCP 連線,
    不綁定任何 loop 的狀態。
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _to_text(content: Any) -> str:
    """把轉接器回的 content(str 或 content block 清單)攤平成純文字。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, str)
            or (isinstance(block, dict) and block.get("type") == "text")
        ]
        return "\n".join(p for p in parts if p)
    return str(content)


def _root_cause(e: BaseException) -> str:
    """攤開 ExceptionGroup,回最底層的例外。

    anyio 會把 ConnectError 之類的包成「unhandled errors in a TaskGroup」,
    只印外層的話 log 看不出是連不上、逾時還是對方回錯。
    """
    while isinstance(e, BaseExceptionGroup) and e.exceptions:
        e = e.exceptions[0]
    return f"{type(e).__name__}: {e}"


def _wrap(remote: StructuredTool) -> StructuredTool:
    """同名、同參數 schema 的包裝:補同步版本、錯誤轉字串、輸出轉純文字。"""
    call_remote = remote.coroutine

    async def _acall(**kwargs: Any) -> str:
        try:
            content, _artifact = await call_remote(**kwargs)
        except ToolException as e:
            # 對方工具自己回報的錯誤(CallToolResult.isError):訊息本來就是給模型看的,原樣轉交
            return f"ERROR: {e}"
        except Exception as e:  # noqa: BLE001 — 連線中斷、逾時等,不能讓整輪 agent 崩掉
            logger.warning("MCP 工具 %s 呼叫失敗:%s", remote.name, _root_cause(e))
            return "ERROR: 校內網站查詢服務暫時無法使用,請稍後再試。"
        return _to_text(content)

    def _call(**kwargs: Any) -> str:
        return _run_sync(_acall(**kwargs))

    return StructuredTool(
        name=remote.name,
        description=remote.description,
        args_schema=remote.args_schema,
        func=_call,
        coroutine=_acall,
        metadata={**(remote.metadata or {}), "mcp_server": CAMPUS_WEB_SERVER},
    )


def load_mcp_tools(url: str, reserved_names: Iterable[str] = ()) -> list[BaseTool]:
    """連到 `url` 的 MCP server(streamable HTTP)取回工具並包好。**任何失敗都回空清單,不 raise。**

    Args:
        url: MCP 端點,例如 `http://localhost:8765/mcp`。空字串 = 不啟用。
        reserved_names: 本地工具名。外部工具撞名時略過 —— 否則 main.py 的側通道
            會把外部工具的輸出當成本地工具來解析。
    """
    if not url:
        return []

    from langchain_mcp_adapters.client import MultiServerMCPClient  # 沒啟用時不必載入

    client = MultiServerMCPClient(
        {
            CAMPUS_WEB_SERVER: {
                "transport": "streamable_http",
                "url": url,
                "timeout": _CALL_TIMEOUT_SEC,
                "sse_read_timeout": _CALL_TIMEOUT_SEC,
            }
        }
    )

    async def _list_tools() -> list[BaseTool]:
        return await asyncio.wait_for(client.get_tools(), _LOAD_TIMEOUT_SEC)

    try:
        remote_tools = _run_sync(_list_tools())
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "連不上校內網站 MCP server(%s):%s —— 本次啟動不含校內網站工具,選課功能不受影響。",
            url, _root_cause(e),
        )
        return []

    reserved = set(reserved_names)
    tools: list[BaseTool] = []
    for remote in remote_tools:
        if remote.name in reserved:
            logger.error("MCP 工具 %r 與本地工具同名,已略過。請對方改名。", remote.name)
            continue
        if not isinstance(remote, StructuredTool) or remote.coroutine is None:
            logger.error("MCP 工具 %r 不是預期的 StructuredTool,已略過。", remote.name)
            continue
        tools.append(_wrap(remote))
    _loaded_names.update(t.name for t in tools)
    logger.info("已載入校內網站 MCP 工具:%s", [t.name for t in tools])
    return tools


def campus_web_tools(reserved_names: Iterable[str] = ()) -> list[BaseTool]:
    """依 `CAMPUS_WEB_MCP_URL` 載入校內網站工具;未設定就是空清單。"""
    return load_mcp_tools(CAMPUS_WEB_MCP_URL, reserved_names)


def tool_status(name: str) -> str | None:
    """外部工具的進度提示文字;不是外部工具就回 None。"""
    return CAMPUS_WEB_STATUS if name in _loaded_names else None
