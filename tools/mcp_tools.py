"""外部 MCP server 提供的工具(目前兩個來源:另一組的校內網站檢索、使用者本人的 NCCU Moodle)。

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
   main.py 取工具輸出、組 LLM 訊息都假設是字串。
4. **依 session 帶 header(Moodle)。** nccu-moodle-mcp 是多人共用的無狀態 server,帳密放在
   每次請求的 HTTP header。包裝層從 LangGraph 注入的 `RunnableConfig` 取 thread_id
   (= 前端的 session_id),向 `session_moodle` 拿這位使用者的帳密,經 interceptor 塞進這一次呼叫。
   **帳密不是工具參數,模型看不到。** 沒登入就不打 server;對方回登入失敗就清掉帳密、不重試
   (NCCU 連續失敗 5 次會鎖帳號 15 分鐘)。
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool, ToolException

from paths import CAMPUS_WEB_MCP_URL, MOODLE_MCP_URL
from tools import session_moodle

logger = logging.getLogger(__name__)

CAMPUS_WEB_SERVER = "campus_web"
MOODLE_SERVER = "moodle"

# 前端進度提示。外部工具的名字由對方決定、隨時可能改,所以不放進 main.py 的
# `_TOOL_STATUS`(那張表開機時會斷言每個 key 都是已註冊的本地工具),改依來源給一句通用的。
CAMPUS_WEB_STATUS = "正在查詢校內網站資料…"
MOODLE_STATUS = "正在查詢你的 Moodle…"

# 開機時取工具清單的上限:對方服務沒開時要快點放棄,不能讓後端卡在啟動。
_LOAD_TIMEOUT_SEC = 10.0
# 單次工具呼叫的 HTTP 逾時。對方若是「檢索 + LLM 生成」會比純檢索慢很多,抓寬一點。
_CALL_TIMEOUT_SEC = 60.0
# Moodle 每次呼叫都要重走一次 SSO 登入,作業清單還要逐份查繳交狀態,再寬一點。
_MOODLE_CALL_TIMEOUT_SEC = 120.0

# 已成功載入的外部工具名 → 它的進度提示,給 tool_status() 用
_status_by_name: dict[str, str] = {}

# 這一次工具呼叫要帶的 HTTP header。用 ContextVar 而不是參數:轉接器的工具簽名是固定的,
# header 只能經 interceptor 注入;ContextVar 跟著各自的 task 走,併發的呼叫互不干擾。
_call_headers: ContextVar[dict[str, str] | None] = ContextVar("mcp_call_headers", default=None)


@dataclass(frozen=True)
class SessionAuth:
    """需要「依使用者帶帳密」的 server 設定(目前只有 Moodle)。"""

    headers: Callable[[str], dict[str, str] | None]  # session_id → header;None = 還沒登入
    not_logged_in: str  # 還沒登入時直接回給模型的 ERROR(不打 server)
    login_failed_marker: str  # 對方錯誤訊息裡代表「登入失敗」的字樣
    # 例外:帶有這些字樣就**不是**登入失敗。nccu-moodle-mcp 把所有 Web Service 錯誤
    # (含權限不足)都包成 "Moodle login failed",只看前綴會把權限錯誤當成密碼失效,
    # 白白把使用者登出。
    login_failed_excludes: tuple[str, ...]
    on_login_failed: Callable[[str], object]  # 登入失敗時呼叫(清掉帳密)
    login_failed: str  # 登入失敗時回給模型的 ERROR


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


async def _inject_call_headers(request: Any, handler: Callable) -> Any:
    """interceptor:把 `_call_headers` 塞進這一次 MCP 請求。

    轉接器每次呼叫都開新連線,request.headers 會套用到那條連線上,
    所以同一個工具可以替不同使用者帶不同的帳密。
    """
    headers = _call_headers.get()
    return await handler(request.override(headers=headers) if headers else request)


def _session_of(config: RunnableConfig | None) -> str:
    return ((config or {}).get("configurable") or {}).get("thread_id") or ""


def _wrap(
    remote: StructuredTool, *, server: str, unavailable: str, auth: SessionAuth | None = None
) -> StructuredTool:
    """同名、同參數 schema 的包裝:補同步版本、錯誤轉字串、輸出轉純文字、依 session 帶帳密。

    `config` 由 LangChain 依型別注入(不會出現在給模型看的參數裡),用來取 session。
    """
    call_remote = remote.coroutine

    async def _acall(config: RunnableConfig = None, **kwargs: Any) -> str:
        session_id = _session_of(config)
        headers = None
        if auth is not None:
            headers = auth.headers(session_id)
            if not headers:
                # 沒登入就不打 server:對方只會回 Missing credentials,白白多一次往返
                return auth.not_logged_in

        token = _call_headers.set(headers)
        try:
            content, _artifact = await call_remote(**kwargs)
            text = _to_text(content)
        except ToolException as e:
            # 對方工具自己回報的錯誤(CallToolResult.isError):訊息本來就是給模型看的,原樣轉交
            text = f"ERROR: {e}"
        except Exception as e:  # noqa: BLE001 — 連線中斷、逾時等,不能讓整輪 agent 崩掉
            logger.warning("MCP 工具 %s 呼叫失敗:%s", remote.name, _root_cause(e))
            return unavailable
        finally:
            _call_headers.reset(token)

        if (
            auth is not None
            and auth.login_failed_marker in text
            and not any(skip in text for skip in auth.login_failed_excludes)
        ):
            # 帳密已經失效(例如使用者改了密碼):立刻清掉,後面的呼叫就不會再替他失敗一次
            auth.on_login_failed(session_id)
            return auth.login_failed
        return text

    def _call(config: RunnableConfig = None, **kwargs: Any) -> str:
        return _run_sync(_acall(config, **kwargs))

    return StructuredTool(
        name=remote.name,
        description=remote.description,
        args_schema=remote.args_schema,
        func=_call,
        coroutine=_acall,
        metadata={**(remote.metadata or {}), "mcp_server": server},
    )


def load_mcp_tools(
    url: str,
    reserved_names: Iterable[str] = (),
    *,
    server: str = CAMPUS_WEB_SERVER,
    status: str = CAMPUS_WEB_STATUS,
    unavailable: str = "ERROR: 校內網站查詢服務暫時無法使用,請稍後再試。",
    timeout: float = _CALL_TIMEOUT_SEC,
    auth: SessionAuth | None = None,
) -> list[BaseTool]:
    """連到 `url` 的 MCP server(streamable HTTP)取回工具並包好。**任何失敗都回空清單,不 raise。**

    Args:
        url: MCP 端點,例如 `http://localhost:8765/mcp`。空字串 = 不啟用。
        reserved_names: 已被佔用的工具名(本地工具、先載入的其他 MCP server 工具)。撞名時略過 ——
            否則 main.py 的側通道會把外部工具的輸出當成本地工具來解析。
        server / status / unavailable / timeout: 這個 server 的名稱、前端進度提示、
            連不上時回給模型的訊息、單次呼叫逾時。
        auth: 要依使用者帶帳密時提供(見 SessionAuth)。
    """
    if not url:
        return []

    from langchain_mcp_adapters.client import MultiServerMCPClient  # 沒啟用時不必載入

    client = MultiServerMCPClient(
        {
            server: {
                "transport": "streamable_http",
                "url": url,
                "timeout": timeout,
                "sse_read_timeout": timeout,
            }
        },
        tool_interceptors=[_inject_call_headers],
    )

    async def _list_tools() -> list[BaseTool]:
        return await asyncio.wait_for(client.get_tools(), _LOAD_TIMEOUT_SEC)

    try:
        remote_tools = _run_sync(_list_tools())
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "連不上 MCP server %s(%s):%s —— 本次啟動不含這個 server 的工具,選課功能不受影響。",
            server, url, _root_cause(e),
        )
        return []

    reserved = set(reserved_names)
    tools: list[BaseTool] = []
    for remote in remote_tools:
        if remote.name in reserved:
            logger.error("MCP 工具 %r(%s)與已載入的工具同名,已略過。請對方改名。", remote.name, server)
            continue
        if not isinstance(remote, StructuredTool) or remote.coroutine is None:
            logger.error("MCP 工具 %r 不是預期的 StructuredTool,已略過。", remote.name)
            continue
        tools.append(_wrap(remote, server=server, unavailable=unavailable, auth=auth))
    _status_by_name.update({t.name: status for t in tools})
    logger.info("已載入 MCP server %s 的工具:%s", server, [t.name for t in tools])
    return tools


def campus_web_tools(reserved_names: Iterable[str] = ()) -> list[BaseTool]:
    """依 `CAMPUS_WEB_MCP_URL` 載入校內網站工具;未設定就是空清單。"""
    return load_mcp_tools(CAMPUS_WEB_MCP_URL, reserved_names)


# ── Moodle:使用者本人的資料,每位使用者自己登入 ─────────────────────────────

# nccu-moodle-mcp 登入失敗時的錯誤字樣(其 app.py 的 run_tool:"Moodle login failed: ...")
_MOODLE_LOGIN_FAILED_MARKER = "Moodle login failed"
# 但它把所有 Web Service 錯誤也包成同一句(`moodle_client.ws`:"WS error [<代碼>]: ...")。
# 例如旁聽的課沒有「檢視課程參與者」權限 —— 那時 SSO 早就登入成功了,帳密是好的,
# 不能當成登入失敗把人登出。
_MOODLE_WS_ERROR_MARKER = "WS error"

MOODLE_AUTH = SessionAuth(
    headers=session_moodle.headers_for,
    not_logged_in=(
        "ERROR: 使用者還沒有連結 Moodle 帳號。請使用者先在畫面上登入 Moodle;"
        "不要重試這個工具,也不要請使用者在對話裡輸入密碼。"
    ),
    login_failed_marker=_MOODLE_LOGIN_FAILED_MARKER,
    # WS error = 對方的 Web Service 回報錯誤(例如某門課沒有「檢視課程參與者」權限)。
    # SSO 其實已經登入成功,帳密是好的,不可以因此把使用者登出。
    login_failed_excludes=(_MOODLE_WS_ERROR_MARKER,),
    on_login_failed=session_moodle.clear_credentials,
    login_failed=(
        "ERROR: Moodle 登入失敗(密碼可能已經變更),已替使用者登出。請使用者重新登入;"
        "不要重試 —— 連續失敗 5 次,學校會鎖帳號 15 分鐘。"
    ),
)


def moodle_enabled() -> bool:
    return bool(MOODLE_MCP_URL)


def moodle_tools(reserved_names: Iterable[str] = (), url: str | None = None) -> list[BaseTool]:
    """依 `MOODLE_MCP_URL` 載入 Moodle 工具;未設定就是空清單。

    取工具清單不需要帳密(nccu-moodle-mcp 只在呼叫工具時才檢查),所以開機時、
    任何使用者登入之前就能先載入。`url` 只給測試覆寫用。
    """
    return load_mcp_tools(
        MOODLE_MCP_URL if url is None else url,
        reserved_names,
        server=MOODLE_SERVER,
        status=MOODLE_STATUS,
        unavailable="ERROR: Moodle 服務暫時無法連線,請稍後再試。",
        timeout=_MOODLE_CALL_TIMEOUT_SEC,
        auth=MOODLE_AUTH,
    )


def verify_moodle_login(username: str, password: str, url: str | None = None) -> tuple[int, str]:
    """用這組帳密實際登入一次,回 (HTTP 狀態碼, 給使用者看的訊息)。

    **只試一次、不重試**:NCCU 連續失敗 5 次會鎖帳號。呼叫的是 `list_courses`
    (一次 SSO 登入 + 一次查詢),順便回報本學期課數,讓使用者確認登入的是自己的帳號。
    `url` 只給測試覆寫用。
    """
    url = MOODLE_MCP_URL if url is None else url
    if not url:
        return 503, "Moodle 功能沒有啟用(後端未設定 MOODLE_MCP_URL)。"

    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            MOODLE_SERVER: {
                "transport": "streamable_http",
                "url": url,
                "timeout": _MOODLE_CALL_TIMEOUT_SEC,
                "sse_read_timeout": _MOODLE_CALL_TIMEOUT_SEC,
                "headers": {
                    session_moodle.USER_HEADER: username,
                    session_moodle.PASS_HEADER: password,
                },
            }
        }
    )

    async def _login() -> str:
        tools = await asyncio.wait_for(client.get_tools(), _LOAD_TIMEOUT_SEC)
        list_courses = next((t for t in tools if t.name == "list_courses"), None)
        if list_courses is None or list_courses.coroutine is None:
            raise RuntimeError("Moodle server 沒有提供 list_courses 工具")
        content, _artifact = await list_courses.coroutine()
        return _to_text(content)

    try:
        text = _run_sync(_login())
    except ToolException as e:
        message = str(e)
        if _MOODLE_WS_ERROR_MARKER in message:
            # SSO 已經成功,只是某門課的 Web Service 沒權限(例如旁聽的課看不到參與者)。
            # 帳密是好的 —— 判成密碼錯的話,這種使用者連登入都會被擋,還會累計失敗次數。
            logger.info("Moodle 登入成功,但列課程時遇到 WS 錯誤:%s", message)
            return 200, "已連結 Moodle(有部分課程的資訊查不到,不影響作業查詢)。"
        if _MOODLE_LOGIN_FAILED_MARKER in message:
            return 401, "學號或密碼錯誤。請確認後再試 —— 連續失敗 5 次,學校會鎖帳號 15 分鐘。"
        logger.warning("Moodle 登入驗證時,工具回報錯誤:%s", message)
        return 502, "Moodle 暫時無法登入,請稍後再試。"
    except Exception as e:  # noqa: BLE001
        logger.warning("Moodle 登入驗證失敗:%s", _root_cause(e))
        return 503, "Moodle 服務暫時無法連線,請稍後再試。"

    try:
        count = json.loads(text).get("count")
    except (ValueError, AttributeError):
        count = None
    if count is None:
        return 200, "已連結 Moodle。"
    return 200, f"已連結 Moodle,本學期有 {count} 門課。"


def tool_status(name: str) -> str | None:
    """外部工具的進度提示文字;不是外部工具就回 None。"""
    return _status_by_name.get(name)
