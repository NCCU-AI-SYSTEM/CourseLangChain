"""MCP 外部工具接線驗證:載入、同步/非同步呼叫、錯誤不外洩、服務掛掉時照常啟動。

跑:`uv run python -m tests.test_mcp_tools`(在 CourseLangChain/ 下)
不需要 LLM 也不需要 PostgreSQL —— 會自己在隨機 port 起 mcp_servers/campus_web_stub.py。
"""
from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import time

from tools.mcp_tools import CAMPUS_WEB_STATUS, load_mcp_tools, tool_status

_PASS = 0
_FAIL = 0


def check(cond: bool, msg: str) -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  ✓ {msg}")
    else:
        _FAIL += 1
        print(f"  ✗ FAIL: {msg}")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_stub(port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "mcp_servers.campus_web_stub", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return proc
        except OSError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError(f"假 MCP server 在 30 秒內沒有起來(port {port})")


def test_disabled() -> None:
    print("[未設定 URL = 不啟用]")
    check(load_mcp_tools("") == [], "空 URL 回空清單,不嘗試連線")


def test_server_down() -> None:
    print("[服務沒開:照常啟動]")
    port = _free_port()
    start = time.time()
    tools = load_mcp_tools(f"http://127.0.0.1:{port}/mcp")
    elapsed = time.time() - start
    check(tools == [], "連不上時回空清單,不 raise")
    check(elapsed < 15, f"不會卡住啟動(實際 {elapsed:.1f}s)")


def test_load_and_call(url: str) -> list:
    print("[載入與呼叫]")
    tools = load_mcp_tools(url)
    by_name = {t.name: t for t in tools}
    check("search_campus_web" in by_name, f"取得 search_campus_web(實際:{list(by_name)})")
    tool = by_name.get("search_campus_web")
    if tool is None:
        return tools

    out = asyncio.run(tool.ainvoke({"query": "註冊組在哪裡"}))
    check(isinstance(out, str), f"非同步呼叫回 str(實際 {type(out).__name__})")
    check("來源: https://aca.nccu.edu.tw" in str(out), "回傳含來源網址")

    out = tool.invoke({"query": "休學要怎麼辦"})
    check(
        isinstance(out, str) and "註冊組" in out,
        "同步呼叫可用(SafeAgentExecutor / 非串流端點走這條)",
    )

    async def _sync_call_inside_loop() -> str:
        return tool.invoke({"query": "註冊組"})

    out = asyncio.run(_sync_call_inside_loop())
    check(isinstance(out, str) and "註冊組" in out, "在 event loop 裡同步呼叫也不會 RuntimeError")

    out = tool.invoke({"query": "今天天氣如何"})
    check("找不到" in out, "查無資料時回白話訊息")

    out = tool.invoke({"query": "   "})
    check(out.startswith("ERROR:"), "server 端回的 ERROR 字串原樣轉交")

    out = tool.invoke({"query": "註冊組", "max_results": "not-a-number"})
    check(
        isinstance(out, str) and out.startswith("ERROR:"),
        f"server 回報 isError 時轉成 ERROR 字串,不 raise(實際:{str(out)[:60]!r})",
    )

    check(tool_status("search_campus_web") == CAMPUS_WEB_STATUS, "外部工具有進度提示")
    check(tool_status("retrieve_tool") is None, "本地工具不受影響(仍走 main.py 的對照表)")
    return tools


def test_reserved_name(url: str) -> None:
    print("[與本地工具撞名]")
    tools = load_mcp_tools(url, reserved_names={"search_campus_web"})
    check(all(t.name != "search_campus_web" for t in tools), "撞名的外部工具被略過")


def test_prompt(tools: list) -> None:
    print("[prompt 只在有外部工具時才加意圖 E]")
    from agents.brain_agent import _campus_web_prompt

    check(_campus_web_prompt([]) == "", "沒有外部工具時 prompt 完全不變")
    text = _campus_web_prompt(tools)
    check("search_campus_web(query, max_results)" in text, "列出工具名與參數")
    check("## E." in text, "加入意圖 E")


def test_server_dies_mid_session(url: str, proc: subprocess.Popen) -> None:
    print("[執行期服務掛掉]")
    tools = load_mcp_tools(url)
    tool = next((t for t in tools if t.name == "search_campus_web"), None)
    proc.terminate()
    proc.wait(timeout=10)
    if tool is None:
        check(False, "server 關掉前沒能載入工具")
        return
    out = tool.invoke({"query": "註冊組"})
    check(
        isinstance(out, str) and out.startswith("ERROR:"),
        f"連線失敗轉成 ERROR 字串,不讓整輪 agent 崩掉(實際:{str(out)[:60]!r})",
    )


if __name__ == "__main__":
    test_disabled()
    test_server_down()
    port = _free_port()
    proc = _start_stub(port)
    url = f"http://127.0.0.1:{port}/mcp"
    try:
        tools = test_load_and_call(url)
        test_reserved_name(url)
        test_prompt(tools)
        test_server_dies_mid_session(url, proc)
    finally:
        if proc.poll() is None:
            proc.kill()
    print(f"\n結果:{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)
