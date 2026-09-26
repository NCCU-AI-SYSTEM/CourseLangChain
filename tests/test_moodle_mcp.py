"""Moodle MCP(每位使用者自己登入)接線驗證。

跑:`uv run python -m tests.test_moodle_mcp`(在 CourseLangChain/ 下)
不需要 LLM、PostgreSQL,也不需要真的 NCCU 帳號 —— 會自己在隨機 port 起
mcp_servers/moodle_stub.py(密碼一律 demo)。
"""
from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import time

from tools import mcp_tools, session_moodle
from tools.mcp_tools import MOODLE_STATUS, moodle_tools, tool_status, verify_moodle_login

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
        [sys.executable, "-m", "mcp_servers.moodle_stub", "--port", str(port)],
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
    raise RuntimeError(f"假 Moodle server 在 30 秒內沒有起來(port {port})")


def _cfg(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def test_disabled() -> None:
    print("[未設定 URL = 不啟用]")
    check(moodle_tools(url="") == [], "空 URL 回空清單")
    code, _ = verify_moodle_login("112703016", "demo", url="")
    check(code == 503, f"未啟用時登入回 503(實際 {code})")


def test_idle_timeout() -> None:
    print("[閒置逾時:自動清掉帳密]")
    session_moodle.set_credentials("sess-ttl", "112703016", "demo")
    check(session_moodle.headers_for("sess-ttl") is not None, "剛登入拿得到 header")

    original = session_moodle._IDLE_TIMEOUT_SEC
    session_moodle._IDLE_TIMEOUT_SEC = 0  # 視同立刻過期
    try:
        check(session_moodle.headers_for("sess-ttl") is None, "閒置過久後拿不到 header")
        check(session_moodle.summary("sess-ttl") == {"connected": False}, "狀態也變成未連結")
    finally:
        session_moodle._IDLE_TIMEOUT_SEC = original

    session_moodle.set_credentials("sess-ttl2", "112703016", "demo")
    check(session_moodle.headers_for("sess-ttl2") is not None, "逾時只清過期的,新的 session 不受影響")


def test_load(url: str) -> dict:
    print("[載入工具:不需要帳密]")
    tools = {t.name: t for t in moodle_tools(url=url)}
    expected = {"list_courses", "upcoming_deadlines", "list_assignments", "get_grades"}
    check(expected <= set(tools), f"取得 Moodle 工具(實際:{sorted(tools)})")
    if "list_courses" in tools:
        check("config" not in tools["list_courses"].args, "注入用的 config 不會出現在給模型看的參數裡")
    check(tool_status("list_courses") == MOODLE_STATUS, "Moodle 工具有自己的進度提示")
    return tools


def test_not_logged_in(tools: dict) -> None:
    print("[還沒登入:不打 server]")
    out = tools["list_courses"].invoke({}, config=_cfg("sess-none"))
    check(out.startswith("ERROR: 使用者還沒有連結 Moodle"), f"回「還沒連結」(實際:{out[:40]!r})")


def test_verify(url: str) -> None:
    print("[登入驗證]")
    code, msg = verify_moodle_login("112703016", "wrong", url=url)
    check(code == 401 and "學號或密碼錯誤" in msg, f"密碼錯回 401(實際 {code}:{msg[:30]})")
    code, msg = verify_moodle_login("112703016", "demo", url=url)
    check(code == 200 and "2 門課" in msg, f"密碼對回 200 並回報本學期課數(實際 {code}:{msg})")
    code, _ = verify_moodle_login("112703016", "demo", url=f"http://127.0.0.1:{_free_port()}/mcp")
    check(code == 503, f"服務沒開回 503(實際 {code})")


def test_logged_in(tools: dict) -> None:
    print("[已登入:依 session 帶帳密]")
    session_moodle.set_credentials("sess-a", "112703016", "demo")
    session_moodle.set_credentials("sess-b", "110000001", "demo")
    list_courses = tools["list_courses"]

    data = json.loads(list_courses.invoke({}, config=_cfg("sess-a")))
    check(data.get("_stub_student_id") == "112703016", "同步呼叫帶到 session A 的學號")
    check(data.get("count") == 2, "預設只回本學期(2 門)")

    async def both() -> list[str]:
        return await asyncio.gather(
            list_courses.ainvoke({"sem": "all"}, config=_cfg("sess-a")),
            list_courses.ainvoke({"sem": "all"}, config=_cfg("sess-b")),
        )

    a, b = (json.loads(x) for x in asyncio.run(both()))
    check(
        a["_stub_student_id"] == "112703016" and b["_stub_student_id"] == "110000001",
        "兩個 session 同時呼叫,各帶各的帳密,不會串",
    )
    check(a["count"] == 3, "參數 sem=all 有傳到 server")

    async def sync_call_inside_loop() -> str:
        return list_courses.invoke({}, config=_cfg("sess-b"))

    data = json.loads(asyncio.run(sync_call_inside_loop()))
    check(data.get("_stub_student_id") == "110000001", "在 event loop 裡同步呼叫也帶得到帳密")

    out = tools["upcoming_deadlines"].invoke({"days": 3}, config=_cfg("sess-a"))
    check("繼承與多型" in out and "鏈結串列" not in out, "參數 days 有作用(3 天內只剩一筆)")


def test_login_failed(tools: dict) -> None:
    print("[執行期登入失敗:清掉帳密、不重試]")
    session_moodle.set_credentials("sess-c", "112703016", "changed")
    out = tools["get_grades"].invoke({"course_id": 90001}, config=_cfg("sess-c"))
    check(out.startswith("ERROR: Moodle 登入失敗"), f"回登入失敗(實際:{out[:40]!r})")
    check(session_moodle.headers_for("sess-c") is None, "帳密已被清掉")
    out = tools["get_grades"].invoke({"course_id": 90001}, config=_cfg("sess-c"))
    check(out.startswith("ERROR: 使用者還沒有連結 Moodle"), "下一次直接回「還沒連結」,不再打 server")


def test_ws_error_is_not_login_failure(tools: dict, url: str) -> None:
    print("[WS 權限錯誤:不可當成登入失敗]")
    session_moodle.set_credentials("sess-ws", "112703016", "wserror")
    out = tools["list_courses"].invoke({}, config=_cfg("sess-ws"))
    check(
        out.startswith("ERROR:") and "nopermissions" in out,
        f"權限錯誤原樣轉交給模型(實際:{out[:50]!r})",
    )
    check(session_moodle.headers_for("sess-ws") is not None, "不會把使用者登出(帳密保留)")
    code, msg = verify_moodle_login("112703016", "wserror", url=url)
    check(code == 200, f"登入驗證不把權限錯誤當成密碼錯(實際 {code}:{msg[:40]})")


def test_prompt(tools: dict) -> None:
    print("[prompt 只在有 Moodle 工具時才加意圖 F]")
    from agents.brain_agent import _moodle_prompt

    check(_moodle_prompt([]) == "", "沒有 Moodle 工具時 prompt 完全不變")
    text = _moodle_prompt(list(tools.values()))
    check("## F." in text and "list_courses(sem)" in text, "列出工具並加入意圖 F")
    check("不要自己翻譯" in text, "role 代碼「表外照原文」的規則有寫進去")


def test_api() -> None:
    print("[API:/api/moodle]")
    from fastapi.testclient import TestClient

    import app as app_module

    client = TestClient(app_module.app)  # 不用 with:不觸發 lifespan 的資料庫檢查

    r = client.get("/api/moodle", params={"session_id": "api-1"})
    check(r.json() == {"enabled": True, "connected": False}, f"未登入狀態(實際 {r.json()})")

    r = client.post(
        "/api/moodle", json={"session_id": "api-1", "username": "112703016", "password": "demo"}
    )
    check(
        r.status_code == 200 and r.json().get("connected") is True,
        f"登入成功(實際 {r.status_code} {r.json()})",
    )
    check("demo" not in r.text and "112703016" not in r.text, "回應裡沒有密碼,學號也有遮罩")

    r = client.get("/api/moodle", params={"session_id": "api-1"})
    check(r.json().get("connected") is True, "登入後查得到狀態")

    r = client.delete("/api/moodle", params={"session_id": "api-1"})
    check(
        r.json().get("connected") is False and session_moodle.headers_for("api-1") is None,
        "登出後帳密清掉",
    )

    codes = [
        client.post(
            "/api/moodle",
            json={"session_id": "api-2", "username": "112703016", "password": "bad"},
        ).status_code
        for _ in range(4)
    ]
    check(codes == [401, 401, 401, 429], f"連錯 3 次後擋下第 4 次,不再替使用者嘗試(實際 {codes})")

    r = client.post("/api/moodle", json={"session_id": "api-3", "username": "", "password": ""})
    check(r.status_code == 400, "空白帳密回 400")


def test_server_down(tools: dict, proc: subprocess.Popen) -> None:
    print("[執行期服務掛掉]")
    session_moodle.set_credentials("sess-d", "112703016", "demo")
    proc.terminate()
    proc.wait(timeout=10)
    out = tools["list_courses"].invoke({}, config=_cfg("sess-d"))
    check(out.startswith("ERROR: Moodle 服務暫時無法連線"), f"回服務無法連線(實際:{out[:40]!r})")
    check(session_moodle.headers_for("sess-d") is not None, "服務掛掉不是登入失敗,帳密保留")


if __name__ == "__main__":
    test_disabled()
    test_idle_timeout()
    port = _free_port()
    proc = _start_stub(port)
    url = f"http://127.0.0.1:{port}/mcp"
    mcp_tools.MOODLE_MCP_URL = url  # 給 /api/moodle 與 brain_agent 用;其他測試直接傳 url
    try:
        tools = test_load(url)
        if tools:
            test_not_logged_in(tools)
            test_verify(url)
            test_logged_in(tools)
            test_login_failed(tools)
            test_ws_error_is_not_login_failure(tools, url)
            test_prompt(tools)
            test_api()
            test_server_down(tools, proc)
    finally:
        if proc.poll() is None:
            proc.kill()
    print(f"\n結果:{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)
