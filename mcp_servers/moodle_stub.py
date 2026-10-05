"""nccu-moodle-mcp 的**假** server —— 給本機開發、測試與展示用,不需要真的 NCCU 帳號。

真的 server 是 Daniel 維護的 nccu-moodle-mcp(每次呼叫都用 header 裡的帳密走 NCCU SSO)。
這支照它的工具名、參數與回傳結構做了其中四個工具,資料全是假的(每筆都有標示)。

帳密規則(錯誤訊息的字樣與真的一致,後端靠 "Moodle login failed" 判斷登入失敗):
- 沒帶 `X-Moodle-Username` / `X-Moodle-Password` → Missing credentials
- 密碼不是 `demo` → Moodle login failed
- 密碼是 `demo` → 任何學號都能登入

用假 server 展示,就沒有「密碼打錯幾次、被學校鎖帳號」的風險。

跑:    uv run python -m mcp_servers.moodle_stub [--host 127.0.0.1] [--port 3033]
後端:  MOODLE_MCP_URL=http://localhost:3033/mcp

刻意不加 `from __future__ import annotations`:FastMCP 靠型別註解辨認 `ctx: Context` 參數。
"""
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError

DEMO_PASSWORD = "demo"
# 用這個密碼登入 = 模擬「SSO 成功,但某個 Web Service 回權限錯誤」的情況
WS_ERROR_PASSWORD = "wserror"
_NOTE = "【PoC 假資料】"
_TPE = ZoneInfo("Asia/Taipei")
_URL = "https://moodle.example.invalid"

_COURSES = [
    {"id": 90001, "name": f"{_NOTE}1151_物件導向程式設計 Object-Oriented Programming",
     "semester": "1151", "role": "student"},
    {"id": 90002, "name": f"{_NOTE}1151_資料結構 Data Structures",
     "semester": "1151", "role": "student"},
    {"id": 90003, "name": f"{_NOTE}1142_微積分 Calculus",
     "semester": "1142", "role": "student"},
]


def _login(ctx: Context) -> str:
    """照 nccu-moodle-mcp 的 run_tool:從 header 取帳密並驗證,回學號。"""
    request = ctx.request_context.request
    headers = request.headers if request is not None else {}
    user = headers.get("x-moodle-username")
    password = headers.get("x-moodle-password")
    if not user or not password:
        raise ToolError(
            "Missing credentials. Configure them in your MCP client settings `headers` "
            "block as 'X-Moodle-Username' and 'X-Moodle-Password'."
        )
    if password == WS_ERROR_PASSWORD:
        # 模擬真 server 的行為:任何 Web Service 錯誤(含權限不足)都會被包成
        # "Moodle login failed",但**帳密其實是好的**(SSO 已經登入成功)。
        raise ToolError(
            "Moodle login failed: WS error [nopermissions]: "
            "抱歉,您目前沒有權限執行 (檢視課程參與者)"
        )
    if password != DEMO_PASSWORD:
        raise ToolError("Moodle login failed: 學號或密碼錯誤(假 server 的密碼是 demo)")
    return user


def _course_rows(sem: str | None) -> list[dict]:
    latest = max(c["semester"] for c in _COURSES)
    want = (sem or "").strip().lower()
    rows = []
    for c in _COURSES:
        if want in ("", "latest") and c["semester"] != latest:
            continue
        if want not in ("", "latest", "all") and c["semester"] != want:
            continue
        rows.append({
            "id": c["id"],
            "name": c["name"],
            "url": f"{_URL}/course/view.php?id={c['id']}",
            "semester": c["semester"],
            "current": c["semester"] == latest,
            "role": c["role"],
        })
    return rows


def _due(days: int) -> str:
    return (datetime.now(_TPE) + timedelta(days=days)).strftime("%Y-%m-%d 23:59")


def list_courses(ctx: Context, sem: str | None = None) -> dict:
    """List the student's enrolled Moodle courses (with the user's role in each),
    filtered by semester. No `sem` = latest semester; "all" = every course."""
    user = _login(ctx)
    items = _course_rows(sem)
    # `_stub_student_id` 是假 server 才有的欄位:讓測試確認 header 真的帶到了這位使用者
    return {"count": len(items), "courses": items, "_stub_student_id": user}


def upcoming_deadlines(ctx: Context, days: int = 14, include_all_role: bool = False) -> dict:
    """List the student's upcoming action events (assignment due dates, quiz closings)
    across courses within the next `days` (default 14)."""
    _login(ctx)
    # (幾天後到期, 事件);只回 `days` 天內的
    upcoming = [
        (2, {"name": f"{_NOTE}作業 2:繼承與多型", "course_id": 90001,
             "course": _COURSES[0]["name"], "role": "student", "due": _due(2),
             "overdue": False, "module": "assign", "url": f"{_URL}/mod/assign/view.php?id=70001"}),
        (5, {"name": f"{_NOTE}小考 1:鏈結串列", "course_id": 90002,
             "course": _COURSES[1]["name"], "role": "student", "due": _due(5),
             "overdue": False, "module": "quiz", "url": f"{_URL}/mod/quiz/view.php?id=70002"}),
    ]
    events = [event for offset, event in upcoming if offset <= days]
    return {"count": len(events), "days": days, "events": events}


def list_assignments(
    ctx: Context, sem: str | None = None, include_all_role: bool = False
) -> dict:
    """List assignments (with due dates and submission status) for the student's courses."""
    _login(ctx)
    items = [
        {"id": 80001, "course_id": 90001, "course": _COURSES[0]["name"], "role": "student",
         "name": f"{_NOTE}作業 1:類別與物件", "due": _due(-5), "opens": _due(-19),
         "cutoff": None, "status": "submitted", "url": f"{_URL}/mod/assign/view.php?id=70003"},
        {"id": 80002, "course_id": 90001, "course": _COURSES[0]["name"], "role": "student",
         "name": f"{_NOTE}作業 2:繼承與多型", "due": _due(2), "opens": _due(-12),
         "cutoff": None, "status": "new", "url": f"{_URL}/mod/assign/view.php?id=70001"},
    ]
    return {"count": len(items), "assignments": items}


def get_grades(ctx: Context, course_id: int) -> dict:
    """Get the student's own grade items for one course (by Moodle course id, from list_courses)."""
    _login(ctx)
    items = [
        {"item": f"{_NOTE}作業 1:類別與物件", "grade": "92.00", "percentage": "92.00 %",
         "range": "0–100", "feedback": "", "type": "mod"},
        {"item": "Course total", "grade": "-", "percentage": "-", "range": "0–100",
         "feedback": "", "type": "course"},
    ]
    return {"course_id": course_id, "count": len(items), "grades": items}


def build_server(host: str, port: int) -> FastMCP:
    """host 在建構時給:FastMCP 依它決定 DNS rebinding 防護(綁 127.0.0.1 時容器連不進來)。"""
    server = FastMCP(
        name="nccu_moodle_stub",
        instructions="NCCU Moodle(PoC 假資料,密碼 demo)。",
        host=host,
        port=port,
    )
    for fn in (list_courses, upcoming_deadlines, list_assignments, get_grades):
        server.tool()(fn)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="nccu-moodle-mcp 的假 server(密碼 demo)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3033)
    args = parser.parse_args()
    build_server(args.host, args.port).run(transport="streamable-http")


if __name__ == "__main__":
    main()
