"""段 2 tool wrapper 驗證:schedule_tool / query_courses_tool / registry。

跑:`.venv/bin/python -m tests.test_schedule_tool`(在 CourseLangChain/ 下)
schedule_tool 用真實 data.db id;query_courses_tool 用假 retriever 測格式(不載 2GB pickle)。
"""
from __future__ import annotations

import sqlite3

from tools import query_courses as qc_mod
from tools.query_courses import query_courses_tool
from tools.schedule_tool import _split_ids, schedule_tool

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


def _real_ids(n: int = 8) -> list[str]:
    conn = sqlite3.connect("data.db")
    rows = conn.execute(
        "SELECT id FROM COURSE WHERE time NOT IN ('','未定或彈性') AND point > 0 LIMIT ?",
        (n,),
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def test_split_ids() -> None:
    print("[_split_ids]")
    check(_split_ids("1,2, 3") == ["1", "2", "3"], "逗號+空白分隔")
    check(_split_ids("1\n2、3 4") == ["1", "2", "3", "4"], "換行/頓號/空白混合")
    check(_split_ids("1,1,2") == ["1", "2"], "去重保序")
    check(_split_ids("  ") == [], "全空白 → 空清單")


def test_schedule_tool_happy() -> None:
    print("[schedule_tool 正常路徑]")
    ids = _real_ids(8)
    check(len(ids) >= 4, f"從 data.db 取得 {len(ids)} 個真實 id")
    out = schedule_tool.invoke(
        {"course_ids": ",".join(ids), "min_credits": 2, "max_credits": 12, "max_results": 3}
    )
    check(not out.startswith("ERROR"), "正常輸入不回 ERROR")
    check("方案" in out and "| 課程名稱 |" in out, "輸出含 Markdown 課表方案")
    check("學分" in out, "輸出含學分欄位")
    print("    --- 範例輸出(前 400 字)---")
    print("    " + out[:400].replace("\n", "\n    "))


def test_schedule_tool_errors() -> None:
    print("[schedule_tool 錯誤/邊界]")
    check(schedule_tool.invoke({"course_ids": ""}).startswith("ERROR"), "空 id → ERROR")
    check(
        schedule_tool.invoke({"course_ids": "0000000000000"}).startswith("ERROR"),
        "全找不到的 id → ERROR",
    )
    check(
        schedule_tool.invoke(
            {"course_ids": "1", "min_credits": 20, "max_credits": 5}
        ).startswith("ERROR"),
        "下限 > 上限 → ERROR",
    )
    # 不可能達成的學分 → 給說明而非 ERROR
    ids = _real_ids(5)
    hint = schedule_tool.invoke(
        {"course_ids": ",".join(ids), "min_credits": 999, "max_credits": 9999}
    )
    check(not hint.startswith("ERROR") and "排不出" in hint, "學分過高 → 回說明(非 ERROR)")


def test_query_courses_format() -> None:
    print("[query_courses_tool 格式(假 retriever)]")

    class _FakeDoc:
        def __init__(self, meta):
            self.metadata = meta

    class _FakeRetriever:
        def invoke(self, _):
            return [
                _FakeDoc({"id": "1142000348021", "name": "管理學", "time": "五D56",
                          "teacher": "李老師", "point": 3}),
                _FakeDoc({"id": "1142000348051", "name": "管理學", "time": "三D56",
                          "teacher": "呂老師", "point": 3}),
            ]

    qc_mod._retriever = _FakeRetriever()  # 注入假 retriever,跳過 pickle 載入
    out = query_courses_tool.invoke({"keyword": "管理", "top_k": 5})
    qc_mod._retriever = None  # 還原
    check("course_id: 1142000348021" in out, "輸出含 13 位 course_id")
    check("學分: 3" in out and "老師: 李老師" in out, "輸出含學分與老師")


def test_query_courses_sql_filter() -> None:
    print("[query_courses_tool sql_filter 分支(合併自 retrieval)]")
    out = query_courses_tool.invoke(
        {"keyword": "管理", "top_k": 3, "sql_filter": "time GLOB '*三*'"}
    )
    check(not out.startswith("ERROR"), "sql_filter 分支不回 ERROR")
    check("course_id" in out, "sql_filter 分支輸出仍含 course_id")
    # 壞 filter 應退回全表(不 crash)
    bad = query_courses_tool.invoke({"keyword": "管理", "top_k": 2, "sql_filter": "this is not sql"})
    check(not bad.startswith("ERROR"), "無效 sql_filter 退回全表而非 ERROR")


def test_registry() -> None:
    print("[registry 註冊]")
    from tools.registry import all_tools

    names = {getattr(t, "name", "") for t in all_tools()}
    check("schedule_tool" in names, "schedule_tool 已註冊")
    check("query_courses_tool" in names, "query_courses_tool 已註冊")


if __name__ == "__main__":
    test_split_ids()
    test_schedule_tool_happy()
    test_schedule_tool_errors()
    test_query_courses_format()
    test_query_courses_sql_filter()
    test_registry()
    print(f"\n結果:{_PASS} passed, {_FAIL} failed")
    raise SystemExit(1 if _FAIL else 0)
