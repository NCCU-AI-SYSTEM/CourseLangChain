"""排課方案側通道測試 —— 「一鍵套用推薦課表」倚賴的兩段。

跑:`uv run python -m tests.test_schedule_plans`(在 CourseLangChain/ 下)

驗兩件事:
1. `plans_to_dicts` 與 `format_schedules_markdown` 對同一份 schedules 的
   方案編號、順序、學分、上課日**完全一致** —— 使用者讀到的「方案 2」與按鈕
   套用的必須是同一組,這兩個函式一旦走鐘就會靜默錯配。
2. `_PLAN_CACHE` 的存取契約:用工具回傳字串領走、領完即刪、超過上限淘汰最舊。

純函數部分不需要資料庫;需要真實課程的案例拿不到資料時會自行略過。
"""
from __future__ import annotations

import os
import sqlite3

import paths
from tools import schedule_tool as st_mod
from tools.schedule_tool import pop_plans, schedule_tool
from tools.scheduler import (
    format_schedules_markdown,
    make_course,
    plans_to_dicts,
    rank_schedules,
)

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
    """取幾個真實課程 id(與 tests/test_schedule_tool.py 同一套取法)。"""
    if not paths.USE_SQLITE:
        try:
            import psycopg2

            conn = psycopg2.connect(paths.DATABASE_URL)
            cur = conn.cursor()
            cur.execute(
                "SELECT id FROM public.course "
                "WHERE y = %s AND s = %s AND time_raw NOT IN ('','未定或彈性') "
                "AND point > 0 LIMIT %s",
                (paths.COURSE_YEAR, paths.COURSE_SEMESTER, n),
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [r[0] for r in rows]
        except Exception:
            return []

    if not os.path.exists("data.db"):
        return []
    conn = sqlite3.connect("data.db")
    rows = conn.execute(
        "SELECT id FROM COURSE WHERE time NOT IN ('','未定或彈性') AND point > 0 LIMIT ?",
        (n,),
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _fake_schedules() -> list[list]:
    a = make_course("1142000000001", "計算機概論", 3, "一234", "王老師")
    b = make_course("1142000000002", "線性代數", 3, "三56", "李老師")
    c = make_course("1142000000003", "統計學", 2, "五12", "陳老師")
    return rank_schedules([[a, b], [a, b, c]])


def test_plans_match_markdown() -> None:
    print("[plans_to_dicts 與 Markdown 一致]")
    schedules = _fake_schedules()
    md = format_schedules_markdown(schedules)
    plans = plans_to_dicts(schedules)

    check(len(plans) == len(schedules), f"方案數一致({len(plans)})")
    check([p["index"] for p in plans] == list(range(1, len(plans) + 1)), "編號從 1 連續遞增")
    for p in plans:
        header = f"### 方案 {p['index']}(共 {p['total_credits']} 學分,上課日:{p['days'] or '未定'})"
        check(header in md, f"方案 {p['index']} 的標題與 Markdown 逐字相同")
    # 課程順序也要一致:Markdown 與 dict 都是 sorted(key=course_id)
    first = plans[0]["courses"]
    check(
        [c["course_id"] for c in first] == sorted(c["course_id"] for c in first),
        "方案內課程依 course_id 排序(與 formatter 同)",
    )
    check(
        all(c["name"] in md for c in first),
        "方案內每門課名都出現在 Markdown 裡",
    )
    check(
        all(len(c["course_id"]) == 13 for p in plans for c in p["courses"]),
        "每門課都帶 13 碼 course_id(Markdown 裡刻意沒有)",
    )
    check(plans_to_dicts([]) == [], "空 schedules → 空清單")


def test_plan_cache_contract() -> None:
    print("[_PLAN_CACHE 存取契約]")
    st_mod._PLAN_CACHE.clear()
    plans = plans_to_dicts(_fake_schedules())

    st_mod._remember_plans("OUTPUT-A", plans)
    check(pop_plans("OUTPUT-A") == plans, "用回傳字串領得到方案")
    check(pop_plans("OUTPUT-A") == [], "領完即刪,不會被第二次事件重複送出")
    check(pop_plans("沒看過的字串") == [], "查無資料回空清單而非 KeyError")

    st_mod._remember_plans("EMPTY", [])
    check(pop_plans("EMPTY") == [], "空方案不入快取")

    st_mod._PLAN_CACHE.clear()
    for i in range(st_mod._PLAN_CACHE_MAX + 5):
        st_mod._remember_plans(f"OUT-{i}", plans)
    check(len(st_mod._PLAN_CACHE) == st_mod._PLAN_CACHE_MAX, f"快取上限 {st_mod._PLAN_CACHE_MAX} 生效")
    check(pop_plans("OUT-0") == [], "最舊的已被淘汰")
    check(pop_plans(f"OUT-{st_mod._PLAN_CACHE_MAX + 4}") == plans, "最新的仍在")
    st_mod._PLAN_CACHE.clear()


def test_schedule_tool_registers_plans() -> None:
    print("[schedule_tool 排完就備好方案]")
    ids = _real_ids(8)
    if not ids:
        print("    (略過:拿不到真實課程資料 —— 先起 postgres 或提供 data.db)")
        return

    out = schedule_tool.invoke(
        {"course_ids": ",".join(ids), "min_credits": 2, "max_credits": 12, "max_results": 3}
    )
    if out.startswith("ERROR") or "方案" not in out:
        print(f"    (略過:這批課排不出方案 —— {out[:60]})")
        return

    plans = pop_plans(out)
    check(bool(plans), f"用工具回傳字串領到 {len(plans)} 個方案")
    check(
        all(c["course_id"] in ids for p in plans for c in p["courses"]),
        "方案裡的 course_id 全部來自輸入的候選清單(沒有憑空冒出來的)",
    )
    for p in plans:
        header = f"### 方案 {p['index']}(共 {p['total_credits']} 學分,上課日:{p['days'] or '未定'})"
        check(header in out, f"方案 {p['index']} 與工具實際輸出的標題一致")
    check(pop_plans(out) == [], "領完即刪")


if __name__ == "__main__":
    test_plans_match_markdown()
    test_plan_cache_contract()
    test_schedule_tool_registers_plans()
    print(f"\n結果:{_PASS} passed, {_FAIL} failed")
    raise SystemExit(1 if _FAIL else 0)
