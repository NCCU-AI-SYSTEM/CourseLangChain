"""段 1 排課純邏輯的 standalone 驗證。

直接跑:`python -m tests.test_scheduler`(在 CourseLangChain/ 底下)
不依賴 pytest,自帶一個極簡 assert + 報表,方便快速確認可行性。
"""
from __future__ import annotations

import sqlite3

from tools.scheduler import (
    CourseSlot,
    find_schedules,
    has_conflict,
    make_course,
    parse_course_slots,
    validate_schedule,
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


def test_parse_slots() -> None:
    print("[parse_course_slots]")
    # 三234 = 週三第2,3,4節
    s = parse_course_slots("三234")
    check(s == {("三", "2"), ("三", "3"), ("三", "4")}, "三234 解析成週三 2/3/4 節")
    # 多段:三234五56
    s2 = parse_course_slots("三234五56")
    check(("五", "5") in s2 and ("五", "6") in s2 and ("三", "2") in s2, "多段 三234五56 兩天都解析")
    # 未定 / 空
    check(parse_course_slots("未定或彈性") == frozenset(), "未定或彈性 → 空集合")
    check(parse_course_slots("") == frozenset(), "空字串 → 空集合")


def test_conflict_and_validate() -> None:
    print("[has_conflict / validate_schedule]")
    a = make_course("1", "課A", 3, "三234")
    b = make_course("2", "課B", 3, "三34五56")  # 與 A 在 三3,三4 衝
    c = make_course("3", "課C", 3, "五78")       # 與 A、B 都不衝
    check(has_conflict(a, b), "課A 與 課B 衝堂(共用 三3/三4)")
    check(not has_conflict(a, c), "課A 與 課C 不衝堂")

    v = validate_schedule([a, b], 0, 99)
    check(any("衝堂" in x for x in v), "validate 抓到 A/B 衝堂")
    v2 = validate_schedule([a, c], 0, 99)
    check(v2 == [], "A+C 無違規")

    # 學分上下限
    check(any("不足" in x for x in validate_schedule([a], 10, 20)), "學分不足被抓")
    check(any("超標" in x for x in validate_schedule([a, c], 0, 5)), "學分超標被抓")
    # 避開星期
    check(any("避開" in x for x in validate_schedule([a, c], 0, 99, avoid_weekdays=["五"])), "避開週五被抓")


def test_solver() -> None:
    print("[find_schedules]")
    a = make_course("1", "課A", 3, "三234")
    b = make_course("2", "課B", 3, "三34五56")  # 與 A 衝
    c = make_course("3", "課C", 3, "五78")
    d = make_course("4", "課D", 2, "一12")
    pool = [a, b, c, d]

    schedules = find_schedules(pool, min_credits=5, max_credits=8, max_results=20)
    check(len(schedules) > 0, f"產出 {len(schedules)} 組合法課表")
    # 每組都必須通過 validator(solver 的剪枝結果不能自打嘴巴)
    all_valid = all(validate_schedule(s, 5, 8) == [] for s in schedules)
    check(all_valid, "所有 solver 產出的課表都通過 validator")
    # A 和 B 永遠不該同時出現(衝堂)
    no_ab = all(not ({"1", "2"} <= {x.course_id for x in s}) for s in schedules)
    check(no_ab, "衝堂的 A、B 從未被排進同一張課表")

    # 避開週五:含 C(五78) 或 B(..五56) 的組合都該被排除
    avoid_fri = find_schedules(pool, 0, 99, avoid_weekdays=["五"], max_results=20)
    no_fri = all(all("五" not in x.weekdays for x in s) for s in avoid_fri)
    check(no_fri, "避開週五時,沒有任何課落在週五")


def test_exclude_undefined_and_dedup() -> None:
    print("[排除未定時間 + 同名去重]")
    mgmt1 = make_course("1", "管理學", 3, "三234")
    mgmt2 = make_course("2", "管理學", 3, "四D56")  # 同名不同班
    flex = make_course("3", "自主學習", 3, "未定或彈性")  # 無時段
    econ = make_course("4", "經濟學", 3, "二56")
    pool = [mgmt1, mgmt2, flex, econ]

    scheds = find_schedules(pool, min_credits=3, max_credits=9, max_results=20)
    # 未定時間的「自主學習」不該出現在任何方案
    no_flex = all(all(c.name != "自主學習" for c in s) for s in scheds)
    check(no_flex, "時間未定的課被排除在所有方案外")
    # 同一方案內「管理學」最多一門
    no_dup = all([c.name for c in s].count("管理學") <= 1 for s in scheds)
    check(no_dup, "同名「管理學」一張課表最多一門")

    # validator 也要抓得到這兩種違規
    v_dup = validate_schedule([mgmt1, mgmt2], 0, 99)
    check(any("重複修課" in x for x in v_dup), "validator 抓到重複修課")
    v_flex = validate_schedule([flex], 0, 99)
    check(any("未定" in x for x in v_flex), "validator 抓到未定時段")


def test_on_real_db() -> None:
    print("[real data.db smoke test]")
    try:
        conn = sqlite3.connect("data.db")
        cur = conn.cursor()
        cur.execute(
            "SELECT id,name,point,time FROM COURSE WHERE time != '' AND time != '未定或彈性' LIMIT 12"
        )
        rows = cur.fetchall()
        conn.close()
    except sqlite3.Error as e:
        check(False, f"讀 data.db 失敗:{e}")
        return

    courses: list[CourseSlot] = [make_course(r[0], r[1], r[2], r[3]) for r in rows]
    parsed_ok = sum(1 for c in courses if c.slots)
    check(parsed_ok == len(courses), f"{len(courses)} 門真實課程時間全部解析出 slot")
    check(all(len(c.course_id) == 13 for c in courses), "真實 course_id 均為 13 位")
    # 用真實資料跑一次 solver,確認不會炸且結果合法
    sched = find_schedules(courses, min_credits=2, max_credits=10, max_results=3)
    print(f"    (真實資料排出 {len(sched)} 組,範例學分:"
          f"{[round(sum(x.credits for x in s),1) for s in sched]})")
    check(all(validate_schedule(s, 2, 10) == [] for s in sched), "真實資料排出的課表均合法")


if __name__ == "__main__":
    test_parse_slots()
    test_conflict_and_validate()
    test_solver()
    test_exclude_undefined_and_dedup()
    test_on_real_db()
    print(f"\n結果:{_PASS} passed, {_FAIL} failed")
    raise SystemExit(1 if _FAIL else 0)
