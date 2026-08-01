"""session 課表(面板 + my_schedule_tool 共用狀態)測試。

跑:`.venv/bin/python -m tests.test_session_schedule`(在 CourseLangChain/ 下)

用**真實 data.db** 取課,但只讀不寫;課表本身是純記憶體,測完 clear 掉。
"""
from __future__ import annotations

import sqlite3

from paths import COURSE_SEMESTER, COURSE_YEAR, DATA_DB
from tools import session_schedule as store

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


def _pick_courses() -> tuple[str, str, str]:
    """從真實 DB 挑三門課:A 與 B 時間完全相同(必衝堂),C 與 A 不衝堂。"""
    conn = sqlite3.connect(DATA_DB)
    try:
        rows = conn.execute(
            "SELECT id, name, time FROM COURSE WHERE y=? AND s=? "
            "AND time IS NOT NULL AND time != '' LIMIT 4000",
            (COURSE_YEAR, COURSE_SEMESTER),
        ).fetchall()
    finally:
        conn.close()

    by_time: dict[str, list[str]] = {}
    for cid, _name, t in rows:
        by_time.setdefault(t, []).append(cid)

    # 找一組同時段的兩門課
    same = next(ids for ids in by_time.values() if len(ids) >= 2)
    a, b = same[0], same[1]
    a_time = next(t for t, ids in by_time.items() if ids[0] == a)

    # 找一門完全不共用星期的課,確保不衝堂
    a_weekdays = {ch for ch in a_time if ch in "一二三四五六日"}
    c = next(
        ids[0]
        for t, ids in by_time.items()
        if not ({ch for ch in t if ch in "一二三四五六日"} & a_weekdays)
    )
    return a, b, c


SID = "test-session-schedule"
A, B, C = _pick_courses()


def test_add_and_view() -> None:
    print("[加課 / 查看]")
    store.clear_schedule(SID)
    check(store.get_schedule(SID) == [], "初始課表為空")

    r = store.add_course(SID, A)
    check(r["ok"] and not r["already"], "加入第一門課成功")
    check(len(store.get_schedule(SID)) == 1, "課表有 1 門課")

    # 重複加同一門
    r = store.add_course(SID, A)
    check(r["already"], "重複加入同一門 → 標記 already")
    check(len(store.get_schedule(SID)) == 1, "重複加入不會變成 2 門")


def test_conflict_overwrites() -> None:
    print("[衝堂覆蓋]")
    store.clear_schedule(SID)
    store.add_course(SID, A)
    r = store.add_course(SID, B)  # B 與 A 同時段

    check(r["ok"], "加入衝堂課仍然成功(覆蓋語意)")
    check(len(r["removed"]) == 1, "回報有 1 門被移除")
    check(r["removed"][0].course_id == A, "被移除的正是衝堂的舊課")
    ids = [c.course_id for c in store.get_schedule(SID)]
    check(ids == [B], f"課表只剩新課(得到 {ids})")


def test_no_conflict_coexists() -> None:
    print("[不衝堂則並存]")
    store.clear_schedule(SID)
    store.add_course(SID, A)
    r = store.add_course(SID, C)
    check(not r["removed"], "不衝堂 → 沒有課被移除")
    check(len(store.get_schedule(SID)) == 2, "兩門課並存")


def test_remove_and_clear() -> None:
    print("[移除 / 清空]")
    store.clear_schedule(SID)
    store.add_course(SID, A)
    store.add_course(SID, C)

    check(store.remove_course(SID, A) is True, "移除存在的課回 True")
    check(store.remove_course(SID, A) is False, "再移除同一門回 False")
    check(len(store.get_schedule(SID)) == 1, "移除後剩 1 門")

    store.clear_schedule(SID)
    check(store.get_schedule(SID) == [], "清空後為空")


def test_session_isolation() -> None:
    print("[session 隔離]")
    store.clear_schedule(SID)
    store.clear_schedule(SID + "-other")
    store.add_course(SID, A)
    check(store.get_schedule(SID + "-other") == [], "另一個 session 看不到這份課表")
    store.clear_schedule(SID)


def test_bad_course_id() -> None:
    print("[錯誤課號]")
    store.clear_schedule(SID)
    r = store.add_course(SID, "9999999999999")
    check(not r["ok"] and r["error"], "不存在的課號回錯誤而非 crash")
    check(store.get_schedule(SID) == [], "失敗不會污染課表")

    r = store.add_course(SID, "")
    check(not r["ok"], "空課號回錯誤")


def test_tool_interface() -> None:
    print("[my_schedule_tool 介面]")
    from tools.my_schedule import my_schedule_tool

    cfg = {"configurable": {"thread_id": SID}}
    store.clear_schedule(SID)

    schema = my_schedule_tool.args_schema.model_json_schema().get("properties", {})
    check("config" not in schema, "config 不暴露給 LLM(session 由框架注入)")

    out = my_schedule_tool.invoke({"action": "view"}, config=cfg)
    check("空的" in out, "空課表回友善訊息")

    out = my_schedule_tool.invoke({"action": "add", "course_id": A}, config=cfg)
    check("已加入" in out, "加課成功訊息")

    out = my_schedule_tool.invoke({"action": "add", "course_id": B}, config=cfg)
    check("已移除" in out, "衝堂加課會說明移除了哪些課")

    out = my_schedule_tool.invoke({"action": "remove", "course_id": B}, config=cfg)
    check("已從課表移除" in out, "移除訊息")

    out = my_schedule_tool.invoke({"action": "bogus"}, config=cfg)
    check(out.startswith("ERROR"), "非法 action 回 ERROR 字串(不 raise)")

    out = my_schedule_tool.invoke({"action": "add"}, config=cfg)
    check(out.startswith("ERROR"), "add 缺 course_id 回 ERROR")

    # 沒有 session 時不該爆
    out = my_schedule_tool.invoke({"action": "view"}, config={"configurable": {}})
    check(out.startswith("ERROR"), "取不到 session 時回 ERROR 字串")

    store.clear_schedule(SID)


def test_credits_and_payload() -> None:
    print("[學分計算 / 前端 payload]")
    store.clear_schedule(SID)
    store.add_course(SID, A)
    store.add_course(SID, C)
    courses = store.get_schedule(SID)

    expected = round(sum(c.credits for c in courses), 1)
    check(store.total_credits(courses) == expected, "學分加總正確")

    payload = store.to_dicts(courses)
    check(len(payload) == 2, "payload 筆數正確")
    keys = set(payload[0])
    check(
        {"course_id", "name", "credits", "time", "teacher", "slots"} <= keys,
        f"payload 欄位齊全(得到 {sorted(keys)})",
    )
    check(all(isinstance(s, list) and len(s) == 2 for s in payload[0]["slots"]),
          "slots 是 [星期, 節次] 的清單(前端畫週課表用)")
    store.clear_schedule(SID)


if __name__ == "__main__":
    print(f"(測試用課程:A={A} B={B}(與A同時段) C={C}(與A不衝堂))\n")
    test_add_and_view()
    test_conflict_overwrites()
    test_no_conflict_coexists()
    test_remove_and_clear()
    test_session_isolation()
    test_bad_course_id()
    test_tool_interface()
    test_credits_and_payload()
    print(f"\n結果:{_PASS} passed, {_FAIL} failed")
    raise SystemExit(1 if _FAIL else 0)
