"""每個 session 一份「已排定課表」,供聊天介面的課表面板與 agent 共用。

定位:這是**對話之外的第二份記憶**。checkpointer 記的是對話訊息,這裡記的是
使用者實際敲定的課表——使用者可能在面板上手動增刪,那些動作不經過 LLM,
所以不能只靠對話歷史還原。

一致性原則(與 harness 同調):衝堂判斷、學分計算一律走 `scheduler.py` 的純函數,
不讓 LLM 自己算。面板改的和 agent 改的是同一份資料、同一套規則。

隱私:純記憶體、per-session,程序重啟即清空,與 in-memory checkpointer 一致。
"""
from __future__ import annotations

import sqlite3
import threading

from paths import COURSE_SEMESTER, COURSE_YEAR, DATA_DB
from tools.scheduler import CourseSlot, has_conflict, make_course

# session_id -> 已排定的 course_id 清單(保序)
_STORE: dict[str, list[str]] = {}
# 多個 request 可能同時改同一份課表(面板 + 聊天),用鎖保護
_LOCK = threading.Lock()

# 一份課表最多幾門,防止誤用把記憶體撐爆
MAX_COURSES = 30


def _fetch_course(course_id: str, db_path: str = DATA_DB) -> CourseSlot | None:
    """依 13 碼 id 取單門課;鎖當前學年學期,避免撈到歷年同名課。"""
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT id, name, point, time, teacher FROM COURSE "
            "WHERE id = ? AND y = ? AND s = ?",
            (str(course_id).strip(), COURSE_YEAR, COURSE_SEMESTER),
        ).fetchone()
    finally:
        conn.close()
    return make_course(row[0], row[1], row[2], row[3], row[4]) if row else None


def _fetch_many(ids: list[str], db_path: str = DATA_DB) -> list[CourseSlot]:
    """依 id 清單取課,回傳順序與 ids 一致(查不到的略過)。"""
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            f"SELECT id, name, point, time, teacher FROM COURSE "
            f"WHERE id IN ({placeholders}) AND y = ? AND s = ?",
            [*ids, COURSE_YEAR, COURSE_SEMESTER],
        ).fetchall()
    finally:
        conn.close()
    found = {str(r[0]): r for r in rows}
    return [make_course(*found[i]) for i in ids if i in found]


def get_schedule(session_id: str, db_path: str = DATA_DB) -> list[CourseSlot]:
    """取這個 session 目前已排定的課(依加入順序)。"""
    with _LOCK:
        ids = list(_STORE.get(session_id, []))
    return _fetch_many(ids, db_path)


def total_credits(courses: list[CourseSlot]) -> float:
    return round(sum(c.credits for c in courses), 1)


def add_course(session_id: str, course_id: str, db_path: str = DATA_DB) -> dict:
    """加一門課進課表;**與現有課衝堂就覆蓋掉衝突的那幾門**。

    Returns:
        {"ok": bool, "added": CourseSlot|None, "removed": [CourseSlot],
         "error": str|None, "already": bool}
    """
    course_id = str(course_id).strip()
    if not course_id:
        return {"ok": False, "added": None, "removed": [], "error": "缺少課程代碼", "already": False}

    course = _fetch_course(course_id, db_path)
    if course is None:
        return {
            "ok": False, "added": None, "removed": [], "already": False,
            "error": f"找不到課程代碼 {course_id}(本學期 {COURSE_YEAR}-{COURSE_SEMESTER} 無此課)",
        }

    with _LOCK:
        current_ids = list(_STORE.get(session_id, []))
        if course_id in current_ids:
            return {"ok": True, "added": course, "removed": [], "error": None, "already": True}
        if len(current_ids) >= MAX_COURSES:
            return {
                "ok": False, "added": None, "removed": [], "already": False,
                "error": f"課表已達上限 {MAX_COURSES} 門,請先移除部分課程",
            }

        existing = _fetch_many(current_ids, db_path)
        # 衝堂的舊課要讓位。時間未定的課 slots 為空,不會與任何課衝突,故不受影響。
        removed = [c for c in existing if has_conflict(c, course)]
        removed_ids = {c.course_id for c in removed}

        _STORE[session_id] = [i for i in current_ids if i not in removed_ids] + [course_id]

    return {"ok": True, "added": course, "removed": removed, "error": None, "already": False}


def remove_course(session_id: str, course_id: str) -> bool:
    """從課表移除一門課。回傳是否真的移除了。"""
    course_id = str(course_id).strip()
    with _LOCK:
        current = _STORE.get(session_id, [])
        if course_id not in current:
            return False
        _STORE[session_id] = [i for i in current if i != course_id]
        return True


def clear_schedule(session_id: str) -> None:
    with _LOCK:
        _STORE.pop(session_id, None)


def to_dicts(courses: list[CourseSlot]) -> list[dict]:
    """轉成前端面板用的 JSON 結構。"""
    return [
        {
            "course_id": c.course_id,
            "name": c.name,
            "credits": c.credits,
            "time": c.time_str,
            "teacher": c.teacher,
            # 面板畫週課表用:[["三","2"], ...]
            "slots": sorted([wd, p] for wd, p in c.slots),
        }
        for c in courses
    ]


def format_markdown(courses: list[CourseSlot]) -> str:
    """給 agent 讀的課表文字。"""
    if not courses:
        return "目前課表是空的(使用者還沒排定任何課程)。"
    lines = [
        f"目前已排定 {len(courses)} 門課,共 {total_credits(courses):g} 學分:",
        "",
        "| 課程代碼 | 課名 | 時間 | 教師 | 學分 |",
        "|---|---|---|---|---|",
    ]
    for c in courses:
        lines.append(
            f"| {c.course_id} | {c.name} | {c.time_str or '(時間未定)'} "
            f"| {c.teacher or '-'} | {c.credits:g} |"
        )
    return "\n".join(lines)
