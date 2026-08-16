import sqlite3
from typing import Optional

from langchain_core.tools import tool

from warnings import deprecated

from paths import (
    COURSE_SEMESTER,
    COURSE_YEAR,
    DATA_DB,
    DATABASE_URL,
    SQLITE_DEPRECATION_MSG,
    USE_SQLITE,
)

_FIELD_TRUNCATE = 800


def _truncate(text: Optional[str], limit: int = _FIELD_TRUNCATE) -> str:
    if not text:
        return "(無)"
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…(已截斷)"


def _format_one(course: dict) -> str:
    point = course.get("point")
    point_str = f"{point} 學分" if point is not None else "(無)"
    return "\n".join(
        [ 
            "## 課程資訊",
            f"- 課程名稱：{course.get('name') or '(無)'}",
            f"- 課程編號：{course.get('id') or '(無)'}",
            f"- 授課老師：{course.get('teacher') or '(無)'}",
            f"- 上課時間：{course.get('time') or '(無)'}",
            f"- 上課教室：{course.get('classroom') or '(無)'}",
            f"- 開課單位：{course.get('unit') or '(無)'}",
            f"- 學分：{point_str}",
            "",
            "## 課程目標",
            _truncate(course.get("objective")),
            "",
            "## 教學內容",
            _truncate(course.get("syllabus")),
            "",
            "## 課程進度",
            _truncate(course.get("schedule")),
            "",
            "## 評分方式",
            _truncate(course.get("evaluation")),
            "",
            "## 教科書",
            _truncate(course.get("textbook")),
            "",
            "## 教學方式",
            _truncate(course.get("teaching_approach")),
            "",
            "## AI 政策",
            _truncate(course.get("ai_policy")),
            "",
            "## 備註",
            _truncate(course.get("note")),
        ]
    )


def _format_candidates(rows: list[dict]) -> str:
    lines = ["找到多筆同名課程,請使用者選擇後再次查詢(把 course_id 一起傳入):"]
    for r in rows:
        lines.append(
            f"- id={r.get('id')}｜{r.get('name')}｜老師：{r.get('teacher')}｜時間：{r.get('time')}"
        )
    return "\n".join(lines)


@tool
def course_detail_tool(
    course_name: str,
    course_id: Optional[str] = None,
    db_path: str = DATA_DB,
) -> str:
    """查詢「單一門」課的詳細資料(教學內容、課程目標、評分方式、教科書、上課方式、備註等)。

    用途:當使用者想知道某門課的「詳細資料」「課綱」「評分方式」「教科書」「上課方式」「教學內容」時呼叫。
    不適用於列出多門候選課程(那是 query_courses_tool 的工作)。

    參數:
    - course_name: 課程名稱,可模糊比對(例如「機器學習」)
    - course_id: 課程編號(優先使用)。若上一輪 query_courses_tool 結果中已知,務必一併帶入,以免比對到多筆同名課。

    回傳:
    - 找到 1 筆 → Markdown 區塊化的詳細資料(課程資訊 / 教學內容 / 評分方式 ...)
    - 找到多筆 → 候選清單,請使用者澄清是哪一門
    - 找不到   → 提示找不到
    """
    columns = [
        "id",
        "name",
        "teacher",
        "time",
        "classroom",
        "unit",
        "point",
        "syllabus",
        "objective",
        "schedule",
        "evaluation",
        "textbook",
        "teaching_approach",
        "ai_policy",
        "note",
    ]
    select_cols = ", ".join(columns)

    if not USE_SQLITE:
        return _fetch_pg(select_cols, course_name, course_id)
    return _fetch_sqlite(select_cols, course_name, course_id, db_path)


@deprecated(SQLITE_DEPRECATION_MSG)
def _fetch_sqlite(
    select_cols: str, course_name: str, course_id: Optional[str], db_path: str
) -> str:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        if course_id:
            cur.execute(
                f"SELECT {select_cols} FROM COURSE WHERE id = ? LIMIT 1",
                (course_id,),
            )
            row = cur.fetchone()
            if row is None:
                return f"找不到 course_id={course_id} 的課程。"
            return _format_one(dict(row))

        # 只給課名時必須鎖當前學期,否則 name LIKE 會撈到歷年同名課(如 101 學年的「管理學」)
        cur.execute(
            f"""
            SELECT {select_cols} FROM COURSE
            WHERE name LIKE ? AND y = ? AND s = ?
            GROUP BY id
            LIMIT 10
            """,
            (f"%{course_name}%", COURSE_YEAR, COURSE_SEMESTER),
        )
        rows = [dict(r) for r in cur.fetchall()]
    except sqlite3.Error as e:
        # 契約:tool 失敗回字串而非 raise(否則整個 graph 會 crash)。
        # 最常見:db_path 指到空檔(no such table: COURSE)→ 資料庫沒掛好/路徑錯。
        return f"ERROR: 課程資料庫無法查詢({e})。請確認 data.db 已就緒。"
    finally:
        conn.close()

    if not rows:
        return f"找不到名稱包含「{course_name}」的課程。"
    if len(rows) == 1:
        return _format_one(rows[0])
    return _format_candidates(rows)


def _fetch_pg(select_cols: str, course_name: str, course_id: Optional[str]) -> str:
    import psycopg2
    import psycopg2.extras

    select_cols_pg = select_cols.replace("time", "time_raw AS time")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        if course_id:
            cur.execute(
                f"SELECT {select_cols_pg} FROM public.course WHERE id = %s LIMIT 1",
                (course_id,),
            )
            row = cur.fetchone()
            if row is None:
                return f"找不到 course_id={course_id} 的課程。"
            return _format_one(dict(row))
        cur.execute(
            f"SELECT {select_cols_pg} FROM public.course "
            f"WHERE name LIKE %s AND y = %s AND s = %s "
            f"LIMIT 10",
            (f"%{course_name}%", COURSE_YEAR, COURSE_SEMESTER),
        )
        rows = [dict(r) for r in cur.fetchall()]
    except Exception as e:
        return f"ERROR: 課程資料庫無法查詢({e})。"
    finally:
        cur.close()
        conn.close()

    if not rows:
        return f"找不到名稱包含「{course_name}」的課程。"
    if len(rows) == 1:
        return _format_one(rows[0])
    return _format_candidates(rows)
