"""schedule_tool —— 接候選 course_id,跑完整排課 pipeline 回課表(段 2)。

對 agent 的契約(扁平參數、永不 raise、失敗回 "ERROR:"):
    schedule_tool(course_ids, min_credits, max_credits, avoid_weekdays, max_results)

pipeline:fetch_courses(DB I/O)→ find_schedules(solver)→ validate(不可跳過)→ rank → format_markdown
solver / validator / ranker / formatter 都在 tools/scheduler.py(純函數,零 LLM)。
本檔只做 I/O 與編排。
"""
from __future__ import annotations

import sqlite3

from langchain_core.tools import tool

from paths import DATA_DB, DATABASE_URL, USE_SQLITE
from tools.scheduler import (
    CourseSlot,
    find_schedules,
    format_schedules_markdown,
    make_course,
    parse_course_slots,
    rank_schedules,
    validate_schedule,
)

from .registry import register_tool

DB_PATH = DATA_DB


def _split_ids(course_ids: str) -> list[str]:
    """把逗號 / 空白 / 換行分隔的 id 字串切成乾淨清單(去重、保序)。"""
    raw = course_ids.replace("\n", ",").replace(" ", ",").replace("、", ",")
    seen: set[str] = set()
    out: list[str] = []
    for piece in raw.split(","):
        cid = piece.strip()
        if cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def _fetch_courses(ids: list[str], db_path: str = DB_PATH) -> tuple[list[CourseSlot], list[str]]:
    """依 id 從 COURSE 撈課,回 (CourseSlot 清單, 找不到的 id 清單)。"""
    if not ids:
        return [], []
    if USE_SQLITE:
        return _fetch_courses_sqlite(ids, db_path)
    return _fetch_courses_pg(ids)


def _fetch_courses_sqlite(ids: list[str], db_path: str) -> tuple[list[CourseSlot], list[str]]:
    placeholders = ",".join("?" for _ in ids)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            f"SELECT id, name, point, time, teacher FROM COURSE WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
    finally:
        conn.close()

    found = {str(r[0]): r for r in rows}
    courses = [
        make_course(r[0], r[1], r[2], r[3], r[4]) for r in (found[i] for i in ids if i in found)
    ]
    missing = [i for i in ids if i not in found]
    return courses, missing


def _fetch_courses_pg(ids: list[str]) -> tuple[list[CourseSlot], list[str]]:
    import psycopg2

    placeholders = ",".join("%s" for _ in ids)
    conn = psycopg2.connect(DATABASE_URL)
    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT id, name, point, time_raw AS time, teacher FROM public.course WHERE id IN ({placeholders})",
            ids,
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    found = {str(r[0]): r for r in rows}
    courses = [
        make_course(r[0], r[1], r[2], r[3], r[4]) for r in (found[i] for i in ids if i in found)
    ]
    missing = [i for i in ids if i not in found]
    return courses, missing


@tool
def schedule_tool(
    course_ids: str,
    min_credits: float = 0,
    max_credits: float = 99,
    avoid_weekdays: str = "",
    max_results: int = 3,
) -> str:
    """當已有一批候選課程 id、要從中排出無衝堂且學分達標的課表時呼叫此工具。

    參數:
    - course_ids: 候選課程的 13 位 id,用逗號分隔(來自 retrieve_tool 的結果)
    - min_credits / max_credits: 學分下限 / 上限
    - avoid_weekdays: 要避開的星期,中文字逗號分隔(如 "五" 或 "一,五"),沒有就留空
    - max_results: 最多回傳幾組方案(預設 3)

    回傳:Markdown 排課方案(每組含課表與學分);排不出時回說明文字;輸入有誤回 "ERROR:..."。
    """
    try:
        ids = _split_ids(course_ids)
        if not ids:
            return "ERROR: course_ids 為空,請先用 retrieve_tool 取得候選課程 id。"
        if min_credits > max_credits:
            return f"ERROR: 學分下限 {min_credits} 大於上限 {max_credits}。"

        courses, missing = _fetch_courses(ids)
        if not courses:
            return f"ERROR: 提供的 course_id 在資料庫中都找不到:{', '.join(ids)}"

        avoid = [w for w in avoid_weekdays.replace("、", ",").split(",") if w.strip()]

        schedules = find_schedules(
            courses,
            min_credits=min_credits,
            max_credits=max_credits,
            avoid_weekdays=avoid,
            max_results=max_results,
        )
        if not schedules:
            hint = (
                f"在這 {len(courses)} 門候選課中,排不出學分介於 "
                f"{min_credits}~{max_credits} 且無衝堂的組合。"
            )
            if avoid:
                hint += f"(已排除星期 {''.join(avoid)})"
            hint += " 可放寬學分範圍或用 retrieve_tool 增加候選課程。"
            return hint

        # validate 不可跳過:剪枝過也要再驗一次,擋掉任何漏網違規
        valid = [s for s in schedules if not validate_schedule(s, min_credits, max_credits, avoid)]
        if not valid:
            return "ERROR: 排序前驗證未通過(solver 與 validator 結果不一致),請回報此情況。"

        ranked = rank_schedules(valid)
        result = format_schedules_markdown(ranked)
        if missing:
            result += f"\n\n> 註:以下 id 在資料庫找不到,已略過:{', '.join(missing)}"
        return result
    except Exception as e:  # noqa: BLE001 — tool 契約要求回字串而非 raise
        return f"ERROR: 排課時發生未預期錯誤:{e}"


register_tool(schedule_tool)


# ---- L3 輸出驗證:防 agent 竄改/幻覺出衝堂的課表 ----
def _validate_schedule_output(output: str) -> tuple[bool, str]:
    """檢查 agent 最終輸出中每個「### 方案」區塊是否有衝堂。

    保守策略:只看 `### 方案` 標題下的表格列,逐區塊做兩兩 slot 交集;
    無法解析時放行(fail-open),只有「確實偵測到重疊」才擋下 —— 因為權威檢查
    已在 schedule_tool 內做過,這層只是防止輸出被改壞,不該誤殺正常回覆。
    """
    if "### 方案" not in output:
        return True, ""  # 不是排課輸出,跳過

    blocks: list[list[tuple[str, frozenset]]] = []
    current: list[tuple[str, frozenset]] = []
    for line in output.splitlines():
        if line.strip().startswith("### 方案"):
            if current:
                blocks.append(current)
            current = []
            continue
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0] in ("課程名稱", "") or set(cells[1]) <= {"-"}:
            continue  # 表頭 / 分隔線
        name, time_str = cells[0], cells[1]
        current.append((name, parse_course_slots(time_str)))
    if current:
        blocks.append(current)

    for blk in blocks:
        for i in range(len(blk)):
            for j in range(i + 1, len(blk)):
                overlap = blk[i][1] & blk[j][1]
                if overlap:
                    slot_str = ", ".join(f"{wd}{p}" for wd, p in sorted(overlap))
                    return False, f"輸出課表衝堂:「{blk[i][0]}」與「{blk[j][0]}」於 {slot_str}"
    return True, ""


try:  # harness 可能在某些精簡環境未載入,validator 缺席不應讓 import 失敗
    from harness import register_validator

    register_validator("schedule_no_conflict", _validate_schedule_output)
except Exception:  # noqa: BLE001
    pass
