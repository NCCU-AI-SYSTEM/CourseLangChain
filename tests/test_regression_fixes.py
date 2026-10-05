"""2026-06 這波修正的回歸測試,釘住四個 bug 不再復發。

跑:`.venv/bin/python -m tests.test_regression_fixes`(在 CourseLangChain/ 下)
沿用專案既有的 standalone 風格(自帶 check + 報表,不依賴 pytest)。
碰 data.db 的測試在資料庫缺席時會跳過(SKIP)而非 FAIL。

對應修正:
1. utils/time.getSessionArray —— 未知字元不再 crash、無幻影節次
2. course_detail_tool —— 只給課名時鎖當前學期(不撈歷年同名課)
3. query_courses_tool(sql_filter 路徑)—— 鎖學期 + jieba 斷詞讓關鍵字真正生效
4. L3 _validate_schedule_output —— 時間欄含逗號/空白不再 crash 而誤拒
"""
from __future__ import annotations

import os
import re
import sqlite3

from paths import COURSE_SEMESTER, COURSE_YEAR, DATA_DB

_PASS = 0
_FAIL = 0
_SKIP = 0

# 當前學期的 course_id 前綴(年3碼 + 學期1碼),例如 114 + 2 -> "1142"
SEM_PREFIX = f"{COURSE_YEAR}{COURSE_SEMESTER}"


def check(cond: bool, msg: str) -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  ✓ {msg}")
    else:
        _FAIL += 1
        print(f"  ✗ FAIL: {msg}")


def skip(msg: str) -> None:
    global _SKIP
    _SKIP += 1
    print(f"  ⊘ SKIP: {msg}")


def _has_db() -> bool:
    return os.path.exists(DATA_DB)


# ---------------------------------------------------------------------------
# Bug 1:時間解析強健化(utils/time.getSessionArray 經 parse_course_slots)
# ---------------------------------------------------------------------------
def test_time_parse_robust() -> None:
    print("[Bug1 時間解析:未知字元/幻影節次]")
    from tools.scheduler import parse_course_slots

    # 正常資料必須維持原解析(回歸防呆)
    check(
        parse_course_slots("三234") == {("三", "2"), ("三", "3"), ("三", "4")},
        "三234 正常解析",
    )
    check(
        parse_course_slots("四CD") == {("四", "C"), ("四", "D")},
        "字母節次 四CD 正常解析",
    )
    multi = parse_course_slots("三234五56")
    check(
        ("三", "2") in multi and ("五", "5") in multi and ("五", "6") in multi,
        "多日 三234五56 兩天都解析",
    )

    # 未知字元(逗號/空白)不可 crash;結果應等同把分隔字元忽略
    try:
        comma = parse_course_slots("三234,五56")
        crashed = False
    except Exception:  # noqa: BLE001
        crashed = True
    check(not crashed, "含逗號不再 crash")
    check(comma == multi, "逗號被當分隔忽略,結果與無逗號相同")

    # 幻影節次:只有星期 / 結尾殘留星期,不可生出不存在的 'A' slot
    check(parse_course_slots("三") == set(), "只有星期『三』→ 無 slot(無幻影)")
    check(
        parse_course_slots("三234五") == {("三", "2"), ("三", "3"), ("三", "4")},
        "結尾殘留星期『五』不產生幻影 五A",
    )
    check(parse_course_slots("未定或彈性") == set(), "未定或彈性 → 無 slot")
    check(parse_course_slots("") == set(), "空字串 → 無 slot")


# ---------------------------------------------------------------------------
# Bug 2:course_detail 只給課名時鎖當前學期
# ---------------------------------------------------------------------------
def test_course_detail_semester_lock() -> None:
    print("[Bug2 course_detail 鎖學期]")
    if not _has_db():
        skip("找不到 data.db,略過")
        return
    from tools.course_detail import course_detail_tool

    # 「管理學」歷年都有開,只給課名時不可撈到非當前學期的課
    out = course_detail_tool.invoke({"course_name": "管理學"})
    ids = re.findall(r"id=(\d{13})", out) + re.findall(r"課程編號：(\d{13})", out)
    if not ids:
        skip("當前學期查無『管理學』候選(資料可能非 114/2),略過")
        return
    check(
        all(i.startswith(SEM_PREFIX) for i in ids),
        f"所有回傳 course_id 都屬當前學期({SEM_PREFIX}…),共 {len(ids)} 筆",
    )
    check(
        not any(i.startswith("1011") for i in ids),
        "不再出現 101 學年等歷年同名課",
    )


# ---------------------------------------------------------------------------
# Bug 3:query_courses sql_filter 路徑 —— 鎖學期 + jieba 讓關鍵字生效
# ---------------------------------------------------------------------------
def test_query_courses_sql_filter() -> None:
    print("[Bug3 query_courses:鎖學期 + 中文斷詞]")
    if not _has_db():
        skip("找不到 data.db,略過")
        return
    from tools.query_courses import query_courses_tool

    flt = "time GLOB '*三*'"  # 只跑 sql_filter 路徑(純 BM25,不載 pickle)

    def ids_for(kw: str) -> list[str]:
        out = query_courses_tool.invoke(
            {"keyword": kw, "sql_filter": flt, "top_k": 5}
        )
        return re.findall(r"course_id: (\d{13})", out)

    mgmt = ids_for("管理")
    quantum = ids_for("量子")
    if not mgmt or not quantum:
        skip("sql_filter 路徑查無結果(資料可能非 114/2),略過")
        return

    # 鎖學期:全部屬當前學期
    check(
        all(i.startswith(SEM_PREFIX) for i in mgmt + quantum),
        f"sql_filter 結果全屬當前學期({SEM_PREFIX}…)",
    )
    # jieba 生效:不同關鍵字必須給出不同結果(修正前三個關鍵字回傳完全相同)
    check(
        set(mgmt) != set(quantum),
        "不同關鍵字回傳不同課程(關鍵字真的有作用,非被 BM25 忽略)",
    )


# ---------------------------------------------------------------------------
# Bug 4:L3 驗證器吃到帶逗號/空白的時間欄不再 crash 而誤拒
# ---------------------------------------------------------------------------
def test_l3_messy_time_cell() -> None:
    print("[Bug4 L3 時間欄含逗號/空白不誤殺]")
    from tools.schedule_tool import _validate_schedule_output

    # 時間欄帶逗號(模型常見輸出),兩門不衝堂 → 應 crash-free 並放行
    messy = (
        "### 方案 1(共 6.0 學分,上課日:三五)\n"
        "| 課程名稱 | 上課時間 | 授課老師 | 學分 |\n"
        "|----------|----------|----------|------|\n"
        "| 課A | 三234, 五56 | 王 | 3.0 |\n"
        "| 課B | 二34 | 李 | 3.0 |\n"
    )
    try:
        ok, msg = _validate_schedule_output(messy)
        crashed = False
    except Exception:  # noqa: BLE001
        crashed = True
        ok, msg = False, ""
    check(not crashed, "時間欄含逗號不再讓 validator crash")
    check(ok, "不衝堂的課表(即使時間欄有逗號)放行,不誤拒")

    # 仍要能抓到真正衝堂(逗號不影響交集判斷)
    conflict = (
        "### 方案 1\n| 課程名稱 | 上課時間 | 授課老師 | 學分 |\n"
        "|--|--|--|--|\n| 課A | 三234, 五56 | 王 | 3 |\n| 課B | 五56 | 李 | 3 |\n"
    )
    okc, _ = _validate_schedule_output(conflict)
    check(not okc, "帶逗號的時間欄仍能正確抓到衝堂(五56)")


if __name__ == "__main__":
    test_time_parse_robust()
    test_course_detail_semester_lock()
    test_query_courses_sql_filter()
    test_l3_messy_time_cell()
    print(f"\n結果:{_PASS} passed, {_FAIL} failed, {_SKIP} skipped")
    raise SystemExit(1 if _FAIL else 0)
