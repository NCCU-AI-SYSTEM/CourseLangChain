"""課程詳細資料卡片(tools/course_info.py 與 GET /api/courses/{id})的測試。

跑:`uv run python -m tests.test_course_info`(在 CourseLangChain/ 下)

驗三件事:
1. 課號格式:4 碼學期 + 9 碼科目代號,科目代號可含大寫字母。
2. 資料列 → 卡片 JSON:欄位一定齊全、空字串變 None、不是網址的連結不給。
3. API 的錯誤處理:格式錯 400、找不到 404、資料庫掛了 503,小寫課號會轉大寫。
   這部分把資料庫查詢換成假函式,不需要資料庫。

最後一段需要真實資料庫,連不上時會自行略過。
"""
from __future__ import annotations

import sys

from tools import course_info
from tools.course_info import COURSE_ID, FIELDS, to_card

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


def test_course_id() -> None:
    print("[課號格式]")
    check(bool(COURSE_ID.match("1151000711002")), "純數字的 13 碼通過")
    check(bool(COURSE_ID.match("1151044M01001")), "科目代號含大寫字母也通過")
    check(not COURSE_ID.match("115100071100"), "12 碼不通過")
    check(not COURSE_ID.match("1151044m01001"), "小寫字母不通過(API 會先轉大寫)")
    check(not COURSE_ID.match("1151000711002; DROP"), "夾帶其他字元不通過")


def test_to_card() -> None:
    print("[資料列 → 卡片]")
    card = to_card({
        "id": "1151000711002", "name": " 統計學 ", "teacher": "王小明",
        "objective": "", "point": 3.0,
        "teaschmurl": "https://newdoc.nccu.edu.tw/x", "teaexpurl": "javascript:alert(1)",
    })
    check(set(card) == set(FIELDS.values()), "每個 API 欄位都存在,資料庫沒有的欄位也在")
    check(card["name"] == "統計學", "字串前後空白去掉")
    check(card["objective"] is None, "空字串變成 None")
    check(card["syllabus"] is None, "資料庫沒有的欄位給 None")
    check(card["credits"] == 3.0, "數字原樣保留")
    check(card["syllabus_url"] == "https://newdoc.nccu.edu.tw/x", "https 連結保留")
    check(card["teacher_url"] is None, "不是網址的連結不給")


def test_api() -> None:
    print("[API 錯誤處理](資料庫查詢換成假函式)")
    from fastapi.testclient import TestClient

    import app as app_module

    client = TestClient(app_module.app)  # 不用 with:不觸發 lifespan,也就不需要資料庫
    original = course_info.fetch_card
    calls: list[str] = []
    try:
        def fake(cid: str):
            calls.append(cid)
            return to_card({"id": cid, "name": "統計學"}) if cid.endswith("002") else None

        course_info.fetch_card = fake
        r = client.get("/api/courses/abc")
        check(r.status_code == 400 and "13 碼" in r.json()["detail"], "格式錯回 400 並說明格式")
        r = client.get("/api/courses/1151000711002")
        check(r.status_code == 200 and r.json()["name"] == "統計學", "找得到回 200 與卡片")
        r = client.get("/api/courses/1151000711009")
        check(r.status_code == 404, "找不到回 404")
        client.get("/api/courses/1151044m01002")
        check(calls[-1] == "1151044M01002", "小寫課號先轉成大寫再查")

        def broken(_cid: str):
            raise ConnectionError("db down")

        course_info.fetch_card = broken
        r = client.get("/api/courses/1151000711002")
        check(r.status_code == 503 and "暫時無法查詢" in r.json()["detail"],
              "資料庫連不上回 503,不丟出堆疊")
    finally:
        course_info.fetch_card = original


def test_real_db() -> None:
    print("[真實資料庫](連不上就略過)")
    import psycopg2

    from paths import DATABASE_URL

    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=3)
    except psycopg2.OperationalError:
        print("  - 略過:資料庫沒開")
        return
    with conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM public.course WHERE id ~ '[A-Z]' LIMIT 1")
        row = cur.fetchone()
    conn.close()
    if not row:
        print("  - 略過:找不到含字母的課號")
        return
    card = course_info.fetch_card(row[0])
    check(card is not None and card["course_id"] == row[0], f"查得到含字母的課號 {row[0]}")
    check(bool(card and card["name"]), "卡片有課名")


if __name__ == "__main__":
    for fn in (test_course_id, test_to_card, test_api, test_real_db):
        fn()
    print(f"\n結果:{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)
