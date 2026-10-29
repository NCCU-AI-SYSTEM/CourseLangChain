"""課程詳細資料卡片的結構化查詢:給前端用,不經 LLM。

course_detail_tool 回的是給模型讀的 Markdown;前端的課程卡片要的是欄位分開的 JSON,
而且同樣不該讓課號經過模型轉述。所以另寫一個只讀資料庫的函式,工具與 API 各用各的格式。

欄位只讀資料庫裡實際存在的。爬蟲的欄位會隨學期增加(學校新增大綱段落就多一欄),
pgloader 匯入時欄名一律轉成小寫(teaSchmUrl → teaschmurl);寫死欄位清單的話,
少一欄整個查詢就會失敗。
"""
from __future__ import annotations

import re
import threading

# 4 碼學年學期 + 9 碼科目代號。科目代號可能含大寫字母(例:1151044M01001),
# 115-1 有 131 門課是這樣,只收純數字會把它們擋在門外。
COURSE_ID = re.compile(r"^\d{4}[0-9A-Z]{9}$")

# 資料庫欄位 → API 欄位。順序就是 SELECT 的順序。
FIELDS = {
    "id": "course_id",
    "name": "name",
    "nameen": "name_en",
    "teacher": "teacher",
    "time_raw": "time",
    "classroom": "classroom",
    "unit": "unit",
    "point": "credits",
    "lang": "language",
    "objective": "objective",
    "syllabus": "syllabus",
    "schedule": "schedule",
    "evaluation": "evaluation",
    "textbook": "textbook",
    "teaching_approach": "teaching_approach",
    "ai_policy": "ai_policy",
    "info": "info",
    "note": "note",
    "teaschmurl": "syllabus_url",
    "teaexpurl": "teacher_url",
}
LINK_FIELDS = ("syllabus_url", "teacher_url")

_columns: set[str] | None = None
_columns_lock = threading.Lock()


def to_card(row: dict) -> dict:
    """資料庫的一列 → 卡片 JSON。純函數。

    - 每個 API 欄位一定存在;資料庫沒有該欄、或值是空字串,一律給 None,
      前端只需要判斷「有沒有值」。
    - 連結只接受 http(s)。不是網址的值不給,避免前端產生壞連結或 javascript: 這類連結。
    """
    card: dict = {}
    for column, key in FIELDS.items():
        value = row.get(column)
        if isinstance(value, str):
            value = value.strip() or None
        card[key] = value
    for key in LINK_FIELDS:
        url = card[key]
        if url is not None and not str(url).startswith(("https://", "http://")):
            card[key] = None
    return card


def _available_columns(cur) -> set[str]:
    global _columns
    if _columns is None:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'course'"
        )
        found = {row[0] for row in cur.fetchall()}
        with _columns_lock:
            _columns = found
    return _columns


def fetch_card(course_id: str) -> dict | None:
    """以 13 碼課號查一門課;找不到回 None。只走 PostgreSQL。

    course 表裡一門課會因為分屬多個類別而有好幾列,內容相同,所以取一列即可。
    """
    import psycopg2
    import psycopg2.extras

    from paths import DATABASE_URL

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            available = _available_columns(cur)
        # 欄名只會來自 FIELDS 這份寫死的清單,與資料庫實際欄位取交集,不會有注入問題
        columns = [c for c in FIELDS if c in available]
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT {', '.join(columns)} FROM public.course WHERE id = %s LIMIT 1",
                (course_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    return to_card(dict(row)) if row else None
