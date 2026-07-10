"""集中管理資料檔的絕對路徑,**不依賴程序的工作目錄(cwd)**。

問題背景:`data.db` / `vectorstore.pkl` 用相對路徑時,若 server / IDE 從非
`CourseLangChain/` 的目錄啟動,SQLite 會自動建一個空 db(→ `no such table: COURSE`)、
pickle 則 FileNotFoundError。這裡以「本檔所在目錄」為專案根錨定,徹底避開 cwd 問題。
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DB = os.path.join(PROJECT_ROOT, "data.db")
VECTORSTORE_PKL = os.path.join(PROJECT_ROOT, "vectorstore.pkl")

# 當前學期 —— 必須與 build.py 建索引時用的 y/s 一致(docker 預設 114 學年第 2 學期)。
# data.db 含跨學年共 10 萬+ 筆,但 vectorstore.pkl 只建單一學期;query_courses 的
# sql_filter 路徑會跳過 pickle 直接查 COURSE,故須用這組常數把查詢鎖回同一學期,
# 否則會撈到別的學年(course_id 開頭學期碼對不上,排課也會錯)。
COURSE_YEAR = os.getenv("COURSE_YEAR", "114")
COURSE_SEMESTER = os.getenv("COURSE_SEMESTER", "2")

# PostgreSQL hybrid search toggle
# true  = SQLite + vectorstore.pkl (original, no Docker needed)
# false = PostgreSQL + pgvector + pg_bm25 (run `docker compose up embedder` first)
USE_SQLITE = os.getenv("USE_SQLITE", "true").lower() == "true"

# PostgreSQL connection string (only used when USE_SQLITE=false)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/academic",
)

# 使用者自校務系統匯出的成績單 JSON(可選、含個資)。預設放專案根,
# 由 .gitignore 擋下不進 git;沒這個檔時個人化功能自動略過。
USER_RECORD_JSON = os.getenv(
    "USER_RECORD_JSON", os.path.join(PROJECT_ROOT, "user_record.json")
)
