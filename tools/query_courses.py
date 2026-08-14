"""query_courses_tool —— 檢索課程候選,回傳含 13 位 course_id 的清單。

2026-05-25 合併自舊 `retrieval_tool`:查課與排課共用這一個檢索 tool。支援兩種模式:
- **語意/關鍵字檢索**:用 vectorstore.pkl(FAISS+BM25 ensemble,或無 GPU 時純 BM25)
- **sql_filter 時間過濾**:先用 SQL WHERE 篩 COURSE,再對結果跑 BM25

輸出一律含 course_id + 學分(superset),排課時把 course_id 抄給 schedule_tool。
"""
from __future__ import annotations

import pickle
import re
import sqlite3

from langchain_classic.retrievers.bm25 import BM25Retriever
from langchain_core.documents import Document
from langchain_core.tools import tool

from paths import COURSE_SEMESTER, COURSE_YEAR, DATA_DB, VECTORSTORE_PKL
from utils.zh_tokenize import tokenize

from .registry import register_tool

PICKLE_FILE = VECTORSTORE_PKL

_retriever = None  # 進程內快取,避免每次呼叫都重載 2GB+ 的 pickle


def _get_retriever(pickle_file: str = PICKLE_FILE):
    global _retriever
    if _retriever is None:
        with open(pickle_file, "rb") as f:
            _retriever = pickle.load(f)
    return _retriever


def _dict_factory(cursor, row):
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def _format_docs(docs, top_k: int) -> str:
    """把檢索到的 docs(metadata=完整課程 row)格式化成含 course_id 的清單。"""
    lines: list[str] = []
    for i, doc in enumerate(docs[:top_k], 1):
        m = doc.metadata
        lines.append(
            f"{i}. {m.get('name', 'N/A')}"
            f"｜course_id: {m.get('id', 'N/A')}"
            f"｜時間: {m.get('time', 'N/A')}"
            f"｜老師: {m.get('teacher', 'N/A')}"
            f"｜學分: {m.get('point', 'N/A')}"
        )
    return "\n".join(lines)


# 與 _format_docs 成對的反向解析。main.py 的 astream 攔到 on_tool_end 後用它把
# 「給 LLM 讀的文字」還原成結構化候選,經 SSE 側通道送前端畫「加入課表」按鈕。
# 刻意不改 _format_docs 的回傳型別(tool 契約是回 str);兩者相鄰擺放,格式一改就會一起看到。
_LINE_RE = re.compile(
    r"^\s*\d+\.\s*(?P<name>.*?)"
    r"｜course_id:\s*(?P<course_id>\S+?)"
    r"｜時間:\s*(?P<time>.*?)"
    r"｜老師:\s*(?P<teacher>.*?)"
    r"｜學分:\s*(?P<credits>.*?)\s*$"
)


def parse_formatted_docs(text: str) -> list[dict]:
    """把 _format_docs 的輸出解析回結構化課程清單;解析不到的行直接略過。

    只收「13 碼純數字」的 course_id:工具也可能回「找不到符合條件的課程。」或
    「ERROR: ...」,那些行對不上格式就自然被濾掉,不會送出半成品給前端。
    """
    courses: list[dict] = []
    for line in (text or "").splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        course_id = m.group("course_id")
        if not (course_id.isdigit() and len(course_id) == 13):
            continue
        courses.append(
            {
                "course_id": course_id,
                "name": m.group("name").strip(),
                "time": m.group("time").strip(),
                "teacher": m.group("teacher").strip(),
                "credits": m.group("credits").strip(),
            }
        )
    return courses


def _bm25_over_sql(keyword: str, sql_filter: str, top_k: int) -> list:
    """先用 SQL WHERE 篩 COURSE,再對篩出的課程跑 BM25,回傳 docs。

    一律鎖在 build.py 建索引時的同一學期(COURSE_YEAR/COURSE_SEMESTER),否則會撈到
    別的學年(data.db 跨學年共 10 萬+ 筆,但 pickle 只含單一學期)。
    """
    conn = sqlite3.connect(DATA_DB)
    conn.row_factory = _dict_factory
    sem = (COURSE_YEAR, COURSE_SEMESTER)
    try:
        try:
            rows = conn.execute(
                f"SELECT * FROM COURSE WHERE y = ? AND s = ? AND ({sql_filter})", sem
            ).fetchall()
        except sqlite3.Error:
            # filter 無效 → 退回該學期全部課程(仍鎖學期,不掃全表)
            rows = conn.execute(
                "SELECT * FROM COURSE WHERE y = ? AND s = ?", sem
            ).fetchall()
    finally:
        conn.close()
    if not rows:
        return []
    docs = [
        Document(
            page_content=f"課程名稱是{r.get('name', '')}, 上課時間是{r.get('time', '')}, "
            f"這堂課的老師是{r.get('teacher', '')}",
            metadata=r,
        )
        for r in rows
    ]
    # preprocess_func=tokenize:中文斷詞,否則 BM25 對中文整句一個 token,關鍵字會失效
    bm25 = BM25Retriever.from_documents(docs, preprocess_func=tokenize)
    bm25.k = min(top_k, len(docs))
    return bm25.invoke(keyword)


@tool
def query_courses_tool(keyword: str, top_k: int = 10, sql_filter: str = "") -> str:
    """檢索符合條件的課程候選清單(查課與排課都用這個)。

    參數:
    - keyword: 想找的課程描述(如「人工智慧」「資料庫」)
    - top_k: 最多回傳幾門候選(查課用預設即可;排課建議 20)
    - sql_filter: 可選的 SQL WHERE 子句(由 text_to_sql_tool 產生),用來過濾時段;沒有就留空

    回傳:編號候選清單,每門含 **course_id(13 位)**、課名、時間、老師、學分。
    要排課時,把這些 course_id 用逗號接起來傳給 schedule_tool。
    """
    try:
        if sql_filter:
            docs = _bm25_over_sql(keyword, sql_filter, top_k)
        else:
            docs = _get_retriever().invoke(keyword)
        if not docs:
            return "找不到符合條件的課程。"
        return _format_docs(docs, top_k)
    except FileNotFoundError:
        return f"ERROR: 找不到向量庫 {PICKLE_FILE},請先執行 build.py 建立索引。"
    except sqlite3.Error as e:
        return f"ERROR: 課程資料庫無法查詢({e})。請確認 data.db 已就緒。"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: 檢索時發生錯誤:{e}"


register_tool(query_courses_tool)
