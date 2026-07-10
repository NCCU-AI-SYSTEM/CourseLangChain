"""retrieve_tool — 檢索課程候選,回傳含 13 位 course_id 的清單。

支援兩種資料源:
- USE_SQLITE=true  (default): 從 data.db + vectorstore.pkl 檢索(沿用舊路徑)
- USE_SQLITE=false          : 從 PostgreSQL hybrid search 檢索(pgvector + pg_bm25 RRF)

兩者輸出格式完全相同,對 agent 無感。
"""
from __future__ import annotations

import os
import pickle
import sqlite3

from langchain_classic.retrievers.bm25 import BM25Retriever
from langchain_core.documents import Document
from langchain_core.tools import tool

from paths import COURSE_SEMESTER, COURSE_YEAR, DATA_DB, VECTORSTORE_PKL, USE_SQLITE
from utils.zh_tokenize import tokenize

from .registry import register_tool

PICKLE_FILE = VECTORSTORE_PKL

_retriever = None


def _get_retriever(pickle_file: str = PICKLE_FILE):
    global _retriever
    if _retriever is None:
        with open(pickle_file, "rb") as f:
            _retriever = pickle.load(f)
    return _retriever


def _dict_factory(cursor, row):
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def _format_docs(docs, top_k: int) -> str:
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


def _format_rows(rows, top_k: int) -> str:
    lines: list[str] = []
    for i, row in enumerate(rows[:top_k], 1):
        lines.append(
            f"{i}. {row.get('name', 'N/A')}"
            f"｜course_id: {row.get('id', 'N/A')}"
            f"｜時間: {row.get('time', 'N/A') or row.get('time_raw', 'N/A')}"
            f"｜老師: {row.get('teacher', 'N/A')}"
            f"｜學分: {row.get('point', 'N/A')}"
        )
    return "\n".join(lines)


# ── SQLite path (original) ────────────────────────────────────────────────────


def _bm25_over_sql_sqlite(keyword: str, sql_filter: str, top_k: int) -> list:
    conn = sqlite3.connect(DATA_DB)
    conn.row_factory = _dict_factory
    sem = (COURSE_YEAR, COURSE_SEMESTER)
    try:
        try:
            rows = conn.execute(
                f"SELECT * FROM COURSE WHERE y = ? AND s = ? AND ({sql_filter})", sem
            ).fetchall()
        except sqlite3.Error:
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
    bm25 = BM25Retriever.from_documents(docs, preprocess_func=tokenize)
    bm25.k = min(top_k, len(docs))
    return bm25.invoke(keyword)


def _sqlite_retrieve(keyword: str, top_k: int, sql_filter: str) -> str:
    if sql_filter:
        docs = _bm25_over_sql_sqlite(keyword, sql_filter, top_k)
    else:
        docs = _get_retriever().invoke(keyword)
    if not docs:
        return "找不到符合條件的課程。"
    return _format_docs(docs, top_k)


# ── PostgreSQL path ───────────────────────────────────────────────────────────


def _pg_retrieve(keyword: str, top_k: int, sql_filter: str) -> str:
    import jieba
    import psycopg2
    import psycopg2.extras

    from paths import DATABASE_URL

    BM25_INDEX = "idx_course_bm25"
    YEAR, SEMESTER = COURSE_YEAR, COURSE_SEMESTER

    bm25_query = " ".join(t for t in jieba.cut(keyword, cut_all=False) if t.strip())

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    try:
        # Build WHERE conditions
        bm25_where = (
            "y = %(year)s AND s = %(sem)s "
            "AND content_segmented <@> to_bm25query(%(bm25_q)s, %(bm25_idx)s) < 0"
        )
        cols = "id, name, time_raw AS time, teacher, point"
        order = "score"

        if sql_filter:
            escaped = sql_filter.replace("%", "%%")
            bm25_where += " AND " + escaped

        params = {
            "bm25_q": bm25_query,
            "bm25_idx": BM25_INDEX,
            "year": YEAR,
            "sem": SEMESTER,
            "top_k": top_k,
        }

        cur.execute(f"""
            SELECT {cols},
                   content_segmented <@> to_bm25query(%(bm25_q)s, %(bm25_idx)s) AS score
            FROM public.course
            WHERE {bm25_where}
            ORDER BY score
            LIMIT %(top_k)s
        """, params)
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    if not rows:
        return "找不到符合條件的課程。"
    return _format_rows(rows, top_k)


# ── Tool ──────────────────────────────────────────────────────────────────────


@tool
def retrieve_tool(keyword: str, top_k: int = 10, sql_filter: str = "") -> str:
    """檢索符合條件的課程候選清單(查課與排課都用這個)。

    參數:
    - keyword: 想找的課程描述(如「人工智慧」「資料庫」)
    - top_k: 最多回傳幾門候選(查課用預設即可;排課建議 20)
    - sql_filter: 可選的 SQL WHERE 子句(由 text_to_sql_tool 產生),用來過濾時段;沒有就留空

    回傳:編號候選清單,每門含 **course_id(13 位)**、課名、時間、老師、學分。
    要排課時,把這些 course_id 用逗號接起來傳給 schedule_tool。
    """
    try:
        if sql_filter and sql_filter.startswith("ERROR:"):
            return f"ERROR: 時間過濾條件無效，請重新執行 text_to_sql_tool。原始錯誤: {sql_filter[len('ERROR:'):]}"
        if USE_SQLITE:
            return _sqlite_retrieve(keyword, top_k, sql_filter)
        else:
            return _pg_retrieve(keyword, top_k, sql_filter)
    except FileNotFoundError:
        return f"ERROR: 找不到向量庫 {PICKLE_FILE},請先執行 build.py 建立索引。"
    except sqlite3.Error as e:
        return f"ERROR: 課程資料庫無法查詢({e})。請確認 data.db 已就緒。"
    except Exception as e:
        return f"ERROR: 檢索時發生錯誤:{e}"


register_tool(retrieve_tool)
