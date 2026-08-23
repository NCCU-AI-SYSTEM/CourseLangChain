"""retrieve_tool — course candidate retrieval, returns 13-digit course_ids.

Two backends, both BM25 + vector fused with RRF:

- USE_SQLITE=false (default, the supported one): PostgreSQL, fused in SQL.
- USE_SQLITE=true  (deprecated): data.db + faiss_index/, fused by EnsembleRetriever.

Same output format, but the two use different embedding models, so rankings differ —
don't diff them row by row. Data artifacts come from the course-data-prep repo.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from warnings import deprecated

from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.tools import tool

from paths import (
    COURSES_JSONL,
    COURSE_SEMESTER,
    COURSE_YEAR,
    DATA_DB,
    FAISS_INDEX_DIR,
    SQLITE_DEPRECATION_MSG,
    USE_SQLITE,
)
from utils.zh_tokenize import tokenize

from .registry import register_tool

logger = logging.getLogger(__name__)

_MISSING_ARTIFACT_HINT = (
    "請先用 course-data-prep repo 產生檢索索引(faiss_index/ 與 courses.jsonl),"
    "並放到專案根目錄。"
)

_retriever = None
_retriever_lock = threading.Lock()


# ── SQLite / FAISS path (deprecated — do not add features here) ──────────────


def _load_documents(jsonl_path: str) -> list[Document]:
    docs: list[Document] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            docs.append(
                Document(page_content=row["page_content"], metadata=row["metadata"])
            )
    return docs


def _build_retriever():
    """Assemble the BM25 + FAISS RRF ensemble from prep's artifacts.

    Not a pickle: pickling the retriever would pin the class path, the langchain
    version and the embedding device, and loading it would execute arbitrary code.
    Rebuilding BM25 here also makes k adjustable per query (see _set_k).
    """
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    from paths import SQLITE_EMBED_MODEL

    docs = _load_documents(COURSES_JSONL)

    embeddings = HuggingFaceEmbeddings(
        model_name=SQLITE_EMBED_MODEL,
        model_kwargs={"device": "cpu"},
    )
    vectorstore = FAISS.load_local(
        FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )

    bm25 = BM25Retriever.from_documents(docs, preprocess_func=tokenize)
    faiss_retriever = vectorstore.as_retriever()
    return EnsembleRetriever(retrievers=[bm25, faiss_retriever], weights=[0.5, 0.5])


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = _build_retriever()
    return _retriever


def _set_k(ensemble, k: int) -> None:
    """Propagate top_k to both sub-retrievers.

    k=5 used to be baked into the pickle, so scheduling silently got ~10 candidates
    even though the prompt asks for top_k=20.
    """
    for sub in ensemble.retrievers:
        if hasattr(sub, "search_kwargs"):  # FAISS VectorStoreRetriever
            sub.search_kwargs["k"] = k
        else:  # BM25Retriever
            sub.k = k


def _dict_factory(cursor, row):
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

# 時間是原始代碼,附在結果後面而不是寫進工具說明 —— 讀它的時候要跟資料在同一則訊息裡,
# 呼叫工具之前看到沒有用。
_TIME_LEGEND = (
    "\n\n(時間代碼:星期+節次,一個字串可含多組。"
    "A=06-07 B=07-08 1=08-09 2=09-10 3=10-11 4=11-12 C=12-13 D=13-14 "
    "5=14-15 6=15-16 7=16-17 8=17-18 E=18-19 F=19-20 G=20-21 H=21-22。"
    "早上=1-4、中午=C,D、下午=5-8、晚上=E-H。"
    "例:三234=星期三 09:00-12:00 上午;一12三EFG=星期一 08:00-10:00 和星期三 18:00-21:00;"
    "未定或彈性=無固定時間。)"
)


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


@deprecated(SQLITE_DEPRECATION_MSG)
def _sqlite_retrieve(keyword: str, top_k: int, sql_filter: str) -> str:
    if sql_filter:
        docs = _bm25_over_sql_sqlite(keyword, sql_filter, top_k)
    else:
        with _retriever_lock:
            retriever = _get_retriever()
            _set_k(retriever, top_k)
            docs = retriever.invoke(keyword)
    if not docs:
        return "找不到符合條件的課程。"
    return _format_docs(docs, top_k) + _TIME_LEGEND


# ── PostgreSQL path ───────────────────────────────────────────────────────────

_DUP_FACTOR = 8
# course_chunk holds ~10 rows per course, so the vector side has to scan deeper
# than the BM25 side to surface the same number of distinct courses.
_CHUNK_FACTOR = 16

_encoder = None
_encoder_lock = threading.Lock()
_pg_has_vectors: bool | None = None


def _get_encoder():
    """Query encoder. Must be the same model prep used to fill the table.

    contract.yaml + check_contract() enforce that: a different model produces
    incomparable vectors, and retrieval fails silently rather than erroring.
    """
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                import os

                from sentence_transformers import SentenceTransformer

                from paths import EMBED_MODEL

                device = os.getenv("EMBED_DEVICE", "cpu")
                logger.info("loading query encoder %s on %s", EMBED_MODEL, device)
                _encoder = SentenceTransformer(EMBED_MODEL, device=device)
    return _encoder


def _encode_query(keyword: str) -> str:
    vec = _get_encoder().encode(keyword, show_progress_bar=False)
    return "[" + ",".join(map(str, vec.tolist())) + "]"


def _check_pg_vectors(cur) -> bool:
    global _pg_has_vectors
    if _pg_has_vectors is None:
        cur.execute("SELECT EXISTS (SELECT 1 FROM public.course_chunk WHERE embedding IS NOT NULL)")
        _pg_has_vectors = bool(cur.fetchone()[0])
        if not _pg_has_vectors:
            logger.warning(
                "PostgreSQL 沒有任何 embedding,退回純 BM25 檢索。"
                "要拿到混合檢索請用 course-data-prep 跑 embed。"
            )
    return _pg_has_vectors


# `|||` is OR; the default `@@@` is AND and returns 0 rows for conversational queries.
_RRF_SQL = """
WITH bm25_raw AS (
    SELECT id, pdb.score(pk) AS score
    FROM public.course
    WHERE content_text ||| %(bm25_q)s
      AND y = %(year)s AND s = %(sem)s
      {filter}
    ORDER BY score DESC
    LIMIT %(raw)s
),
bm25 AS (
    SELECT id, RANK() OVER (ORDER BY MAX(score) DESC) AS rk
    FROM bm25_raw GROUP BY id
    ORDER BY rk LIMIT %(cand)s
),
vec_raw AS (
    -- One row per chunk, so this LIMIT counts chunks rather than courses —
    -- hence its own, larger budget. vec below collapses them with MIN(dist):
    -- a course ranks by its single best-matching section.
    SELECT k.course_id AS id, k.embedding <=> %(qvec)s::vector AS dist
    FROM public.course_chunk k
    JOIN public.course c ON c.id = k.course_id
    WHERE c.y = %(year)s AND c.s = %(sem)s
      AND k.embedding IS NOT NULL
      {filter}
    ORDER BY dist
    LIMIT %(vec_raw)s
),
vec AS (
    SELECT id, RANK() OVER (ORDER BY MIN(dist)) AS rk
    FROM vec_raw GROUP BY id
    ORDER BY rk LIMIT %(cand)s
)
SELECT * FROM (
    SELECT DISTINCT ON (c.id)
           c.id, c.name, c.time_raw AS time, c.teacher, c.point,
           COALESCE(1.0 / (%(rrf_k)s + b.rk), 0)
         + COALESCE(1.0 / (%(rrf_k)s + v.rk), 0) AS rrf
    FROM bm25 b
    FULL OUTER JOIN vec v ON b.id = v.id
    JOIN public.course c ON c.id = COALESCE(b.id, v.id)
    ORDER BY c.id
) t
ORDER BY rrf DESC
LIMIT %(top_k)s
"""

_BM25_ONLY_SQL = """
WITH raw AS (
    SELECT id, pdb.score(pk) AS score
    FROM public.course
    WHERE content_text ||| %(bm25_q)s
      AND y = %(year)s AND s = %(sem)s
      {filter}
    ORDER BY score DESC
    LIMIT %(raw)s
),
best AS (
    SELECT id, MAX(score) AS score FROM raw GROUP BY id ORDER BY score DESC LIMIT %(top_k)s
)
SELECT * FROM (
    SELECT DISTINCT ON (c.id)
           c.id, c.name, c.time_raw AS time, c.teacher, c.point, b.score
    FROM best b JOIN public.course c ON c.id = b.id
    ORDER BY c.id
) t
ORDER BY score DESC
LIMIT %(top_k)s
"""


def _pg_retrieve(keyword: str, top_k: int, sql_filter: str) -> str:
    import psycopg2
    import psycopg2.extras

    from paths import DATABASE_URL, RRF_CANDIDATES, RRF_K

    filter_sql = f" AND {sql_filter.replace('%', '%%')}" if sql_filter else ""

    params = {
        "bm25_q": keyword,
        "year": COURSE_YEAR,
        "sem": COURSE_SEMESTER,
        "top_k": top_k,
        "cand": RRF_CANDIDATES,
        "raw": RRF_CANDIDATES * _DUP_FACTOR,
        "vec_raw": RRF_CANDIDATES * _CHUNK_FACTOR,
        "rrf_k": RRF_K,
    }

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        if _check_pg_vectors(cur):
            params["qvec"] = _encode_query(keyword)
            cur.execute(_RRF_SQL.format(filter=filter_sql), params)
        else:
            cur.execute(_BM25_ONLY_SQL.format(filter=filter_sql), params)
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    if not rows:
        return "找不到符合條件的課程。"
    return _format_rows(rows, top_k) + _TIME_LEGEND


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
    except FileNotFoundError as e:
        return f"ERROR: 找不到檢索索引({e.filename or e})。{_MISSING_ARTIFACT_HINT}"
    except sqlite3.Error as e:
        return f"ERROR: 課程資料庫無法查詢({e})。請確認 data.db 已就緒。"
    except Exception as e:
        return f"ERROR: 檢索時發生錯誤:{e}"


register_tool(retrieve_tool)
