from langchain_core.tools import tool
from typing import Optional
import pickle
import sqlite3
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.bm25 import BM25Retriever
from langchain_core.documents import Document


@tool
def retrieval_tool(
    question: str,
    sql_filter: Optional[str] = None,
    db_path: str = "data.db",
    pickle_file: str = "vectorstore.pkl",
) -> str:
    """執行課程檢索。

    用途：當需要找課程時使用此工具。

    參數：
    - question: 用戶想要找的課程描述
    - sql_filter: SQL WHERE 子句（可選），用於過濾不想要的時段

    流程：
    1. 若有 sql_filter，用 SQL 過濾 COURSE 表，取得符合條件的課程 IDs
    2. 只對過濾後的課程執行 Vector Search / BM25
    3. 回傳格式化後的課程列表

    回傳：相關課程列表（名稱、時間、老師）
    """

    def dict_factory(cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    conn = sqlite3.connect(db_path)
    conn.row_factory = dict_factory
    cursor = conn.cursor()

    if sql_filter:
        try:
            query = f"SELECT * FROM COURSE WHERE {sql_filter}"
            cursor.execute(query)
        except sqlite3.Error as e:
            cursor.execute("SELECT * FROM COURSE")
    else:
        cursor.execute("SELECT * FROM COURSE")

    all_courses = cursor.fetchall()
    conn.close()

    if sql_filter and all_courses:
        all_docs = []
        for course in all_courses:
            time_str = course.get("time", "")
            name = course.get("name", "")
            teacher = course.get("teacher", "")
            content = f"課程名稱是{name}, 上課時間是{time_str}, 這堂課的老師是{teacher}"
            doc = Document(page_content=content, metadata=course)
            all_docs.append(doc)

        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = min(10, len(all_docs))
        docs = bm25_retriever.invoke(question)
    else:
        with open(pickle_file, "rb") as f:
            retriever = pickle.load(f)
        docs = retriever.invoke(question)

    if not docs:
        return "找不到符合條件的課程。"

    result_lines = []
    for i, doc in enumerate(docs, 1):
        course = doc.metadata
        name = course.get("name", "N/A")
        time = course.get("time", "N/A")
        teacher = course.get("teacher", "N/A")
        result_lines.append(f"{i}. 課程名稱：{name}")
        result_lines.append(f"   上課時間：{time}")
        result_lines.append(f"   授課老師：{teacher}")
        result_lines.append("")

    return "\n".join(result_lines)
