"""段 2 tool wrapper 驗證:schedule_tool / query_courses_tool / registry。

跑:`uv run python -m tests.test_schedule_tool`(在 CourseLangChain/ 下)

schedule_tool 用真實課程 id(預設從 PostgreSQL 取,USE_SQLITE=true 時退回 data.db);
query_courses_tool 用假 retriever 測格式與 top_k 傳遞,不載真的索引。
"""
from __future__ import annotations

import os
import sqlite3

import paths
from tools.query_courses import query_courses_tool
from tools import retrieve as retrieve_mod
from tools.schedule_tool import _split_ids, schedule_tool

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


def _real_ids(n: int = 8) -> list[str]:
    """取幾個真實課程 id。預設走 PostgreSQL;USE_SQLITE=true 才回頭找 data.db。

    兩邊都拿不到就回空清單,呼叫端會把該案例標成略過 —— 純執行期的 repo 不該因為
    缺資料成品而測試失敗。
    """
    if not paths.USE_SQLITE:
        try:
            import psycopg2

            conn = psycopg2.connect(paths.DATABASE_URL)
            cur = conn.cursor()
            cur.execute(
                "SELECT id FROM public.course "
                "WHERE y = %s AND s = %s AND time_raw NOT IN ('','未定或彈性') "
                "AND point > 0 LIMIT %s",
                (paths.COURSE_YEAR, paths.COURSE_SEMESTER, n),
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [r[0] for r in rows]
        except Exception:
            return []

    if not os.path.exists("data.db"):
        return []
    conn = sqlite3.connect("data.db")
    rows = conn.execute(
        "SELECT id FROM COURSE WHERE time NOT IN ('','未定或彈性') AND point > 0 LIMIT ?",
        (n,),
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def test_split_ids() -> None:
    print("[_split_ids]")
    check(_split_ids("1,2, 3") == ["1", "2", "3"], "逗號+空白分隔")
    check(_split_ids("1\n2、3 4") == ["1", "2", "3", "4"], "換行/頓號/空白混合")
    check(_split_ids("1,1,2") == ["1", "2"], "去重保序")
    check(_split_ids("  ") == [], "全空白 → 空清單")


def test_schedule_tool_happy() -> None:
    print("[schedule_tool 正常路徑]")
    ids = _real_ids(8)
    if not ids:
        print("    (略過:拿不到真實課程資料 —— 先起 postgres 或提供 data.db)")
        return
    check(len(ids) >= 4, f"取得 {len(ids)} 個真實 id")
    out = schedule_tool.invoke(
        {"course_ids": ",".join(ids), "min_credits": 2, "max_credits": 12, "max_results": 3}
    )
    check(not out.startswith("ERROR"), "正常輸入不回 ERROR")
    check("方案" in out and "| 課程名稱 |" in out, "輸出含 Markdown 課表方案")
    check("學分" in out, "輸出含學分欄位")
    print("    --- 範例輸出(前 400 字)---")
    print("    " + out[:400].replace("\n", "\n    "))


def test_schedule_tool_errors() -> None:
    print("[schedule_tool 錯誤/邊界]")
    check(schedule_tool.invoke({"course_ids": ""}).startswith("ERROR"), "空 id → ERROR")
    check(
        schedule_tool.invoke({"course_ids": "0000000000000"}).startswith("ERROR"),
        "全找不到的 id → ERROR",
    )
    check(
        schedule_tool.invoke(
            {"course_ids": "1", "min_credits": 20, "max_credits": 5}
        ).startswith("ERROR"),
        "下限 > 上限 → ERROR",
    )
    # 不可能達成的學分 → 給說明而非 ERROR
    ids = _real_ids(5)
    if not ids:
        print("    (略過學分過高案例:拿不到真實課程資料)")
        return
    hint = schedule_tool.invoke(
        {"course_ids": ",".join(ids), "min_credits": 999, "max_credits": 9999}
    )
    check(not hint.startswith("ERROR") and "排不出" in hint, "學分過高 → 回說明(非 ERROR)")


def test_query_courses_format() -> None:
    print("[query_courses_tool 格式(假 retriever)]")

    class _FakeDoc:
        def __init__(self, meta):
            self.metadata = meta

    class _FakeSub:
        """假的子檢索器,記下 _set_k 傳進來的 k。"""

        def __init__(self, vectorstore_style: bool):
            if vectorstore_style:
                self.search_kwargs = {"k": 5}  # 模擬 FAISS VectorStoreRetriever
            else:
                self.k = 5  # 模擬 BM25Retriever

        @property
        def received_k(self) -> int:
            return self.search_kwargs["k"] if hasattr(self, "search_kwargs") else self.k

    class _FakeRetriever:
        def __init__(self):
            self.retrievers = [_FakeSub(False), _FakeSub(True)]

        def invoke(self, _):
            return [
                _FakeDoc({"id": "1142000348021", "name": "管理學", "time": "五D56",
                          "teacher": "李老師", "point": 3}),
                _FakeDoc({"id": "1142000348051", "name": "管理學", "time": "三D56",
                          "teacher": "呂老師", "point": 3}),
            ]

    fake = _FakeRetriever()
    # 這條是已棄用的 SQLite 路徑,預設不會走到,測格式時明確切過去
    orig_use_sqlite = retrieve_mod.USE_SQLITE
    retrieve_mod.USE_SQLITE = True
    retrieve_mod._retriever = fake
    try:
        out = query_courses_tool.invoke({"keyword": "管理", "top_k": 5})
    finally:
        retrieve_mod._retriever = None
        retrieve_mod.USE_SQLITE = orig_use_sqlite

    check("course_id: 1142000348021" in out, "輸出含 13 位 course_id")
    check("學分: 3" in out and "老師: 李老師" in out, "輸出含學分與老師")


def test_query_courses_topk_propagates() -> None:
    """回歸測試:top_k 必須傳到兩個子檢索器。

    舊版把 k=5 烤進 pickle,不管 agent 要 top_k=20 都只拿得到 ~10 筆 —— 而
    brain_agent 的 prompt 明確要求排課用 top_k=20,等於排課長期候選不足還沒人發現。
    """
    print("[query_courses_tool top_k 傳遞]")

    class _FakeDoc:
        def __init__(self, meta):
            self.metadata = meta

    class _FakeSub:
        def __init__(self, vectorstore_style: bool):
            self.vectorstore_style = vectorstore_style
            if vectorstore_style:
                self.search_kwargs = {"k": 5}
            else:
                self.k = 5

        @property
        def received_k(self) -> int:
            return self.search_kwargs["k"] if self.vectorstore_style else self.k

    class _FakeRetriever:
        def __init__(self):
            self.retrievers = [_FakeSub(False), _FakeSub(True)]

        def invoke(self, _):
            return [_FakeDoc({"id": "1" * 13, "name": "測試課", "time": "一12",
                              "teacher": "王老師", "point": 3})]

    fake = _FakeRetriever()
    orig_use_sqlite = retrieve_mod.USE_SQLITE
    retrieve_mod.USE_SQLITE = True
    retrieve_mod._retriever = fake
    try:
        query_courses_tool.invoke({"keyword": "測試", "top_k": 20})
    finally:
        retrieve_mod._retriever = None
        retrieve_mod.USE_SQLITE = orig_use_sqlite

    bm25_sub, faiss_sub = fake.retrievers
    check(bm25_sub.received_k == 20, f"BM25 子檢索器收到 k=20(實際 {bm25_sub.received_k})")
    check(faiss_sub.received_k == 20, f"FAISS 子檢索器收到 k=20(實際 {faiss_sub.received_k})")


def test_query_courses_sql_filter() -> None:
    print("[query_courses_tool sql_filter 分支]")
    if not _real_ids(1):
        print("    (略過:拿不到真實課程資料)")
        return

    # 兩個資料源的 WHERE 方言不同:SQLite 是 GLOB 比對 time 文字欄,
    # PostgreSQL 的欄位已改名 time_raw 且支援 LIKE
    good = "time GLOB '*三*'" if paths.USE_SQLITE else "time_raw LIKE '%三%'"
    out = query_courses_tool.invoke({"keyword": "管理", "top_k": 3, "sql_filter": good})
    check(not out.startswith("ERROR"), "合法 sql_filter 不回 ERROR")
    check("course_id" in out, "sql_filter 分支輸出仍含 course_id")

    # 壞 filter 的行為兩邊本來就不同,如實測而不是假裝一致:
    #   SQLite —— 撈不動就退回全表(不 crash)
    #   PostgreSQL —— 整句 SQL 無效,回 ERROR 字串讓 agent 重跑 text_to_sql
    bad = query_courses_tool.invoke(
        {"keyword": "管理", "top_k": 2, "sql_filter": "this is not sql"}
    )
    if paths.USE_SQLITE:
        check(not bad.startswith("ERROR"), "無效 sql_filter 退回全表而非 ERROR")
    else:
        check(bad.startswith("ERROR"), "無效 sql_filter 回 ERROR(agent 會重跑 text_to_sql)")


def test_registry() -> None:
    print("[registry 註冊]")
    from tools.registry import all_tools

    names = {getattr(t, "name", "") for t in all_tools()}
    check("schedule_tool" in names, "schedule_tool 已註冊")
    check("retrieve_tool" in names, "retrieve_tool 已註冊")


if __name__ == "__main__":
    test_split_ids()
    test_schedule_tool_happy()
    test_schedule_tool_errors()
    test_query_courses_format()
    test_query_courses_topk_propagates()
    test_query_courses_sql_filter()
    test_registry()
    print(f"\n結果:{_PASS} passed, {_FAIL} failed")
    raise SystemExit(1 if _FAIL else 0)
