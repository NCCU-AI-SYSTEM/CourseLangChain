"""三種檢索模式:只用 BM25、只用向量、RRF 融合。需要 PostgreSQL。

刻意重用 tools.retrieve 的 SQL 與參數,而不是另寫一份:評估量到的必須是正式系統
實際在跑的那段檢索,否則分數再漂亮,也只是另一個系統的分數。
唯一自己寫的是「只用向量」那句,內容照抄 _RRF_SQL 的 vec_raw 一段,排序依據同樣是
每門課最接近的那一塊(MIN(dist)),也就是 RRF 裡向量那一路的名次。
"""
from __future__ import annotations

import re

import psycopg2
import psycopg2.extras

from paths import COURSE_SEMESTER, COURSE_YEAR, DATABASE_URL, RRF_CANDIDATES, check_contract
from tools.retrieve import _BM25_ONLY_SQL, _CHUNK_FACTOR, _DUP_FACTOR, _RRF_SQL, _encode_query

from .metrics import stable_order

# 多抓幾筆,再在 Python 端排序截斷:SQL 的 LIMIT 可能剛好切在同分的中間。
_TIE_MARGIN = 20

_VECTOR_ONLY_SQL = """
WITH vec_raw AS (
    SELECT k.course_id AS id, k.embedding <=> %(qvec)s::vector AS dist
    FROM public.course_chunk k
    JOIN public.course c ON c.id = k.course_id
    WHERE c.y = %(year)s AND c.s = %(sem)s
      AND k.embedding IS NOT NULL
    ORDER BY dist
    LIMIT %(vec_raw)s
)
SELECT id, -MIN(dist) AS score
FROM vec_raw GROUP BY id
ORDER BY score DESC
LIMIT %(top_k)s
"""

_RRF_NAME = re.compile(r"^rrf(\d+)$")


class Retriever:
    """一條資料庫連線跑完整批查詢;同一句查詢的向量只算一次。"""

    def __init__(self, dsn: str = DATABASE_URL) -> None:
        # 資料庫與 contract.yaml 不符時,向量會來自不同模型:檢索不報錯,只會排出無關的課。
        check_contract()
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = True  # 純讀取;避免一句出錯後整條連線卡在中止的交易裡
        self._qvec: dict[str, str] = {}
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM public.course_chunk WHERE embedding IS NOT NULL)"
            )
            if not cur.fetchone()[0]:
                raise SystemExit(
                    "course_chunk 沒有任何向量,「只用向量」與 RRF 無從比較。"
                    "請先用 CourseDataPrep 跑 embed。"
                )

    def __enter__(self) -> Retriever:
        return self

    def __exit__(self, *exc: object) -> None:
        self.conn.close()

    def _vec(self, query: str) -> str:
        if query not in self._qvec:
            self._qvec[query] = _encode_query(query)
        return self._qvec[query]

    def _params(self, query: str, top_k: int) -> dict:
        # 與 tools.retrieve._pg_retrieve 相同的候選預算,只有 top_k 多留同分的餘裕。
        return {
            "bm25_q": query,
            "year": COURSE_YEAR,
            "sem": COURSE_SEMESTER,
            "top_k": top_k + _TIE_MARGIN,
            "cand": RRF_CANDIDATES,
            "raw": RRF_CANDIDATES * _DUP_FACTOR,
            "vec_raw": RRF_CANDIDATES * _CHUNK_FACTOR,
        }

    def _rows(self, sql: str, params: dict, score_key: str) -> list[tuple[str, float]]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql, params)
            return [(str(row["id"]), float(row[score_key])) for row in cur.fetchall()]

    def search(self, system: str, query: str, top_k: int = 10) -> list[str]:
        """system:bm25、vector,或 rrf<k>(例如 rrf60)。回排序好的 course_id。"""
        params = self._params(query, top_k)
        if system == "bm25":
            rows = self._rows(_BM25_ONLY_SQL.format(filter=""), params, "score")
        elif system == "vector":
            params["qvec"] = self._vec(query)
            rows = self._rows(_VECTOR_ONLY_SQL, params, "score")
        elif m := _RRF_NAME.match(system):
            params["qvec"] = self._vec(query)
            params["rrf_k"] = int(m.group(1))
            rows = self._rows(_RRF_SQL.format(filter=""), params, "rrf")
        else:
            raise ValueError(f"未知的系統 {system!r},可用:bm25、vector、rrf<k>")
        return stable_order(rows, top_k)
