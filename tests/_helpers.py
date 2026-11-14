"""測試共用的小工具。

目前只有「資料庫在不在」這一項:需要 PostgreSQL 的測試在連不上時一律**略過**,
不要崩潰、也不要報成失敗 —— 否則「沒開資料庫」和「程式壞了」在結果裡長得一樣。
CI 沒有資料庫(data.sql.gz 不進 git),靠的就是這個。
"""
from __future__ import annotations

from functools import cache


@cache
def db_available() -> bool:
    """連得上 PostgreSQL、且有當期課程資料。結果在同一個程序內只查一次。"""
    try:
        import psycopg2

        from paths import COURSE_SEMESTER, COURSE_YEAR, DATABASE_URL

        conn = psycopg2.connect(DATABASE_URL, connect_timeout=3)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM public.course WHERE y = %s AND s = %s LIMIT 1",
                (COURSE_YEAR, COURSE_SEMESTER),
            )
            return cur.fetchone() is not None
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 — 任何原因連不上都當作沒有資料庫
        return False


DB_SKIP_REASON = "連不上 PostgreSQL(或沒有當期資料),略過"
