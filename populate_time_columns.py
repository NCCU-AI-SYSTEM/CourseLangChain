from __future__ import annotations

import json
import os

import psycopg2
import psycopg2.extras

PERIOD_CHARS = ["A", "B", "1", "2", "3", "4", "C", "D", "5", "6", "7", "8", "E", "F", "G", "H"]
WEEKDAY_CHARS = ["一", "二", "三", "四", "五", "六", "日"]


def _start_hour(idx: int) -> int:
    return 6 + idx if idx > 0 else 6


def _end_hour(idx: int) -> int:
    return 7 + idx if idx > 0 else 7


def parse_time_str(time_str: str) -> list[dict]:
    if not time_str or time_str.strip() in {"未定或彈性", "未定"}:
        return []

    sessions: list[dict] = []
    cur_weekday: int | None = None
    cur_start_idx: int | None = None
    cur_end_idx: int | None = None

    def flush() -> None:
        nonlocal cur_start_idx, cur_end_idx
        if cur_weekday is None or cur_start_idx is None or cur_end_idx is None:
            return
        sessions.append({
            "weekday":    cur_weekday + 1,
            "start_hour": _start_hour(cur_start_idx),
            "end_hour":   _end_hour(cur_end_idx),
        })
        cur_start_idx = None
        cur_end_idx = None

    for ch in time_str:
        if ch in WEEKDAY_CHARS:
            flush()
            cur_weekday = WEEKDAY_CHARS.index(ch)
            continue
        if ch not in PERIOD_CHARS:
            continue
        idx = PERIOD_CHARS.index(ch)
        if cur_start_idx is None:
            cur_start_idx = cur_end_idx = idx
        elif idx == cur_end_idx + 1:
            cur_end_idx = idx
        else:
            flush()
            cur_start_idx = cur_end_idx = idx

    flush()
    return sessions


def get_connection():
    url = os.environ.get("DATABASE_URL")
    if url:
        return psycopg2.connect(url)
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5432")),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", "postgres"),
        dbname=os.getenv("PG_DBNAME", "nccu"),
    )


BATCH_SIZE = 500

UPDATE_SQL = """
    UPDATE public.course SET
        sessions      = %(sessions)s::jsonb,
        weekdays      = %(weekdays)s::integer[],
        has_morning   = %(has_morning)s,
        has_noon      = %(has_noon)s,
        has_afternoon = %(has_afternoon)s,
        has_evening   = %(has_evening)s
    WHERE id = %(id)s
"""


def main() -> None:
    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    cur.execute("SELECT COUNT(*) FROM public.course WHERE sessions IS NULL")
    total = cur.fetchone()[0]
    print(f"[populate] {total} rows to process")

    if total == 0:
        print("[populate] Nothing to do.")
        cur.close()
        conn.close()
        return

    cur.execute("SELECT id, time_raw FROM public.course WHERE sessions IS NULL")
    rows = cur.fetchall()

    batch: list[dict] = []
    processed = 0

    def flush_batch(b: list[dict]) -> None:
        if not b:
            return
        upd_cur = conn.cursor()
        psycopg2.extras.execute_batch(upd_cur, UPDATE_SQL, b, page_size=BATCH_SIZE)
        conn.commit()
        upd_cur.close()

    for row in rows:
        cid = row["id"]
        time_raw = row["time_raw"] or ""
        sessions = parse_time_str(time_raw)
        weekdays = sorted(set(s["weekday"] for s in sessions))

        batch.append({
            "id":           cid,
            "sessions":     json.dumps(sessions, ensure_ascii=False),
            "weekdays":     weekdays,
            "has_morning":  any(8  <= s["start_hour"] < 12 for s in sessions),
            "has_noon":     any(12 <= s["start_hour"] < 13 for s in sessions),
            "has_afternoon":any(13 <= s["start_hour"] < 18 for s in sessions),
            "has_evening":  any(s["start_hour"] >= 18       for s in sessions),
        })

        if len(batch) >= BATCH_SIZE:
            flush_batch(batch)
            processed += len(batch)
            print(f"[populate] {processed}/{total}")
            batch = []

    flush_batch(batch)
    processed += len(batch)
    print(f"[populate] Done. {processed}/{total} rows updated.")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
