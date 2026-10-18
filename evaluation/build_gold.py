"""建立評估用的查詢集與相關性標註。需要 PostgreSQL。

  uv run python -m evaluation.build_gold sample   # 抽「教師」「課名」兩層,規則自動標註
  uv run python -m evaluation.build_gold pool     # 匯出「主題」「口語」兩層的待標註表

兩步都可以重跑。sample 用固定種子,結果不變,也不會動到人工標註;pool 只補上新出現
的候選,已經填好的分數一律保留。
"""
from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict

from .data import (
    JUDGMENT_FIELDS,
    JUDGMENTS,
    QUERIES,
    QUERIES_MANUAL,
    QUERY_FIELDS,
    load_queries,
    read_tsv,
    write_tsv,
)

# 每題相關課程數的上限。相關課程超過 10 門的題目,Recall@10 再好也到不了 1,
# 題目之間就不可比了,所以規則層只抽相關課程不超過這個數的教師與課名。
MAX_RELEVANT = 8
DEFAULT_SEED = 20261015

_TEACHER_SPLIT = re.compile(r"[、,，/;；\s]+")
_TEACHER_TOKEN = re.compile(r"^[一-鿿]{2,4}$")
# 教師欄偶爾出現的佔位字,不是人名。sample 之後仍建議把 queries.tsv 掃過一遍。
_NOT_A_NAME = {"待聘", "未定", "教師", "老師", "助教", "多位教師", "全體教師", "本系教師"}

POOL_SYSTEMS = ("bm25", "vector", "rrf10", "rrf30", "rrf60", "rrf100")
SNIPPET_CHARS = 150


def _eligible(index: dict[str, set[str]]) -> list[str]:
    return sorted(key for key, ids in index.items() if 1 <= len(ids) <= MAX_RELEVANT)


def build_rule_queries(
    courses: list[tuple[str, str, str]], n_teacher: int, n_title: int, seed: int
) -> tuple[list[dict], list[dict]]:
    """從 (course_id, 課名, 教師) 清單抽題並標註,回 (queries, judgments)。純函數。

    教師題「某某老師開的課」:那位老師開的每一門課都是 grade 2,其餘不相關。
    課名題直接用完整課名查:同名的每一門(不同班)都是 grade 2。
    兩者的相關性由規則完全決定,不需要人工判斷,標註也就不會漏。
    """
    by_teacher: dict[str, set[str]] = defaultdict(set)
    by_title: dict[str, set[str]] = defaultdict(set)
    info: dict[str, tuple[str, str]] = {}
    for cid, name, teacher in courses:
        info[cid] = (name, teacher)
        for token in _TEACHER_SPLIT.split(teacher):
            if _TEACHER_TOKEN.match(token) and token not in _NOT_A_NAME:
                by_teacher[token].add(cid)
        if len(name) >= 2:
            by_title[name].add(cid)

    rng = random.Random(seed)
    teachers = _eligible(by_teacher)
    titles = _eligible(by_title)
    picked_teachers = rng.sample(teachers, min(n_teacher, len(teachers)))
    picked_titles = rng.sample(titles, min(n_title, len(titles)))

    queries: list[dict] = []
    judgments: list[dict] = []

    def add(qid: str, stratum: str, text: str, ids: set[str]) -> None:
        queries.append({"qid": qid, "stratum": stratum, "query": text, "source": "rule"})
        for cid in sorted(ids):
            name, teacher = info[cid]
            judgments.append(
                {"qid": qid, "course_id": cid, "grade": 2, "source": "rule",
                 "name": name, "teacher": teacher, "snippet": ""}
            )

    for i, teacher in enumerate(picked_teachers, 1):
        add(f"T{i:02d}", "teacher", f"{teacher}老師開的課", by_teacher[teacher])
    for i, title in enumerate(picked_titles, 1):
        add(f"N{i:02d}", "title", title, by_title[title])
    return queries, judgments


def _semester_courses(conn) -> list[tuple[str, str, str]]:
    from paths import COURSE_SEMESTER, COURSE_YEAR

    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT ON (id) id, COALESCE(name, ''), COALESCE(teacher, '') "
            "FROM public.course WHERE y = %s AND s = %s ORDER BY id",
            (COURSE_YEAR, COURSE_SEMESTER),
        )
        return [(str(cid), name.strip(), teacher.strip()) for cid, name, teacher in cur.fetchall()]


def _course_info(conn, ids: list[str]) -> dict[str, tuple[str, str, str]]:
    """course_id -> (課名, 教師, 課程目標或簡介的開頭),給標註的人判斷用。"""
    if not ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'course' "
            "AND column_name IN ('objective', 'syllabus') ORDER BY column_name"
        )
        cols = [row[0] for row in cur.fetchall()]
        text = " || ' ' || ".join(f"COALESCE({c}, '')" for c in cols) or "''"
        cur.execute(
            f"SELECT DISTINCT ON (id) id, COALESCE(name, ''), COALESCE(teacher, ''), {text} "
            "FROM public.course WHERE id::text = ANY(%s) ORDER BY id",
            (ids,),
        )
        return {
            str(cid): (name, teacher, " ".join(snippet.split())[:SNIPPET_CHARS])
            for cid, name, teacher, snippet in cur.fetchall()
        }


def cmd_sample(args: argparse.Namespace) -> None:
    import psycopg2

    from paths import DATABASE_URL, check_contract

    check_contract()
    conn = psycopg2.connect(DATABASE_URL)
    try:
        courses = _semester_courses(conn)
    finally:
        conn.close()

    rule_q, rule_j = build_rule_queries(courses, args.n_teacher, args.n_title, args.seed)
    manual_q = [dict(row, source="manual") for row in read_tsv(QUERIES_MANUAL)]
    write_tsv(QUERIES, QUERY_FIELDS, rule_q + manual_q)
    load_queries()  # 寫完立刻讀回驗證一次:題型、qid 重複

    # 規則層每次重新產生;人工層的列(含已填的分數)原樣保留
    kept = [row for row in read_tsv(JUDGMENTS) if row.get("source") != "rule"]
    write_tsv(JUDGMENTS, JUDGMENT_FIELDS, rule_j + kept)

    n_teacher = sum(q["stratum"] == "teacher" for q in rule_q)
    print(f"{len(courses)} 門課 → 教師題 {n_teacher}、課名題 {len(rule_q) - n_teacher}、"
          f"人工題 {len(manual_q)}")
    print(f"已寫入 {QUERIES.name} 與 {JUDGMENTS.name}。下一步:build_gold pool")


def cmd_pool(args: argparse.Namespace) -> None:
    from .modes import Retriever

    queries = load_queries()
    order = {q.qid: i for i, q in enumerate(queries)}
    manual = [q for q in queries if q.source != "rule"]
    rows = read_tsv(JUDGMENTS)
    existing = {(row.get("qid", ""), row.get("course_id", "")) for row in rows}

    new_rows: list[dict] = []
    with Retriever() as retriever:
        for i, q in enumerate(manual, 1):
            found: list[str] = []
            for system in POOL_SYSTEMS:
                for cid in retriever.search(system, q.query, args.depth):
                    if cid not in found:
                        found.append(cid)
            fresh = [cid for cid in found if (q.qid, cid) not in existing]
            # 打亂順序、不標來源:標註的人看不出哪一路找到、排第幾,判斷才不會被名次帶著走
            random.Random(f"{args.seed}-{q.qid}").shuffle(fresh)
            info = _course_info(retriever.conn, fresh)
            for cid in fresh:
                name, teacher, snippet = info.get(cid, ("", "", ""))
                new_rows.append(
                    {"qid": q.qid, "course_id": cid, "grade": "", "source": "manual",
                     "name": name, "teacher": teacher, "snippet": snippet}
                )
            print(f"\r匯集候選 {i}/{len(manual)}", end="", flush=True)
    print()

    merged = rows + new_rows
    merged.sort(key=lambda row: order.get(row.get("qid", ""), len(order)))  # 穩定排序,題內順序不變
    write_tsv(JUDGMENTS, JUDGMENT_FIELDS, merged)
    todo = sum(
        1 for row in merged
        if row.get("source") != "rule" and not str(row.get("grade", "")).strip()
    )
    print(f"新增 {len(new_rows)} 筆候選;目前共 {todo} 筆待標註(grade 欄填 0、1、2)。")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="建立檢索評估的查詢集與相關性標註")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sample = sub.add_parser("sample", help="抽教師、課名兩層並自動標註")
    sample.add_argument("--n-teacher", type=int, default=30)
    sample.add_argument("--n-title", type=int, default=30)
    sample.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sample.set_defaults(func=cmd_sample)

    pool = sub.add_parser("pool", help="匯出主題、口語兩層的待標註候選")
    pool.add_argument("--depth", type=int, default=10, help="每個系統取前幾名進標註池")
    pool.add_argument("--seed", type=int, default=DEFAULT_SEED)
    pool.set_defaults(func=cmd_pool)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
