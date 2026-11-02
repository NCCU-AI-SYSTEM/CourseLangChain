"""規則層標註的自動檢查。需要 PostgreSQL。

  uv run python -m evaluation.check_rules

四項檢查,全部通過才代表規則層的標註可以直接信任:
1. 換一種查法重算相關課程(教師用 LIKE、課名用 =),集合要和標註完全相同。
2. 標註裡的每個課號都存在於契約指定的學期。
3. 每題的相關課程數在 1 到 MAX_RELEVANT 之間。
4. 評估的 RRF 前 10 名與正式 retrieve_tool 的回傳一致:證明評估量的就是正式系統。
"""
from __future__ import annotations

import re
import sys

import psycopg2

from paths import COURSE_SEMESTER, COURSE_YEAR, DATABASE_URL, RRF_K, check_contract

from .build_gold import MAX_RELEVANT
from .data import load_judgments, load_queries

_ID_IN_TOOL = re.compile(r"course_id:\s*(\S+?)｜")
_problems = 0


def _fail(msg: str) -> None:
    global _problems
    _problems += 1
    print(f"  ✗ {msg}")


def main() -> None:
    check_contract()
    queries = [q for q in load_queries() if q.source == "rule"]
    judged = load_judgments()
    sem = (COURSE_YEAR, COURSE_SEMESTER)
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT id FROM public.course WHERE y = %s AND s = %s", sem)
    semester_ids = {str(r[0]) for r in cur.fetchall()}

    print(f"[1-3] 重算相關課程、課號存在、題數範圍({len(queries)} 題)")
    for q in queries:
        labeled = {cid for cid, g in judged.get(q.qid, {}).items() if g and g >= 1}
        if q.stratum == "teacher":
            name = q.query.removesuffix("老師開的課")
            cur.execute("SELECT DISTINCT id FROM public.course WHERE y = %s AND s = %s "
                        "AND teacher LIKE %s", (*sem, f"%{name}%"))
        else:
            cur.execute("SELECT DISTINCT id FROM public.course WHERE y = %s AND s = %s "
                        "AND name = %s", (*sem, q.query))
        recomputed = {str(r[0]) for r in cur.fetchall()}
        if recomputed != labeled:
            _fail(f"{q.qid} {q.query}:標註多了 {sorted(labeled - recomputed)},"
                  f"少了 {sorted(recomputed - labeled)}")
        missing = labeled - semester_ids
        if missing:
            _fail(f"{q.qid}:課號不在 {COURSE_YEAR}-{COURSE_SEMESTER} 學期 {sorted(missing)}")
        if not 1 <= len(labeled) <= MAX_RELEVANT:
            _fail(f"{q.qid}:相關課程 {len(labeled)} 門,超出 1~{MAX_RELEVANT}")
    conn.close()

    print(f"[4] 評估的 rrf{RRF_K} 與正式 retrieve_tool 比對")
    from tools.retrieve import retrieve_tool

    from .modes import _RRF_SQL, Retriever

    with Retriever() as r:
        for q in queries:
            ours = r.search(f"rrf{RRF_K}", q.query, 10)
            text = retrieve_tool.invoke({"keyword": q.query, "top_k": 10})
            prod = _ID_IN_TOOL.findall(text)
            if ours == prod:
                continue
            params = r._params(q.query, 30)
            params["qvec"], params["rrf_k"] = r._vec(q.query), RRF_K
            score = dict(r._rows(_RRF_SQL.format(filter=""), params, "rrf"))
            if set(ours) == set(prod):
                # 順序不同只有一種可接受的原因:錯開的那幾個位置,兩邊的課分數相同。
                swapped = [(a, b) for a, b in zip(ours, prod) if a != b]
                if all(abs(score[a] - score[b]) < 1e-12 for a, b in swapped):
                    print(f"  · {q.qid}:順序不同,但錯開的位置都是同分(可接受)")
                else:
                    _fail(f"{q.qid}:同一批課但排序真的不同 {swapped[:3]}")
            else:
                # 第 10 名剛好同分時,正式系統不固定順序,截斷會切到不同的課。
                # 只要兩邊多出來的課與第 10 名分數完全相同,就是同分、不是錯。
                diff = set(ours) ^ set(prod)
                cutoff = score[ours[-1]]
                if all(abs(score.get(c, -1) - cutoff) < 1e-12 for c in diff):
                    print(f"  · {q.qid}:第 10 名同分,截到不同的課(可接受)")
                else:
                    _fail(f"{q.qid}:前 10 名不同,評估獨有 {sorted(set(ours) - set(prod))},"
                          f"正式獨有 {sorted(set(prod) - set(ours))}")

    print(f"\n結果:{_problems} 個問題" + ("" if _problems else ",規則層標註與評估管線皆可信"))
    sys.exit(1 if _problems else 0)


if __name__ == "__main__":
    main()
