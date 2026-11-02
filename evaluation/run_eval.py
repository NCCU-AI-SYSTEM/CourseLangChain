"""跑評估並寫出報告。需要 PostgreSQL。

  uv run python -m evaluation.run_eval                # 報告寫到 evaluation/results/
  uv run python -m evaluation.run_eval --boot 2000    # 快一點的版本

只納入標註完成的查詢:規則層一律可用;人工層要該題的候選全部填好分數才算。
填到一半的題目會被略過並列在報告裡,不會用半套標註算分。
"""
from __future__ import annotations

import argparse
import datetime as dt

from .data import (
    MANUAL_STRATA,
    RESULTS_DIR,
    STRATA,
    STRATUM_LABEL,
    Query,
    load_judgments,
    load_queries,
    write_tsv,
)
from .metrics import judged_at_k, mean, mrr_at_k, ndcg_at_k, paired_bootstrap, recall_at_k

K = 10
METRICS = {"Recall@10": recall_at_k, "MRR@10": mrr_at_k, "nDCG@10": ndcg_at_k}
SWEEP = (10, 30, 60, 100)
MIN_STRATUM_N = 5  # 單一題型少於這個題數就不估信賴區間


def _systems(prod_k: int) -> tuple[list[str], str]:
    ks = sorted(set(SWEEP) | {prod_k})
    return ["bm25", "vector", *(f"rrf{k}" for k in ks)], f"rrf{prod_k}"


def _label(system: str, prod: str) -> str:
    if system == "bm25":
        return "只用 BM25"
    if system == "vector":
        return "只用向量"
    return f"RRF k={system[3:]}" + ("（正式設定）" if system == prod else "")


def _select(queries: list[Query], judgments: dict) -> tuple[list[tuple[Query, dict]], dict]:
    """回 (可評估的題目與其標註, 各題型被略過的原因與題數)。"""
    ready: list[tuple[Query, dict]] = []
    skipped = {s: {"unlabeled": 0, "no_relevant": 0} for s in STRATA}
    for q in queries:
        rel = judgments.get(q.qid, {})
        if not rel or any(g is None for g in rel.values()):
            skipped[q.stratum]["unlabeled"] += 1
        elif not any(g >= 1 for g in rel.values()):
            skipped[q.stratum]["no_relevant"] += 1
        else:
            ready.append((q, dict(rel)))
    return ready, skipped


def _fmt_diff(d: float, lo: float, hi: float) -> str:
    star = "＊" if lo > 0 or hi < 0 else ""
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}"


def _report(meta: dict, ready, skipped, systems, prod, scores, args) -> str:
    strata_of = [q.stratum for q, _ in ready]
    main = ["bm25", "vector", prod]
    used = {s: sum(st == s for st in strata_of) for s in STRATA}
    out: list[str] = ["# 檢索評估報告", ""]
    out += [
        f"- 產生時間：{meta['time']}",
        f"- 資料：{meta['semester']} 學期，embedding 模型 {meta['embed']}，每路候選 {meta['cand']} 門",
        f"- 評估深度 K={K}；分層成對 bootstrap {args.boot} 次，種子 {args.seed}",
        "",
        "## 題目數",
        "",
        "| 題型 | 納入 | 尚未標註完 | 標註後無相關課程 |",
        "|---|---|---|---|",
    ]
    for s in STRATA:
        out.append(f"| {STRATUM_LABEL[s]} | {used[s]} | {skipped[s]['unlabeled']} "
                   f"| {skipped[s]['no_relevant']} |")
    if not any(used[s] for s in MANUAL_STRATA):
        out += ["", "> ⚠ 主題與口語兩層還沒有標註完成的題目。目前的結果只涵蓋字面比對型的查詢，"
                "會系統性地偏向 BM25，不能當成整體結論。"]

    out += ["", "## 一、整體表現", "", "| 系統 | " + " | ".join(METRICS) + " |",
            "|---" * (len(METRICS) + 1) + "|"]
    for s in systems:
        out.append(f"| {_label(s, prod)} | "
                   + " | ".join(f"{mean(scores[s][m]):.3f}" for m in METRICS) + " |")

    out += ["", "## 二、各題型表現"]
    for m in METRICS:
        out += ["", f"**{m}**", "", "| 題型 | 題數 | " + " | ".join(_label(s, prod) for s in main) + " |",
                "|---" * (len(main) + 2) + "|"]
        for st in STRATA:
            idx = [i for i, x in enumerate(strata_of) if x == st]
            if not idx:
                continue
            cells = " | ".join(f"{mean([scores[s][m][i] for i in idx]):.3f}" for s in main)
            out.append(f"| {STRATUM_LABEL[st]} | {len(idx)} | {cells} |")

    comparisons = [(prod, "bm25"), (prod, "vector")]
    out += ["", "## 三、差異與 95% 信賴區間", "",
            "整體（各題型分層重抽、同一題成對比較）：", "",
            "| 比較 | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for a, b in comparisons:
        cells = []
        for m in METRICS:
            cells.append(_fmt_diff(*paired_bootstrap(
                scores[a][m], scores[b][m], strata_of, args.boot, args.seed)))
        out.append(f"| {_label(a, prod)} − {_label(b, prod)} | " + " | ".join(cells) + " |")

    out += ["", "各題型（nDCG@10，題型內成對重抽）：", "",
            "| 題型 | " + " | ".join(f"{_label(a, prod)} − {_label(b, prod)}" for a, b in comparisons)
            + " |", "|---" * (len(comparisons) + 1) + "|"]
    for st in STRATA:
        idx = [i for i, x in enumerate(strata_of) if x == st]
        if not idx:
            continue
        if len(idx) < MIN_STRATUM_N:
            out.append(f"| {STRATUM_LABEL[st]} | " + " | ".join(["題數不足"] * len(comparisons)) + " |")
            continue
        cells = []
        for a, b in comparisons:
            xa = [scores[a]["nDCG@10"][i] for i in idx]
            xb = [scores[b]["nDCG@10"][i] for i in idx]
            cells.append(_fmt_diff(*paired_bootstrap(xa, xb, [st] * len(idx), args.boot, args.seed)))
        out.append(f"| {STRATUM_LABEL[st]} | " + " | ".join(cells) + " |")

    out += ["", "## 四、RRF 的 k 值", "", "| k | " + " | ".join(METRICS) + " |",
            "|---" * (len(METRICS) + 1) + "|"]
    for s in systems:
        if s.startswith("rrf"):
            out.append(f"| {_label(s, prod)} | "
                       + " | ".join(f"{mean(scores[s][m]):.3f}" for m in METRICS) + " |")

    manual_idx = [i for i, (q, _) in enumerate(ready) if q.source != "rule"]
    out += ["", "## 五、人工層的標註涵蓋率（前 10 名中已標註的比例）", ""]
    if manual_idx:
        out += ["| 系統 | judged@10 |", "|---|---|"]
        for s in systems:
            out.append(f"| {_label(s, prod)} | "
                       f"{mean([scores[s]['judged@10'][i] for i in manual_idx]):.3f} |")
    else:
        out.append("尚無人工標註完成的題目。")

    out += [
        "", "## 怎麼讀這份報告", "",
        "- ＊ 表示 95% 信賴區間不包含 0。",
        "- 整體平均是逐題平均，各題型的權重等於它的題數。",
        "- 教師與課名兩層由規則標註，偏向字面比對；主題與口語兩層才考驗語意，要分開看。",
        "- 未標註的課程一律視為不相關。涵蓋率低於 0.9 時分數被低估，請重跑 build_gold pool 補標。",
    ]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BM25、向量、RRF 的離線檢索評估")
    parser.add_argument("--boot", type=int, default=10_000, help="bootstrap 重抽次數")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from paths import COURSE_SEMESTER, COURSE_YEAR, EMBED_MODEL, RRF_CANDIDATES, RRF_K

    from .modes import Retriever

    ready, skipped = _select(load_queries(), load_judgments())
    if not ready:
        raise SystemExit("沒有可評估的查詢。先跑 build_gold sample;人工層要把分數填完。")
    systems, prod = _systems(RRF_K)

    runs: dict[str, dict[str, list[str]]] = {}
    with Retriever() as retriever:
        for i, (q, _) in enumerate(ready, 1):
            runs[q.qid] = {s: retriever.search(s, q.query, K) for s in systems}
            print(f"\r檢索中 {i}/{len(ready)}", end="", flush=True)
    print()

    scores = {s: {m: [] for m in (*METRICS, "judged@10")} for s in systems}
    per_query: list[dict] = []
    for q, rel in ready:
        for s in systems:
            ranked = runs[q.qid][s]
            row = {"qid": q.qid, "stratum": q.stratum, "query": q.query, "system": s}
            for m, fn in METRICS.items():
                value = fn(ranked, rel, K)
                scores[s][m].append(value)
                row[m] = f"{value:.4f}"
            judged = judged_at_k(ranked, rel, K)
            scores[s]["judged@10"].append(judged)
            row["judged@10"] = f"{judged:.2f}"
            row["top10"] = " ".join(ranked)
            per_query.append(row)

    meta = {
        "time": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "semester": f"{COURSE_YEAR}-{COURSE_SEMESTER}",
        "embed": EMBED_MODEL,
        "cand": RRF_CANDIDATES,
    }
    report = _report(meta, ready, skipped, systems, prod, scores, args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "report.md").write_text(report, encoding="utf-8")
    write_tsv(RESULTS_DIR / "per_query.tsv",
              ("qid", "stratum", "query", "system", *METRICS, "judged@10", "top10"), per_query)

    print(f"納入 {len(ready)} 題。")
    for s in ("bm25", "vector", prod):
        print(f"  {_label(s, prod):<16}" + "  ".join(f"{m} {mean(scores[s][m]):.3f}" for m in METRICS))
    print(f"報告:{RESULTS_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
