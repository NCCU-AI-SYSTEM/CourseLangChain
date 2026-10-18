"""評估指標與分層成對 bootstrap。純函數、只用標準函式庫,不需要資料庫就能測。

相關性等級(grade):2 = 直接符合需求、1 = 部分相關、0 = 不相關。
Recall 與 MRR 把 grade >= 1 視為相關;nDCG 用分級增益 2^grade - 1,
所以「直接符合」排在前面,比「部分相關」排在前面得分更高。
未標註的課程一律視為不相關。
"""
from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence

Judgments = Mapping[str, int]  # course_id -> grade


def stable_order(scored: Iterable[tuple[str, float]], top_k: int) -> list[str]:
    """依分數由高到低、同分依 course_id 排序,取前 top_k 門不重複的課。

    SQL 的 ORDER BY 遇到同分不保證順序,而 RRF 特別容易同分:兩門課各只出現在
    一路、名次又相同時,分數完全一樣。不固定順序,同一份資料跑兩次分數會不同。
    """
    best: dict[str, float] = {}
    for cid, score in scored:
        if cid not in best or score > best[cid]:
            best[cid] = score
    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    return [cid for cid, _ in ranked[:top_k]]


def _relevant(rel: Judgments) -> set[str]:
    return {cid for cid, grade in rel.items() if grade >= 1}


def recall_at_k(ranked: Sequence[str], rel: Judgments, k: int) -> float | None:
    """前 k 名找回了多少比例的相關課程。沒有任何相關課程的題目回 None(不計分)。"""
    relevant = _relevant(rel)
    if not relevant:
        return None
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def mrr_at_k(ranked: Sequence[str], rel: Judgments, k: int) -> float | None:
    """第一門相關課程名次的倒數;前 k 名都沒有就是 0。"""
    relevant = _relevant(rel)
    if not relevant:
        return None
    for rank, cid in enumerate(ranked[:k], 1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(grades: Sequence[int]) -> float:
    return sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(grades, 1))


def ndcg_at_k(ranked: Sequence[str], rel: Judgments, k: int) -> float | None:
    """分級 nDCG@k:實際排序的 DCG 除以最理想排序的 DCG。"""
    ideal = _dcg(sorted((g for g in rel.values() if g > 0), reverse=True)[:k])
    if ideal == 0:
        return None
    return _dcg([rel.get(cid, 0) for cid in ranked[:k]]) / ideal


def judged_at_k(ranked: Sequence[str], rel: Mapping[str, object], k: int) -> float:
    """前 k 名裡已標註的比例。

    未標註的課程一律算不相關,所以這個比例偏低時,分數是被低估的。通常是改了檢索
    之後冒出標註池裡沒有的新課,要重跑 pool 補標。
    """
    top = ranked[:k]
    if not top:
        return 1.0
    return sum(cid in rel for cid in top) / len(top)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    strata: Sequence[str],
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """回 (平均差異 mean(a - b), 信賴區間下界, 上界)。

    成對:同一題的兩個分數綁在一起抽,差異逐題計算。題目本身的難易因此抵銷,
    這才是「同一組查詢比較兩個系統」該用的設計,而不是當成兩組獨立樣本。
    分層:每一層各自重抽、維持原本的題數。查詢集本來就是分層建的,整體一起抽的話,
    某次重抽可能幾乎抽不到教師題,信賴區間就混進了「題型比例」的變異。
    """
    if not (len(a) == len(b) == len(strata)):
        raise ValueError("a、b、strata 的長度必須相同")
    if not a:
        raise ValueError("沒有任何查詢可以比較")
    diffs = [x - y for x, y in zip(a, b)]
    groups: dict[str, list[float]] = {}
    for d, s in zip(diffs, strata):
        groups.setdefault(s, []).append(d)

    n = len(diffs)
    rng = random.Random(seed)
    boot = sorted(
        sum(sum(rng.choices(g, k=len(g))) for g in groups.values()) / n
        for _ in range(n_boot)
    )
    lo = boot[int(n_boot * alpha / 2)]
    hi = boot[int(n_boot * (1 - alpha / 2)) - 1]
    return sum(diffs) / n, lo, hi
