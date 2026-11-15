"""排第 r 志願選得上的機率 —— 官方分發結果的存活分析(執行期版本)。

方法與驗證在 CoursePreferenceData 的 `analysis/`(README 與 results/official_report.md)。
這裡只放預測需要的部分,刻意不共用程式碼:那邊是研究用的 repo,這裡是執行期。**改方法時兩邊一起改**,
並重跑那邊的時間外驗證。

## 志願序的語意(政大的規則)

- 填的數字只決定自己各志願的先後,系統會重新編成第 1、2、3…志願。所以 r 是「第幾志願」,不是填的數字。
- 官方的「最後分發志願序 L」:第 L − 1 志願以前必上;第 L 志願不一定上(同志願序的人比名額多時,有人上、
  有人沒上);第 L 志願之後上不了。

## 模型

一個觀測 = 一個班次(9 碼科目代號 + 時段)在一個學期的官方結果。排第 r 志願在那個學期的「上榜量」:

| 那個學期 | 上榜量 |
|---|---|
| 未額滿 | 1(排第幾都上;存活分析的 cure fraction) |
| 額滿,r < L | 1 |
| 額滿,r = L | 0.5(不知道上沒上的比例,算一半) |
| 額滿,r > L | 0 |

班次的預測 = (這個班次各學期的上榜量總和 + M × 全體平均) / (學期數 + M),M = 1。
一個班次最多只有 4 個學期,只用自己的歷史會給出 0% 或 100% 這種斷言;往全體平均收縮讓它借用全體的資訊。

時間外驗證(用 113-1 到 114-1 預測 114-2,204 個班次、第 1 到第 10 志願):Brier 0.096,
2026-11 以前工具用的門檻規則是 0.125,差 −0.029,95% 信賴區間 [−0.048, −0.012]。
**第 10 志願以後沒有驗證過**,輸出時要註明。

## 不預測的情況

- 沒有官方資料的課(必修、選修、通識與體育以外的課):只有學生在 Dcard 的自述,樣本少、可能記錯,
  而且上榜與落榜的回報要用區間設限的方法才能合併,還沒有做。照實說「資料不足」。
- 有官方資料但這個時段的班次從沒出現過:班次之間差異太大,全體平均預測不了個別班次(驗證 Brier 0.215)。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path

TIE = 0.5  # 排第 L 志願(最後分發志願序)算多少個上榜
M = 1.0  # 收縮強度:全體平均相當於幾個學期
VALIDATED_MAX_RANK = 10  # 時間外驗證涵蓋到第幾志願
MAX_RANK = 100  # 選課系統的志願序上限


@dataclass(frozen=True)
class TermResult:
    """一個班次在一個學期的官方結果。"""

    term: str  # 例:"1142"
    full: bool
    last: int  # 最後分發志願序 L


@dataclass(frozen=True)
class Chance:
    rank: int
    probability: float
    history: tuple[TermResult, ...]  # 這個班次的歷史,依學期排序

    @property
    def validated(self) -> bool:
        return self.rank <= VALIDATED_MAX_RANK

    @property
    def full_terms(self) -> int:
        return sum(t.full for t in self.history)

    @property
    def strictest(self) -> int | None:
        """額滿學期裡最小的最後分發志願序;從沒額滿過回 None。"""
        full = [t.last for t in self.history if t.full]
        return min(full) if full else None


def credit(t: TermResult, r: int, tie: float = TIE) -> float:
    """排第 r 志願在這個學期選上的程度:1 上、0 沒上,排第 L 志願算 tie。"""
    if not t.full or r < t.last:
        return 1.0
    return tie if r == t.last else 0.0


def section_key(course_id: str | None, time: str | None) -> tuple[str, str] | None:
    """班次識別:9 碼科目代號 + 時段。13 碼課號去掉 4 碼學期;空白一律去掉(PDF 換行會插入空白)。"""
    code = re.sub(r"\s", "", course_id or "")
    if len(code) == 13:
        code = code[4:]
    if len(code) != 9:
        return None
    return code, re.sub(r"\s", "", time or "")


def describe(p: float) -> str:
    """把機率翻成白話。界線刻意保守:校準顯示高機率區實際比預測更容易上。"""
    if p >= 0.9:
        return "幾乎一定上"
    if p >= 0.7:
        return "很有機會"
    if p >= 0.4:
        return "大約一半一半"
    if p >= 0.15:
        return "機會不大"
    return "幾乎上不了"


class ChanceModel:
    """從官方分發結果建立的預測模型。資料只讀一次;全體平均依志願序快取。"""

    def __init__(self, rows: list[dict]) -> None:
        self._history: dict[tuple[str, str], list[TermResult]] = {}
        self._names: dict[tuple[str, str], str] = {}
        self._all: list[TermResult] = []
        for row in rows:
            if row.get("last_choice") is None:  # 沒人選或無資料
                continue
            key = section_key(row.get("course_id"), row.get("time"))
            if key is None:
                continue
            t = TermResult(str(row["term"]), bool(row["is_full"]), int(row["last_choice"]))
            self._history.setdefault(key, []).append(t)
            self._all.append(t)
            # 課名去掉 PDF 換行插入的空白;只用來判斷是不是體育
            self._names[key] = re.sub(r"\s", "", row.get("course_name") or "")
        for terms in self._history.values():
            terms.sort(key=lambda t: t.term)
        self._pooled: dict[int, float] = {}

    @property
    def n_observations(self) -> int:
        return len(self._all)

    def pooled(self, r: int) -> float:
        """全體平均:所有班次、所有學期排第 r 志願的平均上榜量。"""
        if r not in self._pooled:
            self._pooled[r] = (sum(credit(t, r) for t in self._all) / len(self._all)
                               if self._all else 0.0)
        return self._pooled[r]

    def history(self, course_id: str | None, time: str | None) -> tuple[TermResult, ...]:
        key = section_key(course_id, time)
        return tuple(self._history.get(key, ())) if key else ()

    def chance(self, course_id: str | None, time: str | None, rank: int) -> Chance | None:
        """排第 rank 志願選上的機率;這個班次沒有官方歷史就回 None(不預測)。"""
        if not 1 <= rank <= MAX_RANK:
            return None
        hist = self.history(course_id, time)
        if not hist:
            return None
        hits = sum(credit(t, rank) for t in hist)
        p = (hits + M * self.pooled(rank)) / (len(hist) + M)
        return Chance(rank, p, hist)

    def is_pe(self, course_id: str | None, time: str | None) -> bool:
        """官方資料裡這個班次是不是體育課(課名以「體育」開頭)。"""
        key = section_key(course_id, time)
        return bool(key) and self._names.get(key, "").startswith(PE_PREFIX)


# ── 志願序建議:期望選上門數最大的排法 ─────────────────────────────────────────
#
# 政大的規則(2026-11-15 使用者說明):
# - 通識沒有上限:各門互相獨立,期望選上門數直接相加。
# - 體育一次最多上一門:一門選上,排在後面的體育都直接不分發。所以體育只算
#   「至少上一門」的機率 = 1 − Π(每個體育班都沒上的機率)。
# - 清單是「前面最想要、後面是備案」;建議只看機率,不知道使用者比較想上哪一門,
#   所以只給建議,由使用者決定要不要套用。
# - 沒有官方資料的課(必修等)沒有機率,固定在原本的位置,只重排有資料的課。
#
# 體育那一項是相乘,不是線性的指派問題;有資料的課通常不超過 8 門,
# 8! = 40,320 種排法全部算過也在一秒內,所以直接窮舉求精確解;更多門時改用兩兩交換的局部搜尋。

PE_PREFIX = "體育"
MAX_EXACT = 8  # 有資料的課在這個門數以內窮舉(精確解),超過改局部搜尋


@dataclass(frozen=True)
class OrderPlan:
    order: tuple[int, ...]  # 建議順序:原清單的索引
    before: float  # 目前順序的期望選上門數
    after: float  # 建議順序的期望選上門數
    exact: bool  # True = 窮舉的精確解;False = 局部搜尋
    pinned: tuple[int, ...]  # 沒有官方資料、位置不動的索引

    @property
    def changed(self) -> bool:
        return self.order != tuple(range(len(self.order)))


def optimize_order(model: ChanceModel, items: list[tuple[str, str]]) -> OrderPlan:
    """items 是目前清單的 (課號, 時段),依序為第 1、2、3…志願。回傳期望選上門數最大的排法。

    只有嚴格更好才換:與目前順序同分時保留使用者原本的順序。
    """
    n = len(items)
    has_data = [bool(model.history(cid, t)) for cid, t in items]
    pe = [model.is_pe(cid, t) for cid, t in items]
    # 先把「第 i 門課排在第 r 志願」的機率算成表,窮舉時只查表
    prob = [[model.chance(cid, t, r).probability if has_data[i] else 0.0 for r in range(1, n + 1)]
            for i, (cid, t) in enumerate(items)]

    def value(order: tuple[int, ...] | list[int]) -> float:
        ge, miss, any_pe = 0.0, 1.0, False
        for slot, i in enumerate(order):
            if not has_data[i]:
                continue
            p = prob[i][slot]
            if pe[i]:
                any_pe = True
                miss *= 1 - p
            else:
                ge += p
        return ge + (1 - miss if any_pe else 0.0)

    current = tuple(range(n))
    pinned = tuple(i for i in range(n) if not has_data[i])
    free = [i for i in range(n) if has_data[i]]  # 可以重排的位置(= 有資料的課原本的位置)
    best, best_v = current, value(current)

    if len(free) <= MAX_EXACT:
        for perm in permutations(free):
            order = list(current)
            for slot, i in zip(free, perm):
                order[slot] = i
            v = value(order)
            if v > best_v + 1e-12:
                best, best_v = tuple(order), v
        exact = True
    else:
        improved = True
        while improved:
            improved = False
            for a, b in combinations(free, 2):
                order = list(best)
                order[a], order[b] = order[b], order[a]
                v = value(order)
                if v > best_v + 1e-12:
                    best, best_v, improved = tuple(order), v, True
        exact = False
    return OrderPlan(best, value(current), best_v, exact, pinned)


_models: dict[str, ChanceModel] = {}


def get_model(path: Path) -> ChanceModel:
    """讀官方分發結果的 JSONL 建模型,結果快取在進程內。檔案不存在時是空模型(一律不預測)。"""
    key = str(path)
    if key not in _models:
        rows: list[dict] = []
        if path.exists():
            with path.open(encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        _models[key] = ChanceModel(rows)
    return _models[key]
