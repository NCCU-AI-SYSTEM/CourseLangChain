"""preference_order_tool —— 查詢某門課「排第幾志願會上」的歷史資料。

政大選課採志願序分發:學生為每門課設定志願序,系統依身分別與志願序分發。
這個工具回答「這門課我該排第幾」,依據是兩份性質完全不同的歷史資料。

## 兩份資料刻意分開呈現,不合併統計

| | 官方 | Dcard |
|---|---|---|
| 來源 | 政大「體育及通識初選分發結果」 | 學生在 Dcard 政大板的回報 |
| 涵蓋 | 4 學期(113-1~114-2)、**只有通識與體育** | 14 學期(108-1~114-2,27 份貼文=學期×選課階段)、全校各系 |
| 數字 | **最後分發上學生的志願序**(母體統計) | 某位同學「排 N 上了/沒上」(自述) |
| 筆數 | 1,282 | 4,985 |

官方是統計結果、Dcard 是零散樣本。平均在一起會讓權威資料被雜訊稀釋,
而使用者無從分辨哪一句可信 —— 所以輸出時分兩段,各自標明來源與限制。

## 官方數字要搭配「是否額滿」才有意義

`最後分發上學生志願序` 單看會誤導:

- **額滿**(選上人數 = 名額)→ 這個志願序就是真門檻,排在它之後的都沒上
- **未額滿** → 只反映「剛好有人排到那麼後面」,實際門檻其實更寬

實例:同樣是「羽球初級」,五34 額滿且門檻是排 2,二12 額滿但排到 27 都上 ——
不看時段與額滿狀態,這兩個數字放在一起毫無意義。

政大的規則(2026-11-15 確認):填的數字只決定先後,系統會重新編成第 1、2、3…志願;
「最後分發志願序 L」= 第 L − 1 志願以前必上、**第 L 志願不一定上**。

## 選上機率(存活分析)

有官方資料的班次,另外給「排第 r 志願選得上的機率」,方法與驗證見 tools/preference_survival.py。
**沒有官方資料的課(必修、選修等)不給機率**,只列學生回報並說明資料不足 —— 自述樣本太少,估出來的數字
會讓人誤以為有根據。

契約(見 tools/registry.py):扁平參數、回字串不 raise、無 LLM。
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from langchain_core.tools import tool

from .preference_survival import VALIDATED_MAX_RANK, describe, get_model, section_key
from .registry import register_tool

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OFFICIAL_JSONL = DATA_DIR / "preference_official.jsonl"
DCARD_JSONL = DATA_DIR / "preference_dcard.jsonl"

# 官方文件自己的免責聲明,轉述時必須一併帶上
OFFICIAL_DISCLAIMER = (
    "官方聲明:本資料僅供學生設定科目志願序之參考,並不表示學生依此志願序設定,"
    "一定能分發選上課。"
)

# 一次最多列幾筆明細。超過就只給摘要 —— 全部倒出來會塞爆 context,
# 而且使用者要的是判斷依據,不是原始資料傾印。
MAX_ROWS = 12

_cache: dict[str, list[dict]] = {}


def _load(path: Path) -> list[dict]:
    """讀 JSONL,結果快取在進程內。檔案不存在時回空清單而不是拋例外。"""
    key = str(path)
    if key not in _cache:
        if not path.exists():
            _cache[key] = []
        else:
            with path.open(encoding="utf-8") as f:
                _cache[key] = [json.loads(line) for line in f if line.strip()]
    return _cache[key]


def _verdict(is_full: bool, last_choice: int | None) -> str:
    """把「最後分發志願序」翻成白話結論。

    這個數字的語意極反直覺,**必須由工具講結論,不能丟給模型自己推**:
    額滿且數字大 = 好選(排到那麼後面都還有位),額滿且數字小 = 極搶手。
    實測 qwen2:7b 會把「額滿、最後分發志願序 27」讀成「需要較高的志願序才有機會」,
    結論與事實完全相反 —— 那不是模型笨,是我們給了它需要二階推理的原始數字。
    """
    if last_choice is None:
        return "無分發資料（可能沒人選）"
    if not is_full:
        return f"未招滿，排第 {last_choice} 志願的人也上了，通常不必排前面"
    # 最後分發志願序 L:第 L − 1 以前必上,第 L 不一定上
    if last_choice == 1:
        return "**極搶手**，連第 1 志願都不一定上"
    if last_choice <= 2:
        return "**極搶手**，第 1 志願才穩，第 2 志願不一定上"
    if last_choice <= 5:
        return f"偏搶手，第 {last_choice - 1} 志願以內才穩，第 {last_choice} 志願不一定上"
    return f"**好選**，排到第 {last_choice - 1} 志願都上得了"


def _fmt_term(term: str) -> str:
    """1142 → 114-2"""
    return f"{term[:3]}-{term[3:]}" if len(term) == 4 else term


def _subject_code(course_id: str) -> str:
    """13 碼 course_id 去掉學期前綴,留 9 碼科目代號 —— 用它跨學期比對同一門課。"""
    cid = re.sub(r"\D", "", course_id or "")
    return cid[4:] if len(cid) == 13 else ""


def _match_official(course_name: str, course_id: str) -> list[dict]:
    rows = _load(OFFICIAL_JSONL)
    code = _subject_code(course_id)
    if code:
        hit = [r for r in rows if _subject_code(r.get("course_id", "")) == code]
        if hit:
            return hit
    if not course_name:
        return []
    return [r for r in rows if course_name in (r.get("course_name") or "")]


def _match_dcard(course_name: str, teacher: str) -> list[dict]:
    """Dcard 沒有課號,只能用課名比對;老師有給就再收斂。"""
    if not course_name:
        return []
    rows = [r for r in _load(DCARD_JSONL) if course_name in (r.get("course_name") or "")]
    if teacher:
        narrowed = [r for r in rows if teacher in (r.get("teacher") or "")]
        if narrowed:
            rows = narrowed
    return rows


# 機率表固定列這幾個志願;使用者問的那一個另外加進去
_SHOW_RANKS = (1, 2, 3, 5, 8, 10)
# 依機率給建議的兩條線
_SURE, _EVEN = 0.8, 0.5


def _section_label(rows: list[dict]) -> str:
    """班次名稱取最近一學期;去掉 PDF 換行插入的空白(「彼拉提斯墊上 技巧」)。"""
    latest = max(rows, key=lambda r: r["term"])
    name = re.sub(r"\s", "", latest.get("course_name") or "")
    key = section_key(latest.get("course_id"), latest.get("time"))
    code = f"（科目代號 {key[0]}）" if key else ""  # 同時段可能有兩個班,課名相同
    return f"{name} {latest.get('time') or ''}{code}".strip()


def _render_chance(section_rows: list[dict], rank: int) -> list[str]:
    """單一班次的選上機率。沒有可用的歷史時回空清單。"""
    model = get_model(OFFICIAL_JSONL)
    cid, time = section_rows[0].get("course_id"), section_rows[0].get("time")
    ranks = sorted(set(_SHOW_RANKS) | ({rank} if rank else set()))
    chances = {r: model.chance(cid, time, r) for r in ranks}
    if not all(chances.values()):
        return []
    any_chance = chances[ranks[0]]

    out = ["### 選上機率（依官方分發結果估計）", "",
           "| 第幾志願 | " + " | ".join(str(r) for r in ranks) + " |",
           "|---" * (len(ranks) + 1) + "|",
           "| 選上機率 | " + " | ".join(f"{chances[r].probability:.0%}" for r in ranks) + " |", ""]
    if rank:
        c = chances[rank]
        out.append(f"- **排第 {rank} 志願：約 {c.probability:.0%}，{describe(c.probability)}**")
        if not c.validated:
            out.append(f"- 第 {VALIDATED_MAX_RANK} 志願以後的估計沒有經過驗證，僅供參考")

    # 機率隨志願序遞減,所以「最後一個還達標的志願」就是建議的上限
    probs = [(r, model.chance(cid, time, r).probability) for r in range(1, VALIDATED_MAX_RANK + 1)]
    for line, label in ((_SURE, "很有把握"), (_EVEN, "一半以上機會")):
        ok = [r for r, p in probs if p >= line]
        if not ok:
            out.append(f"- 連第 1 志願都不到 {line:.0%}（{label}的門檻）")
        elif ok[-1] == VALIDATED_MAX_RANK:
            out.append(f"- 排到第 {VALIDATED_MAX_RANK} 志願都還有 {line:.0%} 以上（{label}）")
        else:
            out.append(f"- {label}（≥ {line:.0%}）：排第 {ok[-1]} 志願以內")

    n, n_full = len(any_chance.history), any_chance.full_terms
    out += [f"- 依據：這個班次 {n} 個學期的官方結果（{n_full} 個學期額滿），"
            f"再往全體 {model.n_observations:,} 筆的平均收縮；歷史越少，越接近全體平均",
            "- 機率指的是「第幾志願」：填的數字只決定先後，系統會重新編成第 1、2、3…志願"]
    return out


def _render_official(rows: list[dict], rank: int = 0) -> list[str]:
    out = ["### 官方分發結果（權威）", ""]
    rows = sorted(rows, key=lambda r: (r["term"], r.get("time") or ""), reverse=True)
    out += ["| 學期 | 課名 | 時段 | 名額 | 選上 | 最後分發志願序 | 解讀 |",
            "|------|------|------|------|------|----------------|------|"]
    for r in rows[:MAX_ROWS]:
        choice = r["last_choice"] if r["last_choice"] is not None else "—"
        out.append(
            f"| {_fmt_term(r['term'])} | {r['course_name']} | {r.get('time') or '—'} "
            f"| {r['capacity']} | {r['enrolled']} | {choice} | {_verdict(r['is_full'], r['last_choice'])} |"
        )
    if len(rows) > MAX_ROWS:
        out.append(f"\n（另有 {len(rows) - MAX_ROWS} 筆未列出）")

    # 要「分班次」講。同一個關鍵字可能比對到幾十個不同班次
    # (「羽球初級」有男/女/男女合班 × 十幾個時段),把它們的門檻聚合成單一範圍
    # 會得出「排 1 ~ 27」這種等於沒說的結論 —— 各班次的難度本來就天差地別。
    # 班次以「9 碼科目代號 + 時段」識別,與 preference_survival 相同(課名會被 PDF 換行插入空白)。
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = section_key(r.get("course_id"), r.get("time"))
        if key and r["last_choice"] is not None:
            groups[key].append(r)

    if len(groups) == 1:
        g = next(iter(groups.values()))
        full = sorted(r["last_choice"] for r in g if r["is_full"])
        n_unfull = sum(1 for r in g if not r["is_full"])

        # 歷史事實照列,但**以最嚴的那次為準**:要規劃的是下個學期,而下個學期可能剛好是
        # 最搶手的那一年。建議改由下面的機率段落給(排第 L 志願不一定上,舊的「排第 L 以內就穩」是錯的)。
        out += ["", "### 歷年門檻", ""]
        if full:
            lo, hi = full[0], full[-1]
            out.append(f"- 額滿的 {len(full)} 個學期中，最嚴的一次：{_verdict(True, lo)}")
            if hi > lo:
                out.append(f"- 最寬鬆的一次：{_verdict(True, hi)}，但不能指望每年都這樣")
        if n_unfull:
            out.append(f"- {n_unfull} 個學期未招滿，那些年排幾都上")
        chance = _render_chance(g, rank)
        if chance:
            out += [""] + chance
    elif len(groups) > 1:
        model = get_model(OFFICIAL_JSONL)
        at = rank or 3  # 沒指定志願序時,用第 3 志願比較各班次的難度
        scored = []
        for g in groups.values():
            c = model.chance(g[0].get("course_id"), g[0].get("time"), at)
            scored.append((c.probability if c else 1.0, g))
        scored.sort(key=lambda x: x[0])  # 最難上的排前面
        out += ["", f"**比對到 {len(groups)} 個不同班次，難度差異很大**"
                    f"（依「排第 {at} 志願的選上機率」由難到易）：", ""]
        for p, g in scored[:8]:
            full = [r["last_choice"] for r in g if r["is_full"]]
            fact = _verdict(True, min(full)) if full else "從沒額滿過，排幾都上"
            out.append(f"- **{_section_label(g)}**：排第 {at} 志願約 {p:.0%}；最嚴的一次：{fact}")
        if len(scored) > 8:
            out.append(f"- （另有 {len(scored) - 8} 個班次，指定時段可查得更準）")
        out += ["", "**請指定時段**才能得到這個班次完整的機率表 —— "
                    "同一門課不同班次的門檻可能從排 1 到排 27。"]
    out += ["", OFFICIAL_DISCLAIMER]
    return out


def _render_dcard(rows: list[dict]) -> list[str]:
    out = ["### 學生回報（Dcard，自述資料）", ""]
    got = [r for r in rows if r["result"] in ("上", "有上", "一階未滿", "二階未滿")]
    lost = [r for r in rows if r["result"] in ("沒上", "未上")]

    if got:
        mx = max(r["choice"] for r in got if r["choice"])
        out.append(f"- 有人排到 **第 {mx} 志願仍上榜**（共 {len(got)} 筆上榜回報）")
    if lost:
        mn = min(r["choice"] for r in lost if r["choice"])
        out.append(f"- 也有人排 **第 {mn} 志願卻沒上**（共 {len(lost)} 筆落榜回報）")
    if got and lost:
        out.append("- 兩者重疊代表**身分別的影響大於志願序**，同樣的志願序結果可能不同")
    out.append("")

    out += ["| 學期 | 階段 | 時段 | 老師 | 身分別 | 志願序 | 結果 |",
            "|------|------|------|------|--------|--------|------|"]
    for r in sorted(rows, key=lambda r: (r["term"], r["stage"]), reverse=True)[:MAX_ROWS]:
        out.append(
            f"| {_fmt_term(r['term'])} | {r['stage']}階 | {r.get('time') or '—'} "
            f"| {r.get('teacher') or '—'} | {r.get('identity') or '—'} "
            f"| {r['choice'] or '—'} | {r['result'] or '—'} |"
        )
    if len(rows) > MAX_ROWS:
        out.append(f"\n（共 {len(rows)} 筆，僅列出最近 {MAX_ROWS} 筆）")
    out += ["", "以上為學生自行回報，可能有誤記或遺漏，樣本也未必涵蓋你的身分別，僅供參考。"]
    return out


def _dcard_is_redundant(official: list[dict], dcard: list[dict]) -> bool:
    """官方門檻已經比所有學生回報都嚴時,學生回報沒有增加資訊。

    例:官方最嚴門檻是排 3,而學生回報「排 5 上榜」—— 排 5 比 3 寬鬆,
    本來就會上,拿它當獨立證據會讓結論失準(實測模型犯過這個錯)。
    """
    full = [r["last_choice"] for r in official if r["is_full"] and r["last_choice"]]
    got = [r["choice"] for r in dcard
           if r["choice"] and r["result"] in ("上", "有上", "一階未滿", "二階未滿")]
    return bool(full and got and min(full) <= min(got))


_NO_CHANCE = ("**選上機率：資料不足，不提供預測。** 官方分發結果只涵蓋通識與體育；"
              "這門課只有學生自述的回報，樣本少、可能記錯，估出來的數字沒有根據。")


@tool
def preference_order_tool(
    course_name: str, teacher: str = "", course_id: str = "", time: str = "", rank: int = 0
) -> str:
    """查詢某門課「志願序要排第幾才選得上」的歷史資料與選上機率。

    什麼時候用:使用者問「這門課要排第幾」「排 3 有機會嗎」「志願序怎麼填」,
    或在看到課程清單後追問某門課的選上難度時。

    參數:
    - course_name: 課程名稱(必填,可部分比對,例如「羽球」)
    - teacher: 授課老師(可選,用來區分同名不同班)
    - course_id: 13 位課程代碼(可選)。從 retrieve_tool 的結果帶入可精準比對官方資料
    - time: 上課時段(可選,例如「二12」)。同一門課不同時段的難度天差地別,
      使用者若有指出時段就一定要帶入
    - rank: 使用者打算排第幾志願(可選,1~100)。問「排 3 有機會嗎」就填 3;沒提到就留 0

    回傳:分成「官方分發結果」與「學生回報」兩段的 Markdown。有官方資料的班次另附「選上機率」。
    兩份資料性質不同,**不要把它們平均或合併成單一結論**,照實分段轉述。
    官方只涵蓋通識與體育;其他課程只有學生回報,**工具會註明資料不足、不提供機率**,不要自行推估。
    """
    try:
        name = (course_name or "").strip()
        if not name and not course_id:
            return "ERROR: 請提供課程名稱。"
        try:
            rank = int(rank or 0)
        except (TypeError, ValueError):
            rank = 0
        if not 1 <= rank <= 100:
            rank = 0

        official = _match_official(name, course_id)
        has_official_course = bool(official)  # 時段過濾之前:官方資料裡有沒有這門課
        dcard = _match_dcard(name, (teacher or "").strip())
        # 時段是最有效的收斂條件:同一門課不同時段門檻可以差到 26 個志願序
        slot = (time or "").strip()
        if slot:
            o2 = [r for r in official if slot in (r.get("time") or "")]
            d2 = [r for r in dcard if slot in (r.get("time") or "")]
            if o2 or d2:
                official, dcard = o2, d2

        if not official and not dcard:
            return (
                f"查無「{name}」的志願序歷史資料。\n\n"
                "可能原因:官方資料只涵蓋**通識與體育**(113-1 ~ 114-2),"
                "其餘課程僅有 Dcard 學生回報,而學生回報集中在熱門課程,"
                "冷門或新開的課通常沒有人回報。\n\n**選上機率：沒有任何歷史資料，不提供預測。**"
            )

        parts: list[str] = [f"## 「{name}」的志願序歷史資料", ""]
        if official:
            parts += _render_official(official, rank) + [""]
        elif has_official_course:
            # 課在官方資料裡,只是這個時段沒有班次紀錄(新開或改時段)。不能用別的時段代替:
            # 同一門課不同時段的門檻可以從排 1 差到排 27。
            parts += ["### 官方分發結果", "",
                      f"官方資料有這門課，但沒有「{slot}」這個時段的班次紀錄（可能是新開或改時段的班次）。",
                      "", "**選上機率：這個班次沒有歷史，不提供預測。**"
                      "不指定時段再查一次，可以看到這門課其他時段的難度。", ""]
        else:
            parts += ["### 官方分發結果", "",
                      "查無官方資料。官方統計只涵蓋**通識與體育**課程。", "", _NO_CHANCE, ""]
        if dcard:
            if official and _dcard_is_redundant(official, dcard):
                parts += ["> 註:官方門檻比所有學生回報都嚴格，以下僅供佐證，"
                          "**請以官方數字為準**。", ""]
            parts += _render_dcard(dcard)
        else:
            parts += ["### 學生回報（Dcard）", "", "查無學生回報。"]
        return "\n".join(parts)
    except Exception as e:  # noqa: BLE001 — 契約:回字串而非 raise
        return f"ERROR: 查詢志願序資料時發生錯誤({e})。"


register_tool(preference_order_tool)
