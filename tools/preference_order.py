"""preference_order_tool —— 查詢某門課「排第幾志願會上」的歷史資料。

政大選課採志願序分發:學生為每門課設定志願序,系統依身分別與志願序分發。
這個工具回答「這門課我該排第幾」,依據是兩份性質完全不同的歷史資料。

## 兩份資料刻意分開呈現,不合併統計

| | 官方 | Dcard |
|---|---|---|
| 來源 | 政大「體育及通識初選分發結果」 | 學生在 Dcard 政大板的回報 |
| 涵蓋 | 4 學期(113-1~114-2)、**只有通識與體育** | 27 學期(108-1~114-2)、全校各系 |
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

契約(見 tools/registry.py):扁平參數、回字串不 raise、無 LLM。
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from langchain_core.tools import tool

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
    if last_choice <= 2:
        return f"**極搶手**，只有排第 {last_choice} 志願以內的人選上"
    if last_choice <= 5:
        return f"偏搶手，要排在第 {last_choice} 志願以內"
    return f"**好選**，排到第 {last_choice} 志願都還上得了"


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


def _render_official(rows: list[dict]) -> list[str]:
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

    # 門檻要「分班次」講。同一個關鍵字可能比對到幾十個不同班次
    # (「羽球初級」有男/女/男女合班 × 十幾個時段),把它們的門檻聚合成單一範圍
    # 會得出「排 1 ~ 27」這種等於沒說的結論 —— 各班次的難度本來就天差地別。
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["is_full"] and r["last_choice"] is not None:
            groups[(r["course_name"], r.get("time"))].append(r)

    if len(groups) == 1:
        (name, time_str), g = next(iter(groups.items()))
        choices = sorted(r["last_choice"] for r in g)
        lo, hi = choices[0], choices[-1]
        n_unfull = sum(1 for r in rows if not r["is_full"])

        # **建議一律以最嚴的那次為準,不是最寬鬆的那次。**
        # 你要規劃的是下個學期,而下個學期可能剛好是最搶手的那一年。
        # 取 max(門檻) 會給出「排 11 都上得了」這種建議,但只要遇上像 114-2
        # 那種只到排 3 的學期就落榜 —— 這是先前的實作犯過的錯。
        out += ["", "### 建議", ""]
        out.append(f"- 額滿的 {len(g)} 個學期中，**最嚴的一次要排第 {lo} 志願以內**")
        if hi > lo:
            out.append(f"- 最寬鬆的一次到第 {hi} 志願都還有位，但不能指望每年都這樣")
        out.append(f"- **想穩:排第 {lo} 志願以內**"
                   + (f"；可接受風險:排第 {hi} 志願以內" if hi > lo else ""))
        if n_unfull:
            out.append(f"- 另有 {n_unfull} 個學期未招滿，那些年排幾都上")
    elif len(groups) > 1:
        out += ["", f"**比對到 {len(groups)} 個不同班次，門檻差異很大**"
                    "（額滿時該數字才是真門檻）：", ""]
        ranked = sorted(
            groups.items(),
            key=lambda kv: min(r["last_choice"] for r in kv[1]),
        )
        for (name, time_str), g in ranked[:8]:
            lo = min(r["last_choice"] for r in g)   # 最嚴的一次才是規劃依據
            out.append(f"- **{name} {time_str or ''}**：{_verdict(True, lo)}"
                       f"（{len(g)} 個學期額滿）")
        if len(ranked) > 8:
            out.append(f"- （另有 {len(ranked) - 8} 個班次，指定時段可查得更準）")
        out += ["", "**請指定時段或老師**才能得到有意義的參考 —— "
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


@tool
def preference_order_tool(
    course_name: str, teacher: str = "", course_id: str = "", time: str = ""
) -> str:
    """查詢某門課「志願序要排第幾才選得上」的歷史資料。

    什麼時候用:使用者問「這門課要排第幾」「排 3 有機會嗎」「志願序怎麼填」,
    或在看到課程清單後追問某門課的選上難度時。

    參數:
    - course_name: 課程名稱(必填,可部分比對,例如「羽球」)
    - teacher: 授課老師(可選,用來區分同名不同班)
    - course_id: 13 位課程代碼(可選)。從 retrieve_tool 的結果帶入可精準比對官方資料
    - time: 上課時段(可選,例如「二12」)。同一門課不同時段的難度天差地別,
      使用者若有指出時段就一定要帶入

    回傳:分成「官方分發結果」與「學生回報」兩段的 Markdown。
    兩份資料性質不同,**不要把它們平均或合併成單一結論**,照實分段轉述。
    官方只涵蓋通識與體育;其他課程只有學生回報,樣本少時要明講。
    """
    try:
        name = (course_name or "").strip()
        if not name and not course_id:
            return "ERROR: 請提供課程名稱。"

        official = _match_official(name, course_id)
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
                "冷門或新開的課通常沒有人回報。"
            )

        parts: list[str] = [f"## 「{name}」的志願序歷史資料", ""]
        if official:
            parts += _render_official(official) + [""]
        else:
            parts += ["### 官方分發結果", "",
                      "查無官方資料。官方統計只涵蓋**通識與體育**課程。", ""]
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
