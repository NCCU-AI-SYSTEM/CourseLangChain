"""志願序選上機率(存活分析)的測試。

跑:`uv run python -m tests.test_preference_survival`(在 CourseLangChain/ 下)

**不需要 LLM 與資料庫**:模型邏輯用假資料測,工具與 API 用 `data/preference_official.jsonl`。
"""
from __future__ import annotations

import math
import sys

from tools.preference_survival import (MAX_EXACT, MAX_RANK, VALIDATED_MAX_RANK, ChanceModel,
                                       TermResult, credit, describe, get_model, optimize_order,
                                       section_key)

_PASS = 0
_FAIL = 0


def check(cond: bool, msg: str) -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  ✓ {msg}")
    else:
        _FAIL += 1
        print(f"  ✗ FAIL: {msg}")


def row(term: str, code: str, time: str, full: bool, last: int | None, name: str = "通識課") -> dict:
    return {"term": term, "course_id": term + code, "time": time, "is_full": full, "last_choice": last,
            "course_name": name}


def section(code: str, time: str, lasts: list[int | None], name: str = "通識課") -> list[dict]:
    """一個班次三個學期的官方結果;lasts 裡的 None 代表那學期未額滿。"""
    return [row(term, code, time, last is not None, last if last is not None else 30, name)
            for term, last in zip(("1131", "1132", "1141"), lasts)]


def test_credit() -> None:
    print("[政大的規則:最後分發志願序 L = 第 L−1 以前必上、第 L 不一定上]")
    t = TermResult("1142", True, 3)
    check(credit(t, 2) == 1 and credit(t, 3) == 0.5 and credit(t, 4) == 0, "額滿 L=3:排 2 上、排 3 算一半、排 4 沒上")
    check(credit(TermResult("1142", False, 3), 99) == 1, "未額滿:排第幾都上(cure)")


def test_section_key() -> None:
    print("[班次識別:9 碼科目代號 + 時段]")
    check(section_key("1151002012011", "五34") == ("002012011", "五34"), "13 碼去掉學期")
    check(section_key("1131 002012011", "五 34") == ("002012011", "五34"), "空白一律去掉(PDF 換行)")
    check(section_key("002012011", "五34") == ("002012011", "五34"), "9 碼直接用")
    check(section_key("abc", "五34") is None and section_key(None, None) is None, "格式不對回 None")


def test_model() -> None:
    print("[收縮估計]")
    rows = [
        row("1131", "000000001", "一12", True, 2),
        row("1132", "000000001", "一12", True, 4),
        row("1131", "000000002", "二34", False, 10),
        row("1132", "000000003", "三56", True, None),  # 無資料:排除
    ]
    m = ChanceModel(rows)
    check(m.n_observations == 3, "last_choice 為空的列不算觀測")
    check(math.isclose(m.pooled(2), (0.5 + 1 + 1) / 3), "全體平均:三個學期排第 2 的上榜量平均")
    c = m.chance("1151000000001", "一12", 2)
    check(c is not None and math.isclose(c.probability, (0.5 + 1 + m.pooled(2)) / 3),
          "班次預測 = (自己的上榜量 + 1 × 全體平均) / (學期數 + 1)")
    check(c.strictest == 2 and c.full_terms == 2 and len(c.history) == 2, "歷史摘要:最嚴的一次、額滿學期數")
    check(m.chance("1151000000003", "三56", 1) is None, "只有無資料學期的班次不預測")
    check(m.chance("1151000000001", "二12", 1) is None, "同課號不同時段 = 不同班次,沒有歷史不預測")
    check(m.chance("1151999999999", "一12", 1) is None, "沒有官方資料的課不預測")
    check(m.chance("1151000000001", "一12", 0) is None and m.chance("1151000000001", "一12", MAX_RANK + 1) is None,
          "志願序超出 1~100 不預測")
    probs = [m.chance("1151000000001", "一12", r).probability for r in range(1, 30)]
    check(all(a >= b for a, b in zip(probs, probs[1:])), "機率隨志願序遞減")
    check(all(0 <= p <= 1 for p in probs), "機率在 0 到 1 之間")
    check(m.chance("1151000000001", "一12", VALIDATED_MAX_RANK + 1).validated is False, "第 10 志願以後標成未驗證")


def test_optimize() -> None:
    print("[志願序建議:期望選上門數最大]")
    hard_ge = section("000000001", "一12", [1, 1, 2])  # 搶手:只有第 1 志願有機會
    easy_ge = section("000000002", "二34", [None, None, None])  # 從沒額滿:排第幾都上
    hard_pe = section("000000003", "三56", [1, 1, 2], "體育[男女合班]—健走")
    easy_pe = section("000000004", "四78", [None, None, None], "體育[男女合班]—民俗體育")
    m = ChanceModel(hard_ge + easy_ge + hard_pe + easy_pe)
    ge_h, ge_e = ("1151000000001", "一12"), ("1151000000002", "二34")
    pe_h, pe_e = ("1151000000003", "三56"), ("1151000000004", "四78")
    nodata = ("1151999999999", "五12")
    check(m.is_pe(*pe_h) and not m.is_pe(*ge_h) and not m.is_pe(*nodata), "體育由官方課名判斷(以「體育」開頭)")

    plan = optimize_order(m, [ge_e, ge_h])
    check(plan.order == (1, 0) and plan.after > plan.before, "通識:搶手的放前面,本來就會上的放後面")
    check(plan.exact and plan.changed, "門數少時窮舉(精確解)")

    plan = optimize_order(m, [ge_h, ge_e])
    check(plan.order == (0, 1) and not plan.changed and math.isclose(plan.before, plan.after),
          "已經是最佳排法時不動")

    plan = optimize_order(m, [pe_e, pe_h])
    check(plan.after <= 1.0, f"體育最多算一門(兩個體育班的期望值 {plan.after:.2f} ≤ 1)")
    p_e1, p_h2 = m.chance(*pe_e, 1).probability, m.chance(*pe_h, 2).probability
    check(math.isclose(plan.before, 1 - (1 - p_e1) * (1 - p_h2)), "體育那一項 = 1 − 每班都沒上的機率相乘")

    plan = optimize_order(m, [ge_e, nodata, ge_h])
    check(plan.order[1] == 1 and plan.pinned == (1,), "沒有官方資料的課固定在原位,只重排有資料的課")
    check(plan.order == (2, 1, 0), "有資料的課在剩下的位置裡重排")

    many = [(f"11510000001{i:02d}", f"一{i}") for i in range(MAX_EXACT + 2)]
    rows = [r for i, (cid, t) in enumerate(many) for r in section(cid[4:], t, [None, None, None] if i % 2 else [1, 1, 2])]
    big = ChanceModel(rows)
    plan = optimize_order(big, many + [nodata])
    check(not plan.exact and plan.after >= plan.before and plan.order[-1] == len(many),
          f"超過 {MAX_EXACT} 門改局部搜尋,結果不比原本差,無資料的課仍在原位")


def test_describe() -> None:
    print("[白話結論]")
    check(describe(0.95) == "幾乎一定上" and describe(0.5) == "大約一半一半" and describe(0.05) == "幾乎上不了",
          "依機率分段")


def test_real_data() -> None:
    print("[真實資料:data/preference_official.jsonl]")
    from tools.preference_order import OFFICIAL_JSONL

    m = get_model(OFFICIAL_JSONL)
    check(m.n_observations > 1000, f"官方觀測數 {m.n_observations}")
    c = m.chance("1151002012011", "五34", 2)  # 體育[男]—羽球初級 五34
    check(c is not None and len(c.history) >= 1, "115-1 的課號能對到歷史班次")
    if c:
        hits = sum(credit(t, 2) for t in c.history)
        check(math.isclose(c.probability, (hits + m.pooled(2)) / (len(c.history) + 1)), "與公式一致")
    pooled = [m.pooled(r) for r in range(1, MAX_RANK + 1)]
    check(0 < min(pooled) and max(pooled) < 1,
          f"全體平均在 0 與 1 之間({min(pooled):.0%}~{max(pooled):.0%}),所以收縮後不會給出 0% 或 100%")


def test_tool() -> None:
    print("[preference_order_tool 的輸出]")
    from tools.preference_order import preference_order_tool as tool

    out = tool.invoke({"course_name": "羽球初級", "time": "五34", "rank": 3})
    check("### 選上機率" in out and "排第 3 志願：約" in out, "有官方資料的班次給機率,並回答問的那個志願")
    check("想穩:排第" not in out, "不再出現舊的「排第 L 以內就穩」建議")
    many = tool.invoke({"course_name": "羽球初級"})
    check("個不同班次" in many and "科目代號" in many, "多個班次時列出各班次並附科目代號")
    no = tool.invoke({"course_name": "微積分", "rank": 2})
    check("資料不足，不提供預測" in no and "%" not in no.split("### 學生回報")[0],
          "沒有官方資料的課(必修):說資料不足,官方段落不出現任何百分比")
    big = tool.invoke({"course_name": "羽球初級", "time": "五34", "rank": 500})
    check(not big.startswith("ERROR") and "### 選上機率" in big and "排第 500 志願" not in big,
          "rank 超出 1~100 時忽略那個志願,仍給機率表、不報錯")


def test_api() -> None:
    print("[POST /api/preference/chances]")
    from fastapi.testclient import TestClient

    import app as app_module

    client = TestClient(app_module.app)  # 不用 with:不觸發 lifespan,不需要資料庫
    r = client.post("/api/preference/chances", json={"items": [
        {"course_id": "1151002012011", "time": "五34", "rank": 1},
        {"course_id": "1151205064011", "time": "三D56", "rank": 1},  # 微積分:沒有官方資料
    ]})
    body = r.json()
    check(r.status_code == 200 and len(body["items"]) == 2, "回傳筆數與輸入相同")
    a, b = body["items"]
    check(a["available"] and 0 < a["probability"] < 1 and a["label"] and a["validated"], "有歷史的班次給機率與白話")
    check(b == {"course_id": "1151205064011", "rank": 1, "available": False}, "沒有歷史的課只回 available=false")
    check(client.post("/api/preference/chances", json={"items": "x"}).status_code == 400, "items 不是陣列回 400")
    check(client.post("/api/preference/chances", json={"items": [{"course_id": "x", "rank": "a"}]}).status_code == 400,
          "rank 不是整數回 400")

    print("[POST /api/preference/optimize]")
    demo = [("1151002380001", "二34"), ("1151045025001", "三78"), ("1151306013001", "四234"),
            ("1151993015001", "四D5"), ("1151002347021", "三12")]
    r = client.post("/api/preference/optimize", json={"items": [{"course_id": c, "time": t} for c, t in demo]})
    body = r.json()
    check(r.status_code == 200 and sorted(body["order"]) == sorted(c for c, _ in demo), "回傳的順序是同一批課")
    check(body["order"][2] == "1151306013001" and body["pinned"] == ["1151306013001"], "資料結構(無官方資料)固定在第 3")
    check(body["changed"] and body["after"] > body["before"] and body["exact"], "示範清單:建議排法的期望值較高")
    check(client.post("/api/preference/optimize", json={"items": []}).status_code == 400, "空清單回 400")


if __name__ == "__main__":
    for fn in (test_credit, test_section_key, test_model, test_optimize, test_describe, test_real_data,
               test_tool, test_api):
        fn()
    print(f"\n結果:{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)
