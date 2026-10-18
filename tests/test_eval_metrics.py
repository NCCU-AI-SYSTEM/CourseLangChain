"""離線檢索評估(evaluation/)的純函數測試,不需要資料庫。

跑:`uv run python -m tests.test_eval_metrics`(在 CourseLangChain/ 下)

驗三件事:
1. 指標與手算一致:Recall、MRR、分級 nDCG、標註涵蓋率;同分時排序固定。
2. 分層成對 bootstrap:兩系統相同時差異為 0、固定差異時區間退化成一點、
   分層會維持各層題數、同一個種子結果相同。
3. 標註表的防呆與抽題:Excel 把課號轉成科學記號時要大聲失敗;規則層抽題可重現、
   佔位字不當人名、相關課程太多的老師不抽。
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

from evaluation.build_gold import MAX_RELEVANT, build_rule_queries
from evaluation.data import JUDGMENT_FIELDS, check_course_id, parse_grade, read_tsv, write_tsv
from evaluation.metrics import (
    judged_at_k,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap,
    recall_at_k,
    stable_order,
)

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


def close(a: float | None, b: float, tol: float = 1e-9) -> bool:
    return a is not None and abs(a - b) < tol


def raises(fn, contains: str = "") -> bool:
    try:
        fn()
    except ValueError as e:
        return contains in str(e)
    return False


def test_stable_order() -> None:
    print("[同分排序固定]")
    scored = [("b", 1.0), ("a", 1.0), ("c", 2.0), ("a", 0.5)]
    check(stable_order(scored, 3) == ["c", "a", "b"], "分數高者在前、同分依課號、重複的課取最高分")
    check(stable_order(scored, 1) == ["c"], "只取前 top_k 門")


def test_recall_mrr() -> None:
    print("[Recall 與 MRR]")
    rel = {"a": 2, "d": 1, "z": 0}
    check(close(recall_at_k(["a", "b", "c"], rel, 3), 0.5), "兩門相關找回一門 = 0.5(grade 0 不算相關)")
    check(close(recall_at_k(["x", "y"], rel, 2), 0.0), "一門都沒找到 = 0")
    check(recall_at_k(["a"], {"a": 0}, 1) is None, "沒有相關課程的題目回 None,不計分")
    check(close(mrr_at_k(["b", "a"], {"a": 2}, 10), 0.5), "第一門相關在第 2 名 = 1/2")
    check(close(mrr_at_k(["b", "a"], {"a": 2}, 1), 0.0), "相關課程落在 k 之外 = 0")


def test_ndcg() -> None:
    print("[分級 nDCG]")
    check(close(ndcg_at_k(["a"], {"a": 2}, 10), 1.0), "理想排序 = 1")
    check(close(ndcg_at_k(["b", "a"], {"a": 2}, 10), 1 / math.log2(3)), "相關課排第 2 = 1/log2(3)")
    expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    check(close(ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 10), expected),
          "部分相關排在直接符合前面會被扣分(分級增益 2^g - 1)")
    check(ndcg_at_k(["a"], {"a": 0}, 10) is None, "沒有相關課程回 None")


def test_judged() -> None:
    print("[標註涵蓋率]")
    check(close(judged_at_k(["a", "b", "c", "d"], {"a": 0, "c": 2}, 4), 0.5), "前 4 名標過 2 門 = 0.5")
    check(close(judged_at_k([], {}, 10), 1.0), "沒有結果時視為完整")


def test_bootstrap() -> None:
    print("[分層成對 bootstrap]")
    b = [0.5, 0.2, 0.9, 0.4]
    strata = ["x", "x", "y", "y"]
    d, lo, hi = paired_bootstrap(b, b, strata, n_boot=500)
    check(d == lo == hi == 0.0, "兩系統完全相同:差異與區間都是 0")

    a = [x + 0.1 for x in b]
    d, lo, hi = paired_bootstrap(a, b, strata, n_boot=500)
    check(close(d, 0.1, 1e-9) and close(lo, 0.1, 1e-9) and close(hi, 0.1, 1e-9),
          "每題都差 0.1:區間退化成 0.1 一點")

    d, lo, hi = paired_bootstrap([1, 1, 1, 0], [0, 0, 0, 0], ["x", "x", "x", "y"], n_boot=500)
    check(close(d, 0.75) and close(lo, 0.75) and close(hi, 0.75),
          "層內數值相同時每次重抽都是 0.75:分層有維持各層題數")

    a2, b2 = [0.9, 0.1, 0.7, 0.3, 0.8], [0.2, 0.4, 0.1, 0.6, 0.5]
    s2 = ["x"] * 5
    check(paired_bootstrap(a2, b2, s2, 1000, seed=7) == paired_bootstrap(a2, b2, s2, 1000, seed=7),
          "同一個種子,結果完全相同")
    check(raises(lambda: paired_bootstrap([1.0], [1.0, 2.0], ["x"]), "長度"), "長度不一致會報錯")


def test_data_guards() -> None:
    print("[標註表防呆]")
    check(check_course_id(" 1151000711002 ", "t") == "1151000711002", "13 碼課號通過並去除空白")
    check(check_course_id("1151651C19001", "t") == "1151651C19001", "科目代號含大寫字母的課號也合法")
    check(raises(lambda: check_course_id("115165", "t"), "4 碼學年學期"), "長度不對時說明正確格式,而不是怪 Excel")
    check(raises(lambda: check_course_id("1.151E+12", "t"), "科學記號"), "被 Excel 轉成科學記號會報錯並說明原因")
    check(parse_grade("", "t") is None and parse_grade(" 2 ", "t") == 2, "空白 = 未標註;數字可帶空白")
    check(raises(lambda: parse_grade("3", "t"), "0、1、2"), "grade 只接受 0、1、2")

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "j.tsv"
        write_tsv(path, JUDGMENT_FIELDS,
                  [{"qid": "P01", "course_id": "1151000711002", "grade": "",
                    "snippet": "第一行\t有 tab\n第二行"}])
        rows = read_tsv(path)
        check(path.read_bytes().startswith(b"\xef\xbb\xbf"), "寫檔帶 UTF-8 BOM(Excel 開中文不亂碼)")
        check(rows[0]["qid"] == "P01", "讀檔時 BOM 不會黏在第一個欄名上")
        check(rows[0]["snippet"] == "第一行 有 tab 第二行", "tab 與換行被收成空白,不會弄壞欄位")


def test_rule_sampling() -> None:
    print("[規則層抽題]")
    courses = [
        ("1151000000001", "統計學", "王小明"),
        ("1151000000002", "統計學", "王小明、李大華"),
        ("1151000000003", "資料庫系統", "李大華"),
        ("1151000000004", "微積分", "待聘"),
    ]
    courses += [(f"11510000001{i:02d}", f"專題{i:02d}", "陳一") for i in range(MAX_RELEVANT + 1)]
    q1, j1 = build_rule_queries(courses, n_teacher=10, n_title=2, seed=1)
    q2, j2 = build_rule_queries(courses, n_teacher=10, n_title=2, seed=1)
    check(q1 == q2 and j1 == j2, "同一個種子抽出完全相同的題目與標註")

    teacher_q = {q["query"]: q["qid"] for q in q1 if q["stratum"] == "teacher"}
    check(set(teacher_q) == {"王小明老師開的課", "李大華老師開的課"},
          "「待聘」不當人名;相關課程超過上限的老師不抽")
    li = {j["course_id"] for j in j1 if j["qid"] == teacher_q.get("李大華老師開的課")}
    check(li == {"1151000000002", "1151000000003"}, "多位老師合開的課有正確拆開")
    check(all(j["grade"] == 2 and j["source"] == "rule" for j in j1), "規則層一律 grade 2、來源標 rule")
    check(sum(q["stratum"] == "title" for q in q1) == 2, "課名題依指定題數抽出")
    check(all(q["qid"][0] in "TN" and len(q["qid"]) == 3 for q in q1), "題號格式為 T01、N01")


if __name__ == "__main__":
    for fn in (test_stable_order, test_recall_mrr, test_ndcg, test_judged,
               test_bootstrap, test_data_guards, test_rule_sampling):
        fn()
    print(f"\n結果:{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)
