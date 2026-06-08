"""user_profile_tool + transcript_parser 單元測試。

跑:`.venv/bin/python -m tests.test_user_profile`(在 CourseLangChain/ 下)
standalone 風格(自帶 check + 報表)。**測試資料全為去識別化假資料**,
並特意塞入假姓名/學號來驗證 parser 不會把 PII 帶出去。
"""
from __future__ import annotations

import json
import os
import tempfile

from tools.transcript_parser import parse_transcript

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


# 去識別化假成績單;PII 欄位放假值,用來確認 parser 會丟掉它們
FAKE = [
    {
        "課業學習": {
            "aboutMe": {
                "departmentProgramGrade": "資訊三",
                "registerMajor": "資訊科學系",
                "chineseName": "王小明",          # PII(應被剔除)
                "studentNumber": "000000000",     # PII(應被剔除)
                "englishName": "WANG, FAKE",      # PII(應被剔除)
            },
            "rankingDepartment": "31 / 55",       # 排名(應被剔除)
            "totalCredits": "95",
            "coursePlan": {
                "graduationCredit": "128.0",
                "requiredRemark": "[缺程式能力檢定1]",
                "commonPhysical": "[體育欠2門]",
                "liberalTotal": "[通識尚缺7學分]",
                "groupRemark": "[群A缺1門][群B缺1門]",
                "programRemark": "",
            },
            "gradeRecordList": [
                {
                    "AcademicYear": "114",
                    "GradeRecords": [
                        {"courseName": "資料庫系統", "score": "成績未到或無成績",
                         "academicYearSemester": "1142", "credit": "3.0"},
                        {"courseName": "機器學習概論", "score": "88.00",
                         "academicYearSemester": "1141", "credit": "3.0"},
                        {"courseName": "範例停修課", "score": "停修",
                         "academicYearSemester": "1141", "credit": "3.0"},
                        {"courseName": "範例當掉課", "score": "45.00",
                         "academicYearSemester": "1141", "credit": "3.0"},
                    ],
                },
                {
                    "AcademicYear": "113",
                    "GradeRecords": [
                        {"courseName": "演算法", "score": "73.00",
                         "academicYearSemester": "1132", "credit": "3.0"},
                        {"courseName": "資料結構", "score": "71.00",
                         "academicYearSemester": "1131", "credit": "3.0"},
                    ],
                },
            ],
        }
    }
]


def test_parser_core() -> None:
    print("[transcript_parser 解析]")
    p = parse_transcript(FAKE, current_term="1142")

    check(p["department"] == "資訊科學系", "系所解析")
    check(p["grade_label"] == "資訊三", "年級解析")
    check(p["earned_credits"] == 95.0 and p["graduation_credits"] == 128.0, "學分解析")

    comp = p["completed_courses"]
    check("機器學習概論" in comp and "演算法" in comp and "資料結構" in comp, "及格課列入已修")
    check("範例停修課" not in comp, "停修課不列入已修")
    check("範例當掉課" not in comp, "不及格(45)不列入已修")
    check("資料庫系統" not in comp, "本學期在修課不列入已修")

    check(p["in_progress_courses"] == ["資料庫系統"], "本學期(1142)在修課正確")

    gaps = p["graduation_gaps"]
    check(
        {"缺程式能力檢定1", "體育欠2門", "通識尚缺7學分", "群A缺1門", "群B缺1門"} <= set(gaps),
        "畢業缺口字串完整抽出(含 groupRemark 多段)",
    )


def test_parser_strips_pii() -> None:
    print("[transcript_parser 隱私:PII 不外洩]")
    p = parse_transcript(FAKE, current_term="1142")
    blob = json.dumps(p, ensure_ascii=False)
    for pii in ("王小明", "000000000", "WANG, FAKE", "31 / 55"):
        check(pii not in blob, f"輸出不含 PII「{pii}」")


def test_parser_robust() -> None:
    print("[transcript_parser 容錯]")
    check(parse_transcript([], current_term="1142")["completed_courses"] == [], "空 list 不炸")
    check(parse_transcript({}, current_term="1142")["grade_label"] == "", "空 dict 不炸")
    # 直接給內層 dict(沒有 課業學習 包裝)也要能解析
    inner = FAKE[0]["課業學習"]
    check(parse_transcript(inner, current_term="1142")["department"] == "資訊科學系", "未包裝的 dict 也能解析")


def test_tool_missing_file() -> None:
    print("[user_profile_tool 檔案不存在 → 可選提示,非 ERROR]")
    from tools.user_profile import user_profile_tool

    out = user_profile_tool.invoke({"record_path": "/nonexistent/__no_such__.json"})
    check(not out.startswith("ERROR"), "找不到檔不回 ERROR")
    check("可選" in out or "未提供" in out, "提示為可選功能")


def test_tool_happy_and_privacy() -> None:
    print("[user_profile_tool 正常讀檔 + 隱私]")
    from tools.user_profile import user_profile_tool

    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(FAKE, f, ensure_ascii=False)
        out = user_profile_tool.invoke({"record_path": path})
    finally:
        os.remove(path)

    check("機器學習概論" in out and "演算法" in out, "輸出含已修課")
    check("資料庫系統" in out, "輸出含本學期已選")
    check("範例停修課" not in out, "停修課不出現在輸出")
    for pii in ("王小明", "000000000", "WANG, FAKE"):
        check(pii not in out, f"工具輸出不含 PII「{pii}」")


def test_tool_bad_json() -> None:
    print("[user_profile_tool 壞 JSON → ERROR]")
    from tools.user_profile import user_profile_tool

    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("{ not valid json ,,, ")
        out = user_profile_tool.invoke({"record_path": path})
    finally:
        os.remove(path)
    check(out.startswith("ERROR"), "壞 JSON 回 ERROR 字串(不 raise)")


if __name__ == "__main__":
    test_parser_core()
    test_parser_strips_pii()
    test_parser_robust()
    test_tool_missing_file()
    test_tool_happy_and_privacy()
    test_tool_bad_json()
    print(f"\n結果:{_PASS} passed, {_FAIL} failed")
    raise SystemExit(1 if _FAIL else 0)
