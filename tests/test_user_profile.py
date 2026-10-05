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
                        {"courseName": "資料庫系統", "courseCode": "703025001",
                         "score": "成績未到或無成績",
                         "academicYearSemester": "1142", "credit": "3.0"},
                        {"courseName": "機器學習概論", "courseCode": "703901001",
                         "score": "88.00",
                         "academicYearSemester": "1141", "credit": "3.0"},
                        {"courseName": "範例停修課", "courseCode": "000000001",
                         "score": "停修",
                         "academicYearSemester": "1141", "credit": "3.0"},
                        {"courseName": "範例當掉課", "courseCode": "000000002",
                         "score": "45.00",
                         "academicYearSemester": "1141", "credit": "3.0"},
                    ],
                },
                {
                    "AcademicYear": "113",
                    "GradeRecords": [
                        {"courseName": "演算法", "courseCode": "703022001",
                         "score": "73.00",
                         "academicYearSemester": "1132", "credit": "3.0"},
                        {"courseName": "資料結構", "courseCode": "703008001",
                         "score": "71.00",
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
    names = [c["name"] for c in comp]
    codes = [c["code"] for c in comp]
    check("機器學習概論" in names and "演算法" in names and "資料結構" in names, "及格課列入已修")
    check("範例停修課" not in names, "停修課不列入已修")
    check("範例當掉課" not in names, "不及格(45)不列入已修")
    check("資料庫系統" not in names, "本學期在修課不列入已修")
    check({"703901001", "703022001", "703008001"} <= set(codes), "已修課帶出課號")

    check(
        p["in_progress_courses"] == [{"code": "703025001", "name": "資料庫系統"}],
        "本學期(1142)在修課正確(含課號)",
    )

    gaps = p["graduation_gaps"]
    check(
        {"缺程式能力檢定1", "體育欠2門", "通識尚缺7學分", "群A缺1門", "群B缺1門"} <= set(gaps),
        "畢業缺口字串完整抽出(含 groupRemark 多段)",
    )


def _wrap(records: list[dict]) -> list:
    """把逐課紀錄包成學校 schema,供課號比對的測試用。"""
    return [{"課業學習": {
        "aboutMe": {"departmentProgramGrade": "資訊三", "registerMajor": "資訊科學系"},
        "gradeRecordList": [{"AcademicYear": "113", "GradeRecords": records}],
    }}]


def test_parser_dedups_by_course_code() -> None:
    print("[transcript_parser 課號比對]")

    # 同一門課重修 / 課名微調(全形括號),課號相同 → 視為同一門,只留一筆
    p = parse_transcript(_wrap([
        {"courseName": "資料結構", "courseCode": "703008001", "score": "45.00",
         "academicYearSemester": "1131"},
        {"courseName": "資料結構(一)", "courseCode": "703008001", "score": "78.00",
         "academicYearSemester": "1141"},
    ]), current_term="1142")
    check(len(p["completed_courses"]) == 1, "同課號重修只留一筆(課名不同也不重複)")
    check(p["completed_courses"][0]["code"] == "703008001", "去重後保留課號")

    # 不同系開的同名課,課號不同 → 是兩門不同的課,都要保留
    p = parse_transcript(_wrap([
        {"courseName": "統計學", "courseCode": "703031001", "score": "80.00",
         "academicYearSemester": "1131"},
        {"courseName": "統計學", "courseCode": "205016001", "score": "85.00",
         "academicYearSemester": "1132"},
    ]), current_term="1142")
    check(len(p["completed_courses"]) == 2, "同名不同課號視為兩門課(不誤併)")

    # 舊資料沒有 courseCode → 退回課名去重,行為與改動前一致
    p = parse_transcript(_wrap([
        {"courseName": "演算法", "score": "73.00", "academicYearSemester": "1131"},
        {"courseName": "演算法", "score": "90.00", "academicYearSemester": "1132"},
    ]), current_term="1142")
    check(len(p["completed_courses"]) == 1, "缺課號時退回課名去重")
    check(p["completed_courses"][0]["code"] == "", "缺課號時 code 為空字串")

    # 只有課號沒課名的紀錄仍要收(排除比對只需要課號)
    p = parse_transcript(_wrap([
        {"courseCode": "703099001", "score": "70.00", "academicYearSemester": "1131"},
    ]), current_term="1142")
    check(
        p["completed_courses"] == [{"code": "703099001", "name": ""}],
        "只有課號沒課名也收得到",
    )

    # 課號課名都空的髒紀錄 → 跳過
    p = parse_transcript(_wrap([
        {"courseName": "", "courseCode": "", "score": "70.00", "academicYearSemester": "1131"},
    ]), current_term="1142")
    check(p["completed_courses"] == [], "課號課名皆空的紀錄跳過")


def test_parser_strips_pii() -> None:
    print("[transcript_parser 隱私:PII 不外洩]")
    p = parse_transcript(FAKE, current_term="1142")
    blob = json.dumps(p, ensure_ascii=False)
    for pii in ("王小明", "000000000", "WANG, FAKE", "31 / 55"):
        check(pii not in blob, f"輸出不含 PII「{pii}」")


# 模仿校務系統「真實匯出」的多區塊結構(值全為假)。
# 重點:第二個區塊「個人資料」含手機/住址/生日/緊急聯絡人,以及**室友的姓名學號**
# (第三人個資),整塊都不該流進排課流程。
FAKE_FULL_EXPORT = [
    {
        "課業學習": {
            "totalCredits": "95",
            "coursePlan": {
                "graduationCredit": "128.0",
                "requiredRemark": "[缺程式能力檢定1]",
                "commonPhysical": "[體育欠2門]",
                "liberalTotal": "[通識尚缺7學分]",
                "groupRemark": "[群A缺1門][群D缺1門]",
            },
            "rankingClass": "99",
            "rankingDepartment": "99",
            "aboutMe": {
                "studentNumber": "111999888",
                "chineseName": "假名甲",
                "englishName": "FAKE, NAME",
                "registerMajor": "資訊科學系",
                "departmentProgramGrade": "資訊四",
            },
            "totalAverageScore": {
                "averageScore": "83.41",
                "rankingDepartment": "99 / 55",
                "departmentRankPercentage": "56.36 %",
            },
            "averageScoreList": [
                {"academicYear": "114", "averageScore": "79.94", "rankingDepartment": "99 / 54"},
            ],
            "conductRecordList": [{"semester": "2", "score": 85.0, "academicYear": "114"}],
            "gradeRecordList": [
                {
                    "AcademicYear": "114",
                    "GradeRecords": [
                        {"semester": "2", "courseName": "資料庫系統", "credit": "3.0",
                         "courseCode": "703025001", "remark": "", "academicYearSemester": "1142",
                         "requiredOrElectiveCourse": "群", "academicYear": "114",
                         "score": "93.00", "scoreIfPass": ""},
                        {"semester": "2", "courseName": "計程實習", "credit": "0.0",
                         "courseCode": "703958001", "remark": "", "academicYearSemester": "1142",
                         "requiredOrElectiveCourse": "必", "academicYear": "114",
                         "score": "成績未到或無成績", "scoreIfPass": ""},
                        {"semester": "1", "courseName": "假停修課", "credit": "3.0",
                         "courseCode": "070394011", "remark": "", "academicYearSemester": "1141",
                         "requiredOrElectiveCourse": "選", "academicYear": "114",
                         "score": "停修", "scoreIfPass": ""},
                        {"semester": "1", "courseName": "近代臺灣人物", "credit": "2.0",
                         "courseCode": "041089001", "remark": "人文類", "academicYearSemester": "1141",
                         "requiredOrElectiveCourse": "群", "academicYear": "114",
                         "score": "80.00", "scoreIfPass": ""},
                    ],
                },
            ],
        }
    },
    {
        "個人資料": {
            "personalInformation": {
                "chineseName": "假名甲",
                "mainEmail": "fake@example.edu.tw",
                "mobilePhone": "0900000000",
                "mailingAddress": "假地址一段1號",
                "dateOfBirth": "民國99年1月1日",
                "emergencyContactName": "假聯絡人",
                "emergencyContactTelephone": "0911111111",
            },
            "dormitoryLifeList": [
                {"room": "5218", "dormitory": "假宿舍",
                 # 第三人個資:室友姓名 + 學號
                 "roomate": "(1)假室友乙 會計一 111000111、(2)假室友丙 法律二 111000222"},
            ],
        }
    },
]


def test_parser_real_export_shape() -> None:
    print("[transcript_parser 真實匯出結構:多區塊 + PII 隔離]")
    p = parse_transcript(FAKE_FULL_EXPORT, current_term="1142")

    check(p["department"] == "資訊科學系", "從多區塊 list 取到正確的課業學習區塊")
    check(p["earned_credits"] == 95.0, "學分解析正確")
    names = [c["name"] for c in p["completed_courses"]]
    check("資料庫系統" in names and "近代臺灣人物" in names, "及格課解析正確")
    check("假停修課" not in names, "停修不列入")
    check(
        p["in_progress_courses"] == [{"code": "703958001", "name": "計程實習"}],
        "0 學分實習課的在修狀態也抓得到",
    )

    # 整份輸出不得含任何 PII —— 含本人的與室友的
    blob = json.dumps(p, ensure_ascii=False)
    for pii in (
        "假名甲", "111999888", "FAKE, NAME",          # 本人身分
        "0900000000", "假地址一段1號", "民國99年1月1日",  # 聯絡與生日
        "假聯絡人", "0911111111", "fake@example",      # 緊急聯絡人 / email
        "假室友乙", "111000111", "假室友丙",            # 第三人個資
        "99 / 55", "56.36 %", "83.41", "85.0",         # 排名 / 平均 / 操行
        "93.00", "80.00",                              # 個別分數
    ):
        check(pii not in blob, f"輸出不含「{pii}」")


def test_parser_block_order_independent() -> None:
    print("[transcript_parser 區塊順序無關]")
    reversed_export = list(reversed(FAKE_FULL_EXPORT))  # 個人資料排在前面
    p = parse_transcript(reversed_export, current_term="1142")
    check(p["department"] == "資訊科學系", "個人資料排在前面時仍找得到課業學習")
    check(len(p["completed_courses"]) > 0, "課程仍解析得到(不會靜默回空)")

    blob = json.dumps(p, ensure_ascii=False)
    check("假室友乙" not in blob and "0900000000" not in blob, "順序顛倒也不會誤取個人資料")


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
    check("703901001" in out and "703022001" in out, "輸出含已修課的課號")
    check("703025001" in out, "輸出含本學期已選的課號")
    check("後 9 碼" in out, "輸出說明課號與 course_id 的比對方式")
    check("000000001" not in out, "停修課的課號也不出現在輸出")
    for pii in ("王小明", "000000000", "WANG, FAKE"):
        check(pii not in out, f"工具輸出不含 PII「{pii}」")


def test_session_profile_store() -> None:
    print("[session 成績單:只留去識別化結果]")
    from tools import session_profile as sp

    sid = "test-profile-session"
    sp.clear_profile(sid)
    check(sp.get_profile(sid) is None, "初始沒有資料")

    profile = sp.set_profile(sid, FAKE_FULL_EXPORT, current_term="1142")
    check(profile["department"] == "資訊科學系", "解析成功")

    # 存下來的東西不得含原始 JSON 的任何 PII
    blob = json.dumps(sp.get_profile(sid), ensure_ascii=False)
    for pii in ("假名甲", "111999888", "0900000000", "假地址一段1號",
                "假室友乙", "111000111", "假聯絡人", "93.00", "99 / 55"):
        check(pii not in blob, f"session 儲存內容不含「{pii}」")

    # 摘要同樣不能洩漏
    summary_blob = json.dumps(sp.summary(sp.get_profile(sid)), ensure_ascii=False)
    check("假名甲" not in summary_blob and "111999888" not in summary_blob, "摘要不含 PII")
    check(sp.summary(None) == {"has_profile": False}, "沒有資料時摘要正確")

    check(sp.clear_profile(sid) is True, "清除回 True")
    check(sp.clear_profile(sid) is False, "重複清除回 False")
    check(sp.get_profile(sid) is None, "清除後讀不到")


def test_session_profile_rejects_garbage() -> None:
    print("[session 成績單:格式不符要擋下]")
    from tools import session_profile as sp

    for bad in ({}, [], {"不相干": 1}, [{"個人資料": {"personalInformation": {}}}]):
        try:
            sp.set_profile("test-garbage", bad, current_term="1142")
            check(False, f"應該擋下 {bad!r}")
        except ValueError:
            check(True, f"擋下無法解析的輸入 {str(bad)[:24]}")
    sp.clear_profile("test-garbage")


def test_tool_prefers_session_over_file() -> None:
    print("[user_profile_tool:優先用 session 上傳的資料]")
    from tools import session_profile as sp
    from tools.user_profile import user_profile_tool

    sid = "test-tool-session"
    sp.clear_profile(sid)
    cfg = {"configurable": {"thread_id": sid}}

    # 還沒上傳,且本機檔不存在 → 友善提示,不是 ERROR
    out = user_profile_tool.invoke(
        {"record_path": "/nonexistent/__none__.json"}, config=cfg
    )
    check(not out.startswith("ERROR"), "沒上傳時不回 ERROR")
    check("上傳" in out, "提示使用者可上傳成績單")

    # 上傳後即使 record_path 指向不存在的檔,也要讀得到 session 的資料
    sp.set_profile(sid, FAKE_FULL_EXPORT, current_term="1142")
    out = user_profile_tool.invoke(
        {"record_path": "/nonexistent/__none__.json"}, config=cfg
    )
    check("資料庫系統" in out, "讀到 session 上傳的修課資料")
    check("703025001" in out, "輸出含課號")
    for pii in ("假名甲", "111999888", "假室友乙"):
        check(pii not in out, f"工具輸出不含「{pii}」")

    # 另一個 session 讀不到
    other = user_profile_tool.invoke(
        {"record_path": "/nonexistent/__none__.json"},
        config={"configurable": {"thread_id": "someone-else"}},
    )
    check("資料庫系統" not in other, "其他 session 讀不到這份資料")
    sp.clear_profile(sid)


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
    test_parser_dedups_by_course_code()
    test_parser_real_export_shape()
    test_parser_block_order_independent()
    test_parser_strips_pii()
    test_parser_robust()
    test_tool_missing_file()
    test_tool_happy_and_privacy()
    test_session_profile_store()
    test_session_profile_rejects_garbage()
    test_tool_prefers_session_over_file()
    test_tool_bad_json()
    print(f"\n結果:{_PASS} passed, {_FAIL} failed")
    raise SystemExit(1 if _FAIL else 0)
