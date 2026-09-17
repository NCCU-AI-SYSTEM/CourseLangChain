"""把校務系統匯出的成績單 JSON 解析成「排課用的去識別化 profile」。

**純函數、零 LLM。** 給 `user_profile_tool` 用。

隱私原則(很重要):
- 這支只輸出排課需要的欄位;**姓名 / 學號 / 英文名 / 各項排名 / 個別成績分數 / 操行**
  一律不取、不回傳。分數只用來判斷「及格 / 停修 / 在修」三態,不保留數值。
- parser 不做任何 I/O、不寫檔;檔案讀取與錯誤處理在 tool wrapper。

學校 schema(節錄)以 `[{"課業學習": {...}}]` 包一層,核心欄位:
- `aboutMe.departmentProgramGrade` / `registerMajor`  —— 系級
- `totalCredits`、`coursePlan.graduationCredit`        —— 已修 / 應修學分
- `gradeRecordList[].GradeRecords[]`                   —— 逐課紀錄(課號 / 課名 / 分數 / 學期 / 類別)
- `coursePlan` 的各 remark 字串                        —— 畢業缺口(通識/體育/群/必修…)

課號比對(2026-08):成績單的 `courseCode` 就是 `data.db` COURSE 表的 `subNum`,
9 碼、**跨學年穩定**(「資料結構」101~110 學年都是 `703008001`),
且 `COURSE.id = 學年 + 學期 + subNum`(13 碼)。所以已修課一律以課號為主鍵,
不再靠課名字串比對——課名會因改名 / 全形半形 / 「(一)(二)」而誤判。
舊資料若缺 `courseCode` 才退回課名。
"""
from __future__ import annotations

import re
from typing import Any

# 在修判定用:成績單裡「尚無成績」的字樣
_IN_PROGRESS_SCORE = "成績未到或無成績"
_WITHDRAWN_SCORE = "停修"
# NCCU 及格門檻
_PASS_THRESHOLD = 60.0


def _passed(score: Any) -> bool:
    """分數可轉 float 且 >= 60 才算修過(『停修』『成績未到』皆 False)。"""
    try:
        return float(score) >= _PASS_THRESHOLD
    except (TypeError, ValueError):
        return False


def _dedup_key(code: str, name: str) -> str:
    """去重主鍵:優先用課號(跨學年穩定),缺課號的舊紀錄才退回課名。

    加前綴是為了避免課號與課名撞在同一個命名空間。
    """
    return f"code:{code}" if code else f"name:{name}"


_ACADEMIC_KEY = "課業學習"


def _unwrap(data: Any) -> dict:
    """取出「課業學習」區塊,容忍多種包法。

    校務系統實際匯出的是 **多區塊 list**:`[{"課業學習": {…}}, {"個人資料": {…}}]`。
    「個人資料」裡是姓名/手機/住址/生日/緊急聯絡人,甚至室友的姓名學號——
    整塊都不該進到排課流程,所以這裡按 key 找目標區塊,而不是取 `data[0]`
    (靠位置會在區塊順序改變時靜默解析出空 profile)。
    """
    if isinstance(data, list):
        for item in data:  # 掃描找,不依賴順序
            if isinstance(item, dict) and _ACADEMIC_KEY in item:
                return item[_ACADEMIC_KEY] if isinstance(item[_ACADEMIC_KEY], dict) else {}
        data = data[0] if data else {}  # 沒有包裝 key 的舊格式:退回第一個元素
    if isinstance(data, dict) and _ACADEMIC_KEY in data:
        data = data[_ACADEMIC_KEY]
    return data if isinstance(data, dict) else {}


def _extract_gaps(course_plan: dict) -> list[str]:
    """從 coursePlan 各 remark 欄抓出 `[...]` 缺口字串,去括號、去重、保序。"""
    keys = (
        "requiredRemark",
        "liberalTotal",
        "liberalChinese",
        "liberalEnglish",
        "liberalGeneral",
        "commonPhysical",
        "groupRemark",
        "programRemark",
    )
    seen: set[str] = set()
    gaps: list[str] = []
    for k in keys:
        val = course_plan.get(k) or ""
        for seg in re.findall(r"\[([^\[\]]+)\]", val):  # 抓每個 [...] 區段
            seg = seg.strip()
            if seg and seg not in seen:
                seen.add(seg)
                gaps.append(seg)
    return gaps


def parse_transcript(data: Any, current_term: str = "") -> dict:
    """成績單 JSON → 去識別化 profile dict。

    Args:
        data: json.load 後的物件(list / dict 皆可)
        current_term: 當前學期碼(年+期,如 "1142"),用來判斷「本學期已選」

    Returns:
        {
          "department": str, "grade_label": str,
          "earned_credits": float|None, "graduation_credits": float|None,
          "completed_courses": [{"code": 課號, "name": 課名}…],    # 及格、依課號去重
          "in_progress_courses": [{"code": …, "name": …}…],        # current_term 且尚無成績
          "graduation_gaps": [缺口字串…],
        }
        —— 不含任何姓名 / 學號 / 排名 / 分數。

        `code` 是 9 碼課號(= COURSE.subNum),可直接比對 `query_courses_tool`
        回傳的 13 碼 `course_id` 後 9 碼;來源缺 `courseCode` 時為空字串。
    """
    d = _unwrap(data)
    about = d.get("aboutMe") or {}
    course_plan = d.get("coursePlan") or {}

    completed: list[dict[str, str]] = []
    in_progress: list[dict[str, str]] = []
    seen_completed: set[str] = set()
    seen_progress: set[str] = set()

    for year_block in d.get("gradeRecordList") or []:
        for rec in year_block.get("GradeRecords") or []:
            name = (rec.get("courseName") or "").strip()
            code = str(rec.get("courseCode") or "").strip()
            if not (code or name):  # 兩者皆空的紀錄無從比對,跳過
                continue
            score = rec.get("score")
            term = str(rec.get("academicYearSemester") or "")
            key = _dedup_key(code, name)
            if _passed(score):
                if key not in seen_completed:
                    seen_completed.add(key)
                    completed.append({"code": code, "name": name})
            elif (
                current_term
                and term == current_term
                and score == _IN_PROGRESS_SCORE
                and key not in seen_progress
            ):
                seen_progress.add(key)
                in_progress.append({"code": code, "name": name})

    def _to_float(v: Any) -> float | None:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return {
        "department": (about.get("registerMajor") or "").strip(),
        "grade_label": (about.get("departmentProgramGrade") or "").strip(),
        "earned_credits": _to_float(d.get("totalCredits")),
        "graduation_credits": _to_float(course_plan.get("graduationCredit")),
        "completed_courses": completed,
        "in_progress_courses": in_progress,
        "graduation_gaps": _extract_gaps(course_plan),
    }
