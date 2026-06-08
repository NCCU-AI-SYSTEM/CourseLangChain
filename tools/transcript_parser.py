"""把校務系統匯出的成績單 JSON 解析成「排課用的去識別化 profile」。

**純函數、零 LLM。** 給 `user_profile_tool` 用。

隱私原則(很重要):
- 這支只輸出排課需要的欄位;**姓名 / 學號 / 英文名 / 各項排名 / 個別成績分數 / 操行**
  一律不取、不回傳。分數只用來判斷「及格 / 停修 / 在修」三態,不保留數值。
- parser 不做任何 I/O、不寫檔;檔案讀取與錯誤處理在 tool wrapper。

學校 schema(節錄)以 `[{"課業學習": {...}}]` 包一層,核心欄位:
- `aboutMe.departmentProgramGrade` / `registerMajor`  —— 系級
- `totalCredits`、`coursePlan.graduationCredit`        —— 已修 / 應修學分
- `gradeRecordList[].GradeRecords[]`                   —— 逐課紀錄(課名 / 分數 / 學期 / 類別)
- `coursePlan` 的各 remark 字串                        —— 畢業缺口(通識/體育/群/必修…)
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


def _unwrap(data: Any) -> dict:
    """容忍 `[{"課業學習": {...}}]` / `{"課業學習": {...}}` / 直接 {...} 三種包法。"""
    if isinstance(data, list):
        data = data[0] if data else {}
    if isinstance(data, dict) and "課業學習" in data:
        data = data["課業學習"]
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
          "completed_courses": [課名…],      # 及格、去重
          "in_progress_courses": [課名…],    # current_term 且尚無成績
          "graduation_gaps": [缺口字串…],
        }
        —— 不含任何姓名 / 學號 / 排名 / 分數。
    """
    d = _unwrap(data)
    about = d.get("aboutMe") or {}
    course_plan = d.get("coursePlan") or {}

    completed: list[str] = []
    in_progress: list[str] = []
    seen_completed: set[str] = set()
    seen_progress: set[str] = set()

    for year_block in d.get("gradeRecordList") or []:
        for rec in year_block.get("GradeRecords") or []:
            name = (rec.get("courseName") or "").strip()
            if not name:
                continue
            score = rec.get("score")
            term = str(rec.get("academicYearSemester") or "")
            if _passed(score):
                if name not in seen_completed:
                    seen_completed.add(name)
                    completed.append(name)
            elif (
                current_term
                and term == current_term
                and score == _IN_PROGRESS_SCORE
                and name not in seen_progress
            ):
                seen_progress.add(name)
                in_progress.append(name)

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
