"""user_profile_tool —— 讀使用者自帶的成績單,回去識別化的修課狀況(可選功能)。

設計重點(隱私優先):
- **可選**:讀不到成績單檔時回友善提示(非 ERROR),agent 照常排課,不要求使用者提供。
- **不留存**:只在當次呼叫讀檔解析,不寫任何地方。
- **去識別化**:輸出由 `transcript_parser` 產生,已剔除姓名 / 學號 / 排名 / 分數。

契約(見 tools/registry.py):扁平參數、回字串不 raise、無 LLM。
"""
from __future__ import annotations

import json

from langchain_core.tools import tool

from paths import COURSE_SEMESTER, COURSE_YEAR, USER_RECORD_JSON

from .registry import register_tool
from .transcript_parser import parse_transcript

CURRENT_TERM = f"{COURSE_YEAR}{COURSE_SEMESTER}"  # 例如 "1142"

# 已修 + 已選課名清單若太長,截斷顯示的上限(避免塞爆 context)
_MAX_LIST = 60


def _join(names: list[str], limit: int = _MAX_LIST) -> str:
    if not names:
        return "(無)"
    shown = names[:limit]
    tail = f" …等 {len(names)} 門" if len(names) > limit else ""
    return "、".join(shown) + tail


def _format_profile(p: dict) -> str:
    earned = p.get("earned_credits")
    need = p.get("graduation_credits")
    completed = p.get("completed_courses") or []
    in_prog = p.get("in_progress_courses") or []
    gaps = p.get("graduation_gaps") or []

    credit_line = "(未知)"
    if earned is not None and need is not None:
        credit_line = f"已修 {earned:g} / 畢業需 {need:g}(還缺約 {need - earned:g})"
    elif earned is not None:
        credit_line = f"已修 {earned:g}"

    grade = p.get("grade_label") or p.get("department") or "(未知)"
    exclude = completed + in_prog

    return "\n".join(
        [
            "## 修課狀況(來自你提供的成績單,僅本次使用、未儲存,不含姓名學號)",
            f"- 年級:{grade}",
            f"- 學分:{credit_line}",
            f"- 已修過(勿重複推薦):{_join(completed)}",
            f"- 本學期({CURRENT_TERM})已選:{_join(in_prog)}",
            f"- 畢業尚缺:{('、'.join(gaps)) if gaps else '(成績單未提供或已滿足)'}",
            "",
            "## 排課建議",
            "- 排課 / 查課時請排除上面「已修過 + 本學期已選」的課名",
            f"- 排除清單:{_join(exclude)}",
            "- 若畢業尚缺某類別,優先往該類別補課",
        ]
    )


@tool
def user_profile_tool(record_path: str = USER_RECORD_JSON) -> str:
    """讀取使用者自帶的成績單,回傳其修課狀況(已修課、本學期已選、畢業缺口)。

    用途:當使用者要「個人化排課」、提到自身修課狀況、或要避免重複推薦已修過的課時呼叫。
    這是可選功能——使用者需自行提供成績單 JSON;沒有提供時本工具會說明,agent 應照常排課。

    參數:
    - record_path: 成績單 JSON 路徑(預設讀 paths.USER_RECORD_JSON)

    回傳:去識別化的修課狀況與排課建議(不含姓名/學號/排名/分數);未提供檔案時回提示文字。
    """
    try:
        with open(record_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return (
            "使用者尚未提供成績單,略過個人化(此為可選功能)。"
            "若要強化選課推薦,可從校務系統匯出成績單 JSON 放到 "
            f"{record_path};檔案僅當次讀取、不會儲存,也不會送出姓名/學號。"
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return f"ERROR: 成績單 JSON 格式無法解析({e})。請確認是校務系統匯出的原始檔。"
    except OSError as e:
        return f"ERROR: 讀取成績單失敗({e})。"

    try:
        profile = parse_transcript(data, current_term=CURRENT_TERM)
    except Exception as e:  # noqa: BLE001 — 契約:回字串而非 raise
        return f"ERROR: 解析成績單時發生錯誤({e})。"

    if not (profile.get("completed_courses") or profile.get("grade_label")):
        return "成績單已讀取,但解析不出可用的修課資料(格式可能不符)。已略過個人化。"
    return _format_profile(profile)


register_tool(user_profile_tool)
