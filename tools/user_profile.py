"""user_profile_tool —— 回使用者的去識別化修課狀況(可選功能)。

資料來源有兩個,依序嘗試:
1. **本次 session 上傳的成績單**(`tools/session_profile`)——正式使用走這條。
   原始檔在解析後即丟棄,只有去識別化結果留在記憶體。
2. 本機檔案 `paths.USER_RECORD_JSON`——CLI / 開發測試用的後備。

設計重點(隱私優先):
- **可選**:兩邊都沒有時回友善提示(非 ERROR),agent 照常排課,不主動要求使用者提供。
- **不留存**:不寫檔、不進 DB;session 結束或程序重啟即消失。
- **去識別化**:輸出由 `transcript_parser` 產生,已剔除姓名 / 學號 / 排名 / 分數 / 聯絡資訊。

契約(見 tools/registry.py):扁平參數、回字串不 raise、無 LLM。
"""
from __future__ import annotations

import json

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from paths import COURSE_SEMESTER, COURSE_YEAR, USER_RECORD_JSON

from . import session_profile
from .registry import register_tool
from .transcript_parser import parse_transcript

CURRENT_TERM = f"{COURSE_YEAR}{COURSE_SEMESTER}"  # 例如 "1142"

# 已修 + 已選課名清單若太長,截斷顯示的上限(避免塞爆 context)
_MAX_LIST = 60


def _fmt_course(c: dict) -> str:
    """一門課顯示成「課名(課號)」;任一欄缺失就只顯示有的那個。"""
    name, code = (c.get("name") or "").strip(), (c.get("code") or "").strip()
    if name and code:
        return f"{name}({code})"
    return name or code


def _join(items: list[str], limit: int = _MAX_LIST) -> str:
    if not items:
        return "(無)"
    shown = items[:limit]
    tail = f" …等 {len(items)} 門" if len(items) > limit else ""
    return "、".join(shown) + tail


def _join_courses(courses: list[dict], limit: int = _MAX_LIST) -> str:
    return _join([_fmt_course(c) for c in courses], limit)


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
    exclude_codes = [c["code"] for c in exclude if c.get("code")]
    exclude_names = [c["name"] for c in exclude if c.get("name")]

    return "\n".join(
        [
            "## 修課狀況(來自你提供的成績單,僅本次使用、未儲存,不含姓名學號)",
            f"- 年級:{grade}",
            f"- 學分:{credit_line}",
            f"- 已修過(勿重複推薦):{_join_courses(completed)}",
            f"- 本學期({CURRENT_TERM})已選:{_join_courses(in_prog)}",
            f"- 畢業尚缺:{('、'.join(gaps)) if gaps else '(成績單未提供或已滿足)'}",
            "",
            "## 排課建議",
            "- 排課 / 查課時請排除下列「已修過 + 本學期已選」的課,不要放進 schedule_tool",
            f"- 排除課號(主要依據):{_join(exclude_codes)}",
            "- 比對方式:query_courses_tool 回傳的 course_id 是 13 碼(學年+學期+課號),"
            "**後 9 碼**等於上面任一課號就是同一門課,必須排除",
            f"- 排除課名(輔助,防止課號缺漏):{_join(exclude_names)}",
            "- 若畢業尚缺某類別,優先往該類別補課",
        ]
    )


_NO_PROFILE_MSG = (
    "使用者尚未上傳成績單,略過個人化(此為可選功能)。"
    "若要強化選課推薦,可在畫面上傳從校務系統匯出的成績單 JSON;"
    "系統只會讀取修課狀況、不會儲存,也不會取用姓名/學號等個人資料。"
)


@tool
def user_profile_tool(
    record_path: str = USER_RECORD_JSON,
    config: RunnableConfig = None,
) -> str:
    """讀取使用者的修課狀況(已修課、本學期已選、畢業缺口),用於個人化排課。

    用途:當使用者要「個人化排課」、提到自身修課狀況、或要避免重複推薦已修過的課時呼叫。
    這是可選功能——使用者需自行上傳成績單;沒有上傳時本工具會說明,agent 應照常排課,
    不要主動要求使用者提供。

    參數:
    - record_path: 後備的本機成績單路徑(CLI 用;正式流程走 session 上傳,不需要指定)

    回傳:去識別化的修課狀況與排課建議(不含姓名/學號/排名/分數);未提供時回提示文字。
    """
    # 1) 優先用本次 session 上傳的(session_id 由框架注入,LLM 看不到)
    session_id = ((config or {}).get("configurable") or {}).get("thread_id") or ""
    if session_id:
        profile = session_profile.get_profile(session_id)
        if profile:
            return _format_profile(profile)

    # 2) 後備:本機檔案(CLI / 開發測試)
    try:
        with open(record_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return _NO_PROFILE_MSG
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
