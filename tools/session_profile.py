"""每個 session 一份「去識別化修課狀況」,由使用者上傳的成績單解析而來。

隱私模型(這支的存在理由):
- **原始成績單只在解析的那一瞬間存在於記憶體**,解析完立刻丟棄——不寫檔、不進 DB、
  不寫暫存。原始檔含姓名/學號/手機/住址/生日/緊急聯絡人,甚至室友的姓名學號。
- 這裡只留 `transcript_parser.parse_transcript` 的產物:系級、已修課號課名、
  本學期已選、畢業缺口、學分數。PII 在解析階段就被剔掉了(見 tests/test_user_profile.py)。
- per-session、純記憶體,程序重啟即清空;與 in-memory checkpointer 一致。
- 使用者可隨時呼叫 `clear_profile` 移除。

比讀本機檔(`paths.USER_RECORD_JSON`)更安全:那個做法會讓含完整 PII 的原始檔
長期躺在伺服器硬碟上,這裡連落地都不會發生。
"""
from __future__ import annotations

import threading
from typing import Any

from tools.transcript_parser import parse_transcript

# session_id -> 去識別化 profile(**不含**原始 JSON)
_STORE: dict[str, dict] = {}
_LOCK = threading.Lock()


def set_profile(session_id: str, raw: Any, current_term: str = "") -> dict:
    """解析上傳的成績單並存下**去識別化結果**;`raw` 不會被保留。

    Raises:
        ValueError: 解析不出可用的修課資料(格式不符)
    """
    profile = parse_transcript(raw, current_term=current_term)
    if not (profile.get("completed_courses") or profile.get("grade_label")):
        raise ValueError("解析不出可用的修課資料,請確認上傳的是校務系統匯出的成績單 JSON。")

    with _LOCK:
        _STORE[session_id] = profile
    # raw 在此函數結束後即無參照,不留任何副本
    return profile


def get_profile(session_id: str) -> dict | None:
    with _LOCK:
        profile = _STORE.get(session_id)
    return dict(profile) if profile else None


def clear_profile(session_id: str) -> bool:
    """移除這個 session 的修課資料。回傳原本是否存在。"""
    with _LOCK:
        return _STORE.pop(session_id, None) is not None


def summary(profile: dict | None) -> dict:
    """給前端顯示的摘要(同樣不含 PII)。"""
    if not profile:
        return {"has_profile": False}
    return {
        "has_profile": True,
        "department": profile.get("department", ""),
        "grade_label": profile.get("grade_label", ""),
        "earned_credits": profile.get("earned_credits"),
        "graduation_credits": profile.get("graduation_credits"),
        "completed_count": len(profile.get("completed_courses") or []),
        "in_progress_count": len(profile.get("in_progress_courses") or []),
        "graduation_gaps": profile.get("graduation_gaps") or [],
    }
