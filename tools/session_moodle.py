"""每個 session 一組 NCCU Moodle 帳密,由使用者在網頁上自己登入。

nccu-moodle-mcp 是多人共用的無狀態 server:帳密不是工具參數,而是每次呼叫時放在
HTTP header(`X-Moodle-Username` / `X-Moodle-Password`)。所以後端要替「這段對話的使用者」
記住帳密,在呼叫 Moodle 工具時帶上(見 tools/mcp_tools.py)—— 模型看不到、也碰不到密碼。

隱私與安全(這支的存在理由):
- **只存在記憶體**:不寫檔、不進 DB、不進 log,程序重啟即清空,與 in-memory checkpointer 一致。
- 密碼不會出現在任何 API 回應裡;`summary` 連學號都只露後 3 碼。
- **登入前先實際驗證一次**(app.py),錯的帳密不會被存下來。NCCU 帳號連續失敗 5 次會鎖
  15 分鐘 —— 存下錯的密碼,等於讓 agent 每問一題就替使用者失敗一次。
- 同一個 session 驗證失敗 `_MAX_FAILURES` 次就暫停登入,留一點餘裕給使用者在別處打錯的次數。
- 執行期遇到登入失敗(例如使用者中途改了密碼),mcp_tools 會立刻清掉這組帳密、不再重試。
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

USER_HEADER = "X-Moodle-Username"
PASS_HEADER = "X-Moodle-Password"

# 學校的鎖定門檻是 5 次;這裡在 3 次就擋,留餘裕給使用者在其他地方打錯的次數
_MAX_FAILURES = 3
_FAILURE_WINDOW_SEC = 15 * 60

# 閒置多久就自動清掉帳密。關掉分頁時前端沒辦法通知後端(sessionStorage 跟著消失,
# 那個 session_id 再也不會出現),少了這道逾時,那組密碼會留到程序重啟。
# 每次取用都會續期,所以正在進行的對話不會被中途登出。
_IDLE_TIMEOUT_SEC = 8 * 60 * 60

# session_id -> (學號, 密碼)
_STORE: dict[str, tuple[str, str]] = {}
# session_id -> 最後一次被使用的時間
_LAST_USED: dict[str, float] = {}
# session_id -> 最近幾次登入驗證失敗的時間
_FAILURES: dict[str, list[float]] = {}
_LOCK = threading.Lock()


def _purge_expired_locked(now: float) -> None:
    """清掉閒置過久的帳密。**呼叫前必須先持有 `_LOCK`。**"""
    expired = [sid for sid, used in _LAST_USED.items() if now - used > _IDLE_TIMEOUT_SEC]
    for sid in expired:
        _STORE.pop(sid, None)
        _LAST_USED.pop(sid, None)
    if expired:
        logger.info(
            "清掉 %d 組閒置超過 %.1f 小時的 Moodle 帳密", len(expired), _IDLE_TIMEOUT_SEC / 3600
        )


def set_credentials(session_id: str, username: str, password: str) -> None:
    """存下**已驗證過**的帳密,並清掉這個 session 的失敗紀錄。"""
    with _LOCK:
        _STORE[session_id] = (username, password)
        _LAST_USED[session_id] = time.time()
        _FAILURES.pop(session_id, None)


def headers_for(session_id: str) -> dict[str, str] | None:
    """這個 session 呼叫 Moodle 工具時要帶的 header;還沒登入或閒置過久回 None。

    每次取用都會續期 —— 正在進行的對話不會在中途被登出。
    """
    now = time.time()
    with _LOCK:
        _purge_expired_locked(now)
        creds = _STORE.get(session_id)
        if creds:
            _LAST_USED[session_id] = now
    if not creds:
        return None
    return {USER_HEADER: creds[0], PASS_HEADER: creds[1]}


def clear_credentials(session_id: str) -> bool:
    """登出:移除這個 session 的帳密。回傳原本是否存在。"""
    with _LOCK:
        _LAST_USED.pop(session_id, None)
        return _STORE.pop(session_id, None) is not None


def record_failure(session_id: str) -> None:
    with _LOCK:
        _FAILURES.setdefault(session_id, []).append(time.time())


def too_many_failures(session_id: str) -> bool:
    """最近 15 分鐘內這個 session 是否已經驗證失敗太多次。"""
    now = time.time()
    with _LOCK:
        recent = [t for t in _FAILURES.get(session_id, []) if now - t < _FAILURE_WINDOW_SEC]
        _FAILURES[session_id] = recent
    return len(recent) >= _MAX_FAILURES


def summary(session_id: str) -> dict:
    """給前端顯示的狀態。**不含密碼**,學號只露後 3 碼。

    只是查看狀態、不算「使用」,所以刻意不續期。
    """
    now = time.time()
    with _LOCK:
        _purge_expired_locked(now)
        creds = _STORE.get(session_id)
    if not creds:
        return {"connected": False}
    user = creds[0]
    return {"connected": True, "username": "*" * max(0, len(user) - 3) + user[-3:]}
