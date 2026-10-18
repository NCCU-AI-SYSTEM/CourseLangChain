"""Agent 層的 L3 規則:不屬於任何單一工具、但每一則最終回答都要過的檢查。

import 這個模組就會登記 V2、V4、V5;V3 需要完整的 system prompt,由 brain_agent 組好
prompt 之後呼叫 `protect_system_prompt()` 登記。排課的衝堂檢查(V1)在 tools/schedule_tool.py。

| 編號 | 規則           | 方式 | 防的是什麼                                   |
|------|----------------|------|----------------------------------------------|
| V2   | 課號落地       | 擋下 | 模型編出不存在的課號(OWASP LLM09)          |
| V3   | 系統提示洩漏   | 擋下 | 回答照貼 system prompt(OWASP LLM07)        |
| V4   | 外部連結       | 改寫 | 用連結把資料帶到外部網域(OWASP LLM05)      |
| V5   | 工具呼叫外洩   | 擋下 | 小模型把工具呼叫寫成文字當答案               |

全部是確定性程式碼,不呼叫 LLM。
"""
from __future__ import annotations

import json
import logging
import re
from urllib.parse import urlsplit

from harness import current_context, register_rewriter, register_validator

logger = logging.getLogger(__name__)


# ── V2 課號落地 ────────────────────────────────────────────────────────────

# 13 碼課號(學年 3 + 學期 1 + 序號 9);前後不能再接數字,避免從更長的數字中間截一段
_COURSE_ID = re.compile(r"(?<!\d)\d{13}(?!\d)")

COURSE_ID_MESSAGE = (
    "抱歉,這次的回答提到了本學期查不到的課號,為了避免給你錯誤的資訊,已經攔下。"
    "請重新查詢一次,或直接告訴我課名。"
)


def check_course_ids(output: str) -> tuple[bool, str]:
    """回答裡的每個 13 碼課號都必須存在於當期資料庫。

    使用者自己輸入的課號不算:回答「課號 X 查無此課」是正確行為,不能因為覆述就擋下。
    資料庫連不上時放行並記錄錯誤 —— 那時檢索也已經失效,回答本來就拿不到真實課號;
    為了這條規則讓所有回答都失敗,只會把一個故障變成兩個。
    """
    ids = set(_COURSE_ID.findall(output))
    ids -= set(_COURSE_ID.findall(current_context().get("user_input", "")))
    if not ids:
        return True, ""

    from tools.session_schedule import existing_course_ids  # 延後載入:避免循環 import

    try:
        found = existing_course_ids(sorted(ids))
    except Exception as e:  # noqa: BLE001
        logger.error("V2 課號驗證查不到資料庫,本次放行:%s", e)
        return True, ""
    missing = sorted(ids - found)
    if missing:
        return False, f"回答含當期不存在的課號:{missing}"
    return True, ""


# ── V3 系統提示洩漏 ────────────────────────────────────────────────────────

# 比對前去掉空白與 Markdown 符號。prompt 裡有規定輸出格式的表格範本,模型照格式
# 輸出是正常行為;不去掉的話,光是表頭加一排「---」就能重疊五十幾字。
_NOISE = re.compile(r"[\s|\-#*:`>_~=]+")
# 連續多少字與 prompt 相同就算洩漏。實測(正規化後):正常回答與 prompt 最長重疊 13 字,
# 照貼 40 字原文是 34 字。取 30,比正常值高一倍以上,又能抓到約 36 字以上的照貼。
LEAK_WINDOW = 30

PROMPT_LEAK_MESSAGE = "抱歉,這個問題我無法回答。如果有課程查詢或排課的需求,請直接告訴我。"


def _normalize(text: str) -> str:
    return _NOISE.sub("", text)


def make_prompt_leak_check(prompt: str):
    """依 `prompt` 建一個洩漏檢查函數。prompt 的所有 30 字片段先存成集合,
    檢查時只要掃一遍回答,不必每次都算最長共同子字串。"""
    norm = _normalize(prompt)
    windows = {norm[i : i + LEAK_WINDOW] for i in range(len(norm) - LEAK_WINDOW + 1)}

    def check(output: str) -> tuple[bool, str]:
        text = _normalize(output)
        for i in range(len(text) - LEAK_WINDOW + 1):
            if text[i : i + LEAK_WINDOW] in windows:
                return False, f"回答與 system prompt 有連續 {LEAK_WINDOW} 字以上相同"
        return True, ""

    return check


def protect_system_prompt(prompt: str) -> None:
    """登記 V3。由 brain_agent 在組好完整 prompt(含 MCP 段落)後呼叫。"""
    register_validator(
        "system_prompt_leak", make_prompt_leak_check(prompt), user_message=PROMPT_LEAK_MESSAGE
    )


# ── V4 外部連結 ────────────────────────────────────────────────────────────

# 只保留指向這些網域(含子網域)的連結。課程與校務資訊都在學校網域底下。
ALLOWED_LINK_DOMAINS = ("nccu.edu.tw",)

# [文字](網址) 與 ![替代文字](網址)。網址的兩種寫法都要認得(CommonMark 規格):
#   <...>   角括號包住,裡面可以有空白 —— 漏掉的話 [x](<https://外部 網址>) 會繞過
#   裸網址  可以含一層成對括號,例如 javascript:alert(1)
# 後面可以接 "標題"、'標題' 或 (標題)。
_INLINE_LINK = re.compile(
    r"(!?)\[([^\]\n]*)\]\(\s*"
    r"(?:<([^>\n]*)>|((?:[^()\s<>]|\([^()\s]*\))*))"
    r"(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^)\n]*\)))?\s*\)"
)
# 參考式連結的定義行:[標籤]: 網址
_REF_DEF = re.compile(r"^[ \t]{0,3}\[[^\]\n]+\]:[ \t]*<?([^\s>]+)>?.*$", re.MULTILINE)
# <https://...> 自動連結
_AUTOLINK = re.compile(r"<([a-zA-Z][a-zA-Z0-9+.-]*:[^>\s]*)>")


def _allowed_url(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    host = (parts.hostname or "").lower()
    return parts.scheme in ("http", "https") and any(
        host == d or host.endswith("." + d) for d in ALLOWED_LINK_DOMAINS
    )


# 用來確認改寫結果的 Markdown 解析器。markdown-it-py 是前端 markdown-it 的移植版,
# "js-default" 與前端的 `new MarkdownIt({})` 是同一組設定(不開 HTML、不自動連結網址),
# 所以「這裡解析不出外部連結」等於「使用者畫面上不會出現外部連結」。
_MD = None


def _parser():
    global _MD
    if _MD is None:
        from markdown_it import MarkdownIt

        _MD = MarkdownIt("js-default")
    return _MD


def find_unsafe_links(text: str) -> list[str]:
    """以真正的 Markdown 解析結果為準,列出會被渲染成外部連結或圖片的網址。"""
    found: list[str] = []

    def walk(tokens) -> None:
        for t in tokens:
            if t.type == "image":
                found.append(str(t.attrGet("src") or ""))
            elif t.type == "link_open":
                href = str(t.attrGet("href") or "")
                if not _allowed_url(href):
                    found.append(href)
            if t.children:
                walk(t.children)

    walk(_parser().parse(text))
    return found


# 退路:把所有能組成連結的符號跳脫掉。連結語法千變萬化(跳脫的括號、巢狀括號、跨行、
# 參考式定義換行…),regex 不可能全部認得;認不得的時候,寧可所有連結都失效。
_LINK_SYNTAX = str.maketrans({"[": "\\[", "]": "\\]", "<": "\\<"})


def strip_external_links(output: str) -> str:
    """移除不在白名單網域的連結,保留連結文字。圖片一律移除,只留替代文字。

    前端已經不渲染圖片,nginx 也有 CSP;這裡是後端這一側的同一道防線,
    讓 CLI、非串流 API 或日後換掉的前端都一樣安全。

    兩步:先用 regex 改寫常見寫法(保留連結文字,讀起來自然),再用解析器確認結果。
    還有漏網的就把連結語法全部跳脫;連這樣都還有,就 raise —— L3 會把改寫器出錯
    視為失敗,整段回答不送出。
    """
    result = _rewrite_known_syntax(output)
    leftover = find_unsafe_links(result)
    if not leftover:
        return result
    logger.warning("V4 regex 沒處理到的連結寫法,改為跳脫所有連結語法:%s", leftover)
    result = result.translate(_LINK_SYNTAX)
    leftover = find_unsafe_links(result)
    if leftover:
        raise ValueError(f"跳脫後仍解析出外部連結:{leftover}")
    return result


def _rewrite_known_syntax(output: str) -> str:
    removed: list[str] = []

    def inline(m: re.Match) -> str:
        bang, text, angled, bare = m.groups()
        url = angled if angled is not None else bare
        if not bang and _allowed_url(url):
            return m.group(0)
        removed.append(url)
        return text

    def ref_def(m: re.Match) -> str:
        if _allowed_url(m.group(1)):
            return m.group(0)
        removed.append(m.group(1))
        return ""

    def autolink(m: re.Match) -> str:
        if _allowed_url(m.group(1)):
            return m.group(0)
        removed.append(m.group(1))
        return ""

    result = _INLINE_LINK.sub(inline, output)
    result = _REF_DEF.sub(ref_def, result)
    result = _AUTOLINK.sub(autolink, result)
    if removed:
        logger.warning("V4 移除了回答中的外部連結或圖片:%s", removed)
    return result


# ── V5 工具呼叫外洩 ────────────────────────────────────────────────────────

# 模型把工具呼叫寫成文字時的樣子:一個只有 name/arguments 的 JSON 物件。
# 實測 qwen2:7b 有時完全不使用 tool calling 機制,直接把
# {"name": "my_schedule_tool", "arguments": {...}} 當成**最終答案**吐出來 ——
# 那則訊息沒有 tool_calls 可判斷,只能看內容本身。
#
# 刻意只認這個特定形狀(頂層 key 不超出 name/arguments/parameters),
# 而不是「看起來像 JSON 就擋」—— 否則使用者真的要求 JSON 輸出時會被誤殺。
_FENCE_HEAD = re.compile(r"^\s*(?:```(?:json)?\s*)+")
_FENCE_TAIL = re.compile(r"(?:\s*```)+\s*$")

TOOL_CALL_LEAK_MESSAGE = "抱歉,這次沒能整理出結果。請換個問法,或把需求說得更具體一點再試一次。"


def is_tool_call_text(content: str) -> bool:
    text = _FENCE_TAIL.sub("", _FENCE_HEAD.sub("", content)).strip()
    if not (text.startswith("{") and text.endswith("}")):
        return False
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    return (
        isinstance(obj, dict)
        and "name" in obj
        and set(obj) <= {"name", "arguments", "parameters"}
    )


def check_tool_call_leak(output: str) -> tuple[bool, str]:
    if is_tool_call_text(output):
        return False, "模型把工具呼叫當成最終答案輸出"
    return True, ""


# 登記順序就是執行順序(第一個失敗就停):便宜、常見的先跑,要查資料庫的 V2 放後面。
# V3 由 brain_agent 組好 prompt 後才登記,排在這兩個之後。
register_validator("tool_call_leak", check_tool_call_leak, user_message=TOOL_CALL_LEAK_MESSAGE)
register_validator("course_id_grounding", check_course_ids, user_message=COURSE_ID_MESSAGE)
register_rewriter("external_links", strip_external_links)
