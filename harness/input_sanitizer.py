"""L1:輸入清理。

把判定為不安全的輸入直接擋下,不要送進 LLM。

**定位:L1 是降噪,不是安全邊界。** 研究顯示任何輸入過濾都擋得住隨手嘗試、擋不住有心的
攻擊者(Nasr et al., 2025, "The Attacker Moves Second")。真正的保證在 L3 與工具權限。
所以這裡寧可少擋,也不要誤擋正常問題 —— 尤其是多輪對話裡的「忽略之前的條件,改排…」。

比對流程:
1. NFKC 正規化(全形 → 半形)、移除格式字元(零寬空白等)
2. 轉小寫、**去掉所有空白**後比對 —— 「I g n o r e」「ignore  previous」都變成同一個字串,
   所以英文規則一律寫成不含空白的形式
3. 規則涵蓋繁簡中文

公開介面只有 `sanitize_input(raw)`,回傳 `(text, is_safe)`。FastAPI / CLI 入口呼叫這個函數,
`is_safe=False` 時直接給用戶友善訊息,不要再呼叫 agent。新增規則時請同步在
tests/test_input_sanitizer.py 加攻擊句**和**不該被擋的正常句。
"""
from __future__ import annotations

import re
import unicodedata

MAX_INPUT_LENGTH = 500

# ── 在「原文」上比對:特殊 token、角色標記(需要保留換行與符號)────────────
RAW_PATTERNS: tuple[re.Pattern, ...] = (
    # 模型的特殊 token,例如 <|im_start|>、<|endoftext|>。只認短識別字:原本的 <\|.*?\|>
    # 會誤中課綱裡「|<<書名>>|」「|<https://…>|」這種表格寫法(全校 3650 門課中 13 門)
    re.compile(r"<\|[A-Za-z0-9_]{1,40}\|>"),
    re.compile(r"^\s*(system|assistant)\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"</?\s*(system|user|assistant)\s*>", re.IGNORECASE),
)

# ── 在「去空白、小寫」後的字串上比對 ──────────────────────────────────────
_OVERRIDE_VERB = r"(忽略|無視|无视|忘記|忘记|忘掉|不要理會|不要理会|跳過|跳过|拋開|抛开|ignore|disregard|forget)"
_EARLIER = r"(你)?(之前|先前|以上|上面|上述|前面|所有|全部|一切|allof|all|any|every|the|your)*(的)?(所有|全部)?"
# 刻意不含「條件」「限制」「要求」「設定」:多輪對話裡「忽略之前的條件,改排…」是正常用法
_RULE_NOUN = r"(指令|指示|規則|规则|提示|命令|prompt|instruction|rule)"

COMPACT_PATTERNS: tuple[re.Pattern, ...] = (
    # 覆寫指令:「忽略以上所有指令」「无视之前的指令」「ignore all previous instructions」
    re.compile(_OVERRIDE_VERB + _EARLIER + r"(previous|prior|above|earlier|preceding)?" + _RULE_NOUN),
    re.compile(r"(ignore|disregard|forget)(everything|all)(above|before|prior|previously)"),
    re.compile(r"forget(everything|all)(you|that|i)"),
    # 越獄:開發者模式、沒有限制的助理
    re.compile(r"(開發者|开发者|越獄|越狱|無限制|无限制|不受限|上帝|developer|god|dan)(模式|mode)"),
    re.compile(r"jailbreak"),
    re.compile(r"(沒有|没有|不受|無|无)(任何)?(限制|約束|约束|規範|规范)的(助理|助手|ai|模型|機器人|机器人)"),
    re.compile(r"(扮演|假裝|假装|你現在是|你现在是|從現在開始你是|从现在开始你是).{0,6}(沒有|没有|不受|無|无)(任何)?(限制|約束|约束)"),
)

# 索取 system prompt:要同時出現「目標」與「索取動作」才算,避免誤擋「列出提示詞工程的課」
_PROMPT_TARGET = re.compile(
    r"(系統提示|系统提示|系統指令|系统指令|系統訊息|系统消息|systemprompt|initialprompt"
    r"|你的(提示詞|提示词|指令|設定|设定|規則|规则|prompt)|你(收到|被給|被给)的(提示|指令|訊息|消息|prompt))"
)
_REVEAL = re.compile(
    r"(貼出|贴出|輸出|输出|顯示|显示|列出|印出|告訴我|告诉我|重複|重复|洩漏|泄漏|透露|給我看|给我看"
    r"|原文|逐字|一字不漏|全文|reveal|print|show|repeat|output|display|leak)"
)


def _normalize(text: str) -> str:
    """NFKC(全形轉半形)+ 移除格式字元(零寬空白、方向控制等,Unicode 類別 Cf)。"""
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def is_injection(text: str) -> bool:
    """判斷(已正規化的)輸入是否像提示注入。"""
    if any(p.search(text) for p in RAW_PATTERNS):
        return True
    compact = _compact(text)
    if any(p.search(compact) for p in COMPACT_PATTERNS):
        return True
    return bool(_PROMPT_TARGET.search(compact) and _REVEAL.search(compact))


def sanitize_input(raw: str) -> tuple[str, bool]:
    """檢查 + 截斷使用者輸入。

    Returns:
        (cleaned_text, is_safe)
        - is_safe=True:可送進 agent(cleaned_text 已正規化、移除零寬字元、截斷到上限)
        - is_safe=False:呼叫端應拒絕,並回友善訊息
    """
    if not isinstance(raw, str):
        return "", False

    text = _normalize(raw).strip()
    if not text:
        return "", False

    if len(text) > MAX_INPUT_LENGTH:
        text = text[:MAX_INPUT_LENGTH]

    if is_injection(text):
        return "", False

    return text, True
