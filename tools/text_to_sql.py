import json as json_mod
import logging
import time
from warnings import deprecated

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
# ChatOllama(/api/chat),不是 OllamaLLM(/api/generate)。raw completion 沒有 chat
# template,thinking 模型少了界定思考區塊的結構就停不下來 —— 實測同一個 prompt,
# OllamaLLM 370 秒未返回,ChatOllama + reasoning=False 21.7 秒、done_reason=stop、
# 思考區塊 0 字。prompt 裡寫「不要思考過程」擋不住(那是文字指示),num_predict 也
# 擋不住(只會在思考中途被切斷,回空字串)。
from langchain_ollama import ChatOllama

from paths import (
    GOOGLE_API_KEY,
    MODEL,
    OLLAMA_HOST,
    SQLITE_DEPRECATION_MSG,
    USE_GOOGLE_AI,
    USE_SQLITE,
)
from tools.constraints import constraints_to_where, parse_timefilter_from_json

logger = logging.getLogger(__name__)


def _invoke_logged(llm, prompt: str, path: str) -> str:
    """呼叫 LLM 並把原始回應記下來。

    invoke() 要等整串聚合完才返回,模型不停它就不返回 —— 一旦卡住,原本連
    「送出了什麼、收回什麼」都看不到。這裡把兩端都記進 log。
    """
    kind = type(llm).__name__
    logger.info("[%s] %s 送出 prompt(%d 字)", path, kind, len(prompt))
    start = time.time()
    try:
        result = llm.invoke(prompt)
    except Exception:
        logger.exception("[%s] %s 呼叫失敗,已等 %.1fs", path, kind, time.time() - start)
        raise
    elapsed = time.time() - start
    text = result.content if hasattr(result, "content") else str(result)
    text = text.strip()
    logger.info("[%s] %s 回應 %.1fs,%d 字:%r", path, kind, elapsed, len(text), text[:400])
    return text


def _is_json(s: str) -> bool:
    try:
        json_mod.loads(s)
        return True
    except (json_mod.JSONDecodeError, ValueError):
        return False


@tool
def text_to_sql_tool(user_input: str) -> str:
    """將時間限制轉換為 SQL WHERE 子句。

    用途：當用戶提及時間限制時（如「不要星期三」），使用此工具生成 SQL WHERE 子句。

    輸入：用戶對課程時間的限制條件
    輸出：SQL WHERE 子句（不包含 WHERE 關鍵字）

    time 欄位格式說明：
    - 一二三四五六日 = 星期一至星期日
    - A,B = 早上 (06:10-08:00)
    - 1,2,3,4 = 上午 (08:10-12:00)
    - C,D = 中午 (12:10-14:00)
    - 5,6,7,8 = 下午 (14:10-18:00)
    - E,F,G,H = 晚上 (18:10-22:00)

    輸出範例：
    - 不要星期三：「NOT (time GLOB '*三*')」
    - 不要星期五的下午：「NOT (time GLOB '*五*' AND (time GLOB '*5*' OR time GLOB '*6*' OR time GLOB '*7*' OR time GLOB '*8*'))」
    - 不要星期三和五：「NOT (time GLOB '*三*' OR time GLOB '*五*')」

    重要：只輸出 SQL WHERE 子句的條件部分（不包含 WHERE 關鍵字），不要包含任何解釋或 markdown。
    """
    if _is_json(user_input):
        tf = parse_timefilter_from_json(user_input)
        return constraints_to_where(tf)

    if not USE_SQLITE:
        return _pg_json_path(user_input)
    return _sqlite_glob_path(user_input)


@deprecated(SQLITE_DEPRECATION_MSG)
def _sqlite_glob_path(user_input: str) -> str:
    """舊 SQLite 路徑:讓 LLM 直接產 GLOB 字串比對 time 欄位。

    PostgreSQL 路徑改成先產結構化 JSON 再轉 WHERE(_pg_json_path + constraints.py),
    那條路好得多 —— 新功能請只做在那邊。
    """
    if USE_GOOGLE_AI:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY,
        )
    else:
        # num_predict 是保險絲,不是修法 —— 實測它擋不住失控生成(/api/generate 版本
        # 設了 256 仍然 300 秒不返回)。真正讓它停下來的是 reasoning=False。
        # 這裡設 512 純粹是上限:實測輸出 73-324 字、156 tokens,有三倍餘裕。
        llm = ChatOllama(
            model=MODEL, base_url=OLLAMA_HOST, reasoning=False, num_predict=512
        )

    prompt = f"""你是 SQL Agent，專門將時間限制轉換為 SQL WHERE 子句。

time 欄位格式說明：
- 一二三四五六日 = 星期一至星期日
- A,B = 早上 (06:10-08:00)
- 1,2,3,4 = 上午 (08:10-12:00)
- C,D = 中午 (12:10-14:00)
- 5,6,7,8 = 下午 (14:10-18:00)
- E,F,G,H = 晚上 (18:10-22:00)

範例：
- 五D56 = 星期五中午5-8節
- 三234 = 星期三上午2-4節
- 四CD = 星期四中午
- 六567日567 = 星期六、星期日 下午5-7節

用戶限制：{user_input}

只輸出 SQL WHERE 子句的條件部分（不包含 WHERE 關鍵字），不要任何思考過程、解釋或 markdown。
直接回一行 SQL 條件,例如：NOT (time GLOB '*三*')
"""

    return _invoke_logged(llm, prompt, "sqlite_glob")


def _pg_json_path(user_input: str) -> str:
    if USE_GOOGLE_AI:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY,
        )
    else:
        # num_predict 是保險絲,不是修法 —— 實測它擋不住失控生成(/api/generate 版本
        # 設了 256 仍然 300 秒不返回)。真正讓它停下來的是 reasoning=False。
        # 這裡設 512 純粹是上限:實測輸出 73-324 字、156 tokens,有三倍餘裕。
        llm = ChatOllama(
            model=MODEL, base_url=OLLAMA_HOST, reasoning=False, num_predict=512
        )

    prompt = f"""你是課程查詢 Agent，將使用者的時間/課程限制轉換成結構化 JSON。

可用的 JSON 欄位：
{{
  "include_times": [{{"weekday": int(1-7), "start_hour": int, "end_hour": int}}],
  "exclude_times": [{{"weekday": int(1-7), "start_hour": int, "end_hour": int}}],
  "lang": "中文" or "英文",
  "kind": int(1=必修, 2=選修, 3=通識, 4=體育),
  "point_min": float,
  "point_max": float,
  "unit": str(開課單位關鍵字)
}}

時間編碼(僅用於理解使用者輸入)：
- 星期：一二三四五六日 → 1-7
- 節次對照(start_hour/end_hour 用 24 小時制整數)：
    A=6~7   B=7~8
    1=8~9   2=9~10   3=10~11  4=11~12
    C=12~13 D=13~14
    5=14~15 6=15~16  7=16~17  8=17~18
    E=18~19 F=19~20  G=20~21  H=21~22
  例：三234 = 星期三 10~12 之間？不是 —— 2,3,4 是 9~10、10~11、11~12,合起來 9~12。

時段請「原封不動」用下面的邊界,不要自己推算：
- 早上 / 上午 → start_hour 8,  end_hour 12
- 中午        → start_hour 12, end_hour 13
- 下午        → start_hour 13, end_hour 18
- 晚上        → start_hour 18, end_hour 22

使用者只講時段、沒有指定星期幾時,**不要列舉星期**,直接省略 weekday 欄位。
列舉星期會讓查詢慢很多,而且語意不同。

範例：
- "不要星期三" → {{"exclude_times": [{{"weekday": 3, "start_hour": 0, "end_hour": 24}}]}}
- "只能上早上的課" → {{"include_times": [{{"start_hour": 8, "end_hour": 12}}]}}
- "我想上下午的課" → {{"include_times": [{{"start_hour": 13, "end_hour": 18}}]}}
- "英文授課,3學分以上" → {{"lang": "英文", "point_min": 3}}
- "不要星期三下午,選修" → {{"exclude_times": [{{"weekday": 3, "start_hour": 13, "end_hour": 18}}], "kind": 2}}

用戶限制：{user_input}

只輸出 JSON，不要任何其他文字。
"""

    text = _invoke_logged(llm, prompt, "pg_json")
    text = text.removeprefix("```json").removesuffix("```").strip()
    try:
        tf = parse_timefilter_from_json(text)
        return constraints_to_where(tf)
    except Exception as e:
        return f"ERROR: 無法解析時間條件({e})。輸入: {text}"
