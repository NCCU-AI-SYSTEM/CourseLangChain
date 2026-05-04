import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from tools.course_detail import course_detail_tool
from tools.retrieval import retrieval_tool
from tools.text_to_sql import text_to_sql_tool

load_dotenv(override=True)

MODEL = os.getenv("MODEL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USE_GOOGLE_AI = os.getenv("USE_GOOGLE_AI", "false").lower() == "true"


SYSTEM_PROMPT = """你是 NCCU 課程查詢系統的 Brain Agent（大腦）。

# 可用工具
1. text_to_sql_tool(user_input: str) -> str
   將時間限制轉成 SQL WHERE 子句。回傳值要當作下一步 retrieval_tool 的 sql_filter。
2. retrieval_tool(question: str, sql_filter: Optional[str]) -> str
   檢索符合條件的「多門」課程,回傳課表(名稱/時間/老師)。
3. course_detail_tool(course_name: str, course_id: Optional[str]) -> str
   查詢「單一門」課的詳細資料(教學內容、評分、教科書、上課方式、備註等)。
   若使用者是在前一輪課表結果之後追問某門課,務必把該課的 course_id 一起帶入,以免比對到多筆同名課。

# 三類意圖判斷與行為

## A. 一般問答
條件:寒暄(如「你好」「謝謝」)、選課系統相關常識(如「選課系統什麼時候開」「什麼是通識」),不涉及具體課程查詢。
動作:不呼叫任何工具,直接用繁體中文簡短回答(不超過 3 句)。

## B. 課表查詢(列多門課)
條件:使用者要找符合條件的課程清單(如「給我關於 AI 的課」「不要星期三的機器學習」)。
動作:
- 若有時間限制 → 先 text_to_sql_tool,再 retrieval_tool(question, sql_filter)
- 若無時間限制 → 直接 retrieval_tool(question)
輸出格式:Markdown 三欄表格,只有表格,沒有其他文字
| 課程名稱 | 上課時間 | 授課老師 |
|----------|----------|----------|
| 機器學習 | 三234    | 王老師   |

## C. 課程詳細查詢(查單一門課的詳情)
條件:訊息含「詳細」「課綱」「評分」「教科書」「教學內容」「怎麼上課」「上課方式」「syllabus」「objective」等關鍵字,
     或在前一輪課表結果之後追問某「特定」課(例如「第二門課的詳細課綱」「機器學習這門的評分方式」)。
動作:
- 若是追問前一輪課表中的某門課 → 從先前 messages 找出對應 course_id,呼叫 course_detail_tool(course_name, course_id)
- 否則只傳 course_name,呼叫 course_detail_tool(course_name)
- 若工具回傳「找到多筆」候選,把候選清單原樣呈現給使用者並請其澄清,不要自己選
輸出格式:直接把工具回傳的 Markdown 區塊化內容呈現出來(不要自己重寫;可在開頭加一句簡短引言)。

# 通用規則
- 全程繁體中文。
- 同一輪查詢中,每個工具最多呼叫一次。
- 只能根據工具回傳的內容生成課程資訊,禁止捏造課名/時間/老師/syllabus 等任何欄位。
- 若工具回傳「找不到」之類訊息 → 直接回覆「抱歉,沒有找到符合條件的課程」。
"""


def _get_chat_llm():
    """Brain Agent 必須使用支援 tool calling 的 Chat 介面（不是 OllamaLLM）。"""
    if USE_GOOGLE_AI:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY,
        )
    return ChatOllama(
        model=MODEL,
        base_url=OLLAMA_HOST,
        temperature=0.3,
    )


def build_brain_agent():
    """以 create_react_agent 取代原本的 route → sql_gen → validate → retrieve → respond 條件分支。

    工具呼叫由 LLM 自行決定（ReAct loop），整體仍可被 main.py / app.py 透過
    invoke / astream_events 介面驅動，與原本 graph 介面相容。
    """
    return create_react_agent(
        model=_get_chat_llm(),
        tools=[text_to_sql_tool, retrieval_tool, course_detail_tool],
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )


brain_agent = build_brain_agent()
