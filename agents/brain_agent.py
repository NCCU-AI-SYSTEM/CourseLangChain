import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, trim_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from tools.course_detail import course_detail_tool
from tools.my_schedule import my_schedule_tool
from tools.query_courses import query_courses_tool
from tools.schedule_tool import schedule_tool
from tools.text_to_sql import text_to_sql_tool
from tools.user_profile import user_profile_tool

load_dotenv(override=True)

MODEL = os.getenv("MODEL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USE_GOOGLE_AI = os.getenv("USE_GOOGLE_AI", "false").lower() == "true"


SYSTEM_PROMPT = """你是 NCCU 課程查詢系統的 Brain Agent（大腦）。

# 可用工具
1. text_to_sql_tool(user_input: str) -> str
   將時間限制轉成 SQL WHERE 子句。回傳值要當作下一步 query_courses_tool 的 sql_filter。
2. query_courses_tool(keyword: str, top_k: int=10, sql_filter: str="") -> str
   檢索「多門」課程候選,回傳每門的「13 位 course_id」、課名、時間、老師、學分。
   查課與排課都用這個。有時間限制時把 text_to_sql_tool 的結果帶入 sql_filter;排課時 top_k 建議 20。
3. course_detail_tool(course_name: str, course_id: Optional[str]) -> str
   查詢「單一門」課的詳細資料(教學內容、評分、教科書、上課方式、備註等)。
   若使用者是在前一輪課表結果之後追問某門課,務必把該課的 course_id 一起帶入,以免比對到多筆同名課。
4. schedule_tool(course_ids: str, min_credits: float, max_credits: float, avoid_weekdays: str, max_results: int) -> str
   排課:接一串逗號分隔的 course_id,排出「無衝堂、學分達標」的課表方案。
   course_ids 一律從 query_courses_tool 的結果原樣抄過來,不可自己編造。avoid_weekdays 用中文字逗號分隔(如 "五")。
5. my_schedule_tool(action: str, course_id: str) -> str
   讀寫使用者「目前已排定的課表」(就是畫面右側課表面板那一份,使用者可能手動改過)。
   action: view 查看 / add 加課 / remove 移除 / clear 清空。add、remove 需要 13 位 course_id。
   使用者問「我現在排了什麼 / 幾學分」、要求把課加進或移出課表時用。
   加課若衝堂,工具會自動移除衝堂舊課——這是預期行為,照實轉述即可。
6. user_profile_tool(record_path: str) -> str
   讀使用者自帶的成績單,回「去識別化」的修課狀況(已修課、本學期已選、畢業缺口)。這是可選功能。
   當使用者要「個人化排課」、提到自身修課狀況、或要避免重複推薦已修過的課時呼叫。
   若回傳「未提供成績單」之類訊息,就當作沒有這項資訊、照常排課,不要追問或要求使用者提供。

# 三類意圖判斷與行為

## A. 一般問答
條件:寒暄(如「你好」「謝謝」)、選課系統相關常識(如「選課系統什麼時候開」「什麼是通識」),不涉及具體課程查詢。
動作:不呼叫任何工具,直接用繁體中文簡短回答(不超過 3 句)。

## B. 課表查詢(列多門課)
條件:使用者要找符合條件的課程清單(如「給我關於 AI 的課」「不要星期三的機器學習」)。
動作:
- 若有時間限制 → 先 text_to_sql_tool,再 query_courses_tool(keyword, sql_filter=...)
- 若無時間限制 → 直接 query_courses_tool(keyword)
輸出格式:Markdown 三欄表格,只有表格,沒有其他文字(course_id 不必顯示)
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

## D. 排課(幫忙排出一週課表)
條件:使用者要系統「幫忙排課表 / 安排課表 / 湊學分」,常帶學分數、避開時段、想修的主題等
     (如「幫我排 12 學分的課,避開週五」「我想修 AI 相關的課排成課表」)。
動作(course_id 是橋樑):
0.(可選,個人化)若使用者要個人化排課、或提到自身修課狀況 → 先 user_profile_tool。
   - 有拿到 profile:用其「學分」推算 min/max、把「已修過 + 本學期已選」當排除清單(別把這些課排進去),畢業缺口可當 query 的關鍵字方向。
     排除以 profile 給的「排除課號」為準:course_id 的**後 9 碼**等於任一排除課號,就是使用者修過的同一門課,不可排進課表。
   - 回「未提供成績單」就忽略這步,照常排課,不要追問使用者要資料。
1. query_courses_tool(keyword, top_k=20) 取得候選課程清單(內含每門的 course_id)
   - 排課要多一點候選才排得開,top_k 建議設 20
2. 從上一步結果把要納入的 course_id 用逗號接起來(若有排除清單,先比對後 9 碼剔除已修/已選的課),呼叫
   schedule_tool(course_ids, min_credits, max_credits, avoid_weekdays)
   - 使用者給「18 學分」這類單一數字時,可設 min_credits 略低、max_credits 等於該值(如 15 與 18)
   - 沒講避開星期就傳空字串
規則:course_id 一律照抄,禁止自己編。schedule_tool 會自動排除「時間未定」的課、且同名課只取一門,你不必自己過濾。
若 query 候選太少排不出,可再 query 一次不同關鍵字補充候選。
輸出格式:直接把 schedule_tool 回傳的 Markdown 方案呈現出來,可加一句簡短引言說明取捨。

# 通用規則
- 全程繁體中文。
- 查詢類(意圖 B/C)同一輪每個工具最多呼叫一次;排課(意圖 D)可先 query 再 schedule,必要時補一次 query。
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


# 送進 LLM 的歷史訊息則數上限(不含 system prompt)。
# 一個「排課回合」約 4~6 則(human + AI tool_call + tool 結果 + AI 回答),
# 40 則約當最近 7~9 回合。
#
# 這個上限不是為了「塞得下」——gemma4:31b-cloud 的 context 是 262144 token,
# 綽綽有餘。真正的理由是:每輪都會塞進一份 20 門課的候選清單,堆多了會有好幾份
# 結構幾乎相同、只有數字不同的課表,模型容易抄錯 course_id;順帶也省 cloud 模型的 token 費。
MAX_HISTORY_MESSAGES = 40


def _trim_history(state: dict) -> dict:
    """pre_model_hook:只裁「送進 LLM 的」訊息,不動 checkpointer 存的完整歷史。

    回傳 `llm_input_messages` 而非 `messages`,是為了不覆寫 state——
    完整對話仍留在 checkpointer 裡,只有本次餵給模型的視窗被裁短。
    """
    trimmed = trim_messages(
        state["messages"],
        token_counter=len,  # 以「訊息則數」計,不實際算 token(省一次 tokenizer 往返)
        max_tokens=MAX_HISTORY_MESSAGES,
        strategy="last",
        # 確保視窗開頭是 human:否則可能切出孤兒 ToolMessage(對應的 tool_call 被裁掉),
        # 多數 provider 會直接回 400。
        start_on="human",
        # system prompt 由 create_react_agent 的 prompt= 另外注入,不在 messages 裡
        include_system=False,
        allow_partial=False,
    )
    return {"llm_input_messages": trimmed}


def build_brain_agent():
    """以 create_react_agent 取代原本的 route → sql_gen → validate → retrieve → respond 條件分支。

    工具呼叫由 LLM 自行決定（ReAct loop），整體仍可被 main.py / app.py 透過
    invoke / astream_events 介面驅動，與原本 graph 介面相容。

    對話記憶:掛 `InMemorySaver`,呼叫時需帶 `config={"configurable": {"thread_id": ...}}`,
    同一個 thread_id 的多輪對話才接得起來(意圖 C 的「從先前 messages 找出 course_id」
    依賴這個)。刻意用 in-memory 而非 SqliteSaver——歷史裡含 user_profile_tool 解析出的
    修課狀況,落地存檔會牴觸「僅本次使用、不儲存」的隱私承諾。程序重啟即清空。
    """
    return create_react_agent(
        model=_get_chat_llm(),
        tools=[
            text_to_sql_tool,
            query_courses_tool,
            course_detail_tool,
            schedule_tool,
            user_profile_tool,
            my_schedule_tool,
        ],
        prompt=SystemMessage(content=SYSTEM_PROMPT),
        pre_model_hook=_trim_history,
        checkpointer=InMemorySaver(),
    )


brain_agent = build_brain_agent()
