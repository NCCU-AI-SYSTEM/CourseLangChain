import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM

load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("MODEL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
USE_GOOGLE_AI = os.getenv("USE_GOOGLE_AI", "false").lower() == "true"


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
    - 不要星期三：「NOT (time GLOB '*三*')»
    - 不要星期五的下午：「NOT (time GLOB '*五*' AND (time GLOB '*5*' OR time GLOB '*6*' OR time GLOB '*7*' OR time GLOB '*8*'))»
    - 不要星期三和五：「NOT (time GLOB '*三*' OR time GLOB '*五*')»

    重要：只輸出 SQL WHERE 子句的條件部分（不包含 WHERE 關鍵字），不要包含任何解釋或 markdown。
    """
    if USE_GOOGLE_AI:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY,
        )
    else:
        llm = OllamaLLM(model=MODEL, base_url=OLLAMA_HOST)

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

請用 ReAct 格式思考：
1. 理解：用戶想要排除哪些時段
2. 分析：對應的 SQL 邏輯
3. 生成：SQL WHERE 子句
4. 反思：這個 SQL 是否正確？

重要：只輸出 SQL WHERE 子句的條件部分（不包含 WHERE 關鍵字），不要包含任何解釋或 markdown。
"""

    result = llm.invoke(prompt)
    if hasattr(result, "content"):
        return result.content.strip()
    return str(result).strip()
