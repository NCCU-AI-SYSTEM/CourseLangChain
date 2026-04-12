import os
import sqlite3
import asyncio
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from typing import Literal, AsyncIterator, Union
from .state import AgentState

MODEL = os.getenv("MODEL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USE_GOOGLE_AI = os.getenv("USE_GOOGLE_AI", "false").lower() == "true"


def get_llm():
    if USE_GOOGLE_AI:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY,
        )
    return OllamaLLM(model=MODEL, base_url=OLLAMA_HOST)


def get_respond_llm():
    if USE_GOOGLE_AI:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY,
        )
    return OllamaLLM(model=MODEL, base_url=OLLAMA_HOST)


router_prompt = """你是課程查詢系統的 Router Agent（大腦）。

可用工具：
1. text_to_sql_tool - 將時間限制轉換為 SQL WHERE 子句
2. retrieval_tool - 執行課程檢索（可接收 sql_filter 參數）

工作流程：
1. 分析用戶輸入，判斷是否需要時間過濾
2. 若需要，呼叫 text_to_sql_tool
3. 呼叫 retrieval_tool（可傳入 sql_filter）
4. 接收檢索結果，生成最終回答

重要提醒：
- Tool 的結果會回傳給你，由你決定下一步
- 請用繁體中文回應
- 最後將課程整理成表格格式（名稱、時間、老師）
"""

sql_validation_prompt = """你是 Validator，檢查 SQL 是否正確。

原始用戶限制：{user_input}
生成的 SQL：{sql}

請檢查：
1. 語法是否正確？（GLOB/LIKE 使用得當）
2. 邏輯是否正確？（「不要星期三」→ NOT time GLOB '*三*'）
3. 完整性？（是否漏掉任何限制）

輸出格式：
valid: true 或 false
reason: 原因說明
"""


def route(state: AgentState) -> AgentState:
    llm = get_llm()

    user_input = state["user_input"]
    messages = state.get("messages", [])

    route_prompt = f"""{router_prompt}

用戶輸入：{user_input}

請分析並回應：
1. 用戶是否提及時間限制？（如「不要星期三」、「排除星期五下午」）
2. 是否需要呼叫工具？

回應格式（JSON）：
{{
    "needs_sql": true/false,
    "reasoning": "分析原因"
}}
"""

    response = llm.invoke(route_prompt)
    response_text = response.strip() if hasattr(response, "strip") else str(response)

    has_time_keywords = (
        "不" in user_input
        or "排除" in user_input
        or "不要" in user_input
        or "週" in user_input
        or "星期" in user_input
    )
    needs_sql = has_time_keywords or "true" in response_text.lower()

    return {
        "needs_sql": needs_sql,
        "messages": messages
        + [HumanMessage(content=user_input), AIMessage(content=response_text)],
    }


def sql_gen(state: AgentState) -> AgentState:
    from tools.text_to_sql import text_to_sql_tool

    llm = get_llm()
    user_input = state["user_input"]
    messages = state.get("messages", [])

    if state["needs_sql"]:
        sql_result = text_to_sql_tool.invoke({"user_input": user_input})

        reflect_prompt = f"""請反思以下 SQL 是否正確：

生成的 SQL：{sql_result}

檢查項目：
1. 語法：GLOB/LIKE 使用是否正確？
2. 邏輯：「不要星期三」是否轉換為 NOT time GLOB '*三*'？
3. 完整性：是否涵蓋用戶所有限制？

反思："""

        reflection = llm.invoke(reflect_prompt)

        return {
            "sql_filter": sql_result,
            "sql_thoughts": f"生成 SQL: {sql_result}\n反思: {reflection}",
            "sql_retry_count": state.get("sql_retry_count", 0) + 1,
            "messages": messages + [ToolMessage(content=sql_result, tool_call_id="")],
        }
    else:
        return {
            "sql_filter": None,
            "sql_thoughts": "用戶無時間限制，跳過 SQL 生成",
            "messages": messages,
        }


def validate_sql(state: AgentState) -> AgentState:
    llm = get_llm()

    user_input = state["user_input"]
    sql_filter = state.get("sql_filter")
    messages = state.get("messages", [])

    if not sql_filter:
        return {
            "sql_validation": True,
            "sql_validation_msg": "無 SQL 需要驗證",
            "messages": messages,
        }

    is_valid = False
    validation_msg = ""

    try:
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        test_query = f"SELECT * FROM COURSE WHERE {sql_filter} LIMIT 1"
        cursor.execute(test_query)
        cursor.fetchone()
        conn.close()
        is_valid = True
        validation_msg = "SQL 語法正確，可以執行"
    except sqlite3.Error as e:
        validation_msg = f"SQL 執行錯誤: {e}"

    return {
        "sql_validation": is_valid,
        "sql_validation_msg": validation_msg,
        "messages": messages + [AIMessage(content=validation_msg)],
    }


def retrieve(state: AgentState) -> AgentState:
    from tools.retrieval import retrieval_tool

    user_input = state["user_input"]
    sql_filter = state.get("sql_filter")
    messages = state.get("messages", [])

    retrieval_result = retrieval_tool.invoke(
        {"question": user_input, "sql_filter": sql_filter}
    )

    return {
        "courses": retrieval_result,
        "retrieval_thoughts": f"檢索完成，共取得課程資訊",
        "messages": messages + [ToolMessage(content=retrieval_result, tool_call_id="")],
    }


async def astream_respond(state: AgentState) -> AsyncIterator[dict]:
    """Streaming version of respond node - yields chunks as they come."""
    user_input = state["user_input"]
    courses = state.get("courses", "找不到課程")
    sql_filter = state.get("sql_filter")
    messages = state.get("messages", [])

    response_prompt = f"""你是課程查詢系統的助理。

用戶需求：{user_input}

時間過濾條件：{sql_filter if sql_filter else "無"}

檢索到的課程：
{courses}

請將課程整理成表格格式：
| 課程名稱 | 上課時間 | 授課老師 |
|----------|----------|----------|
| ...      | ...      | ...      |

如果找不到課程，請說明原因。

請用繁體中文回應。
"""

    llm = get_respond_llm()

    full_response = ""
    async for chunk in llm.astream(response_prompt):
        chunk_text = chunk.content if hasattr(chunk, "content") else str(chunk)
        full_response += chunk_text
        yield {"final_response": full_response, "final_response_chunk": chunk_text}

    yield {
        "final_response": full_response,
        "messages": messages + [AIMessage(content=full_response)],
    }


def respond(state: AgentState) -> AgentState:
    """Synchronous version - uses ainvoke."""
    user_input = state["user_input"]
    courses = state.get("courses", "找不到課程")
    sql_filter = state.get("sql_filter")
    messages = state.get("messages", [])

    response_prompt = f"""你是課程查詢系統的助理。

用戶需求：{user_input}

時間過濾條件：{sql_filter if sql_filter else "無"}

檢索到的課程：
{courses}

請將課程整理成表格格式：
| 課程名稱 | 上課時間 | 授課老師 |
|----------|----------|----------|
| ...      | ...      | ...      |

如果找不到課程，請說明原因。

請用繁體中文回應。
"""

    llm = get_respond_llm()
    response = llm.invoke(response_prompt)

    if hasattr(response, "content"):
        response_text = response.content
    elif hasattr(response, "strip"):
        response_text = response.strip()
    else:
        response_text = str(response)

    return {
        "final_response": response_text,
        "messages": messages + [AIMessage(content=response_text)],
    }


def should_regenerate_sql(state: AgentState) -> Literal["sql_gen", "retrieve"]:
    retry_count = state.get("sql_retry_count", 0)

    if not state.get("sql_validation", True) and state.get("sql_filter"):
        if retry_count < 3:
            return "sql_gen"
        else:
            return "retrieve"
    return "retrieve"


def should_skip_sql(state: AgentState) -> Literal["sql_gen", "retrieve"]:
    if state.get("needs_sql", False):
        return "sql_gen"
    return "retrieve"
