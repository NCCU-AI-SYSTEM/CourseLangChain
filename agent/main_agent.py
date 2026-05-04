"""ReAct agent 入口。

組裝順序:
    sanitize_input (L1) → ChatPromptTemplate + create_react_agent + AgentExecutor
                       → SafeAgentExecutor (L2) → validate_output (L3)

協作者通常只要做兩件事就能上線新功能:
- 在 `tools/` 加新檔案 + 在 `tools/registry.py` 註冊
- (可選)在新 tool 裡 `harness.register_validator(...)` 掛 L3 檢查

模型選擇用環境變數 `AGENT_MODEL`,fallback 到 `MODEL`(舊 graph 共用)。建議目前設成
`gemma4:31b-cloud`(已存在於本機 Ollama 的 cloud route)。

Langfuse:若 `LANGFUSE_*` 三把 key 齊備,所有 ReAct 步驟自動上傳追蹤,
Langfuse UI 上可以看到 prompt / tool call / response 的完整 trace。

CLI 煙霧測試:
    uv run python agent/main_agent.py "你好"
"""
from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from harness import (
    SafeAgentExecutor,
    sanitize_input,
    validate_output,
)
from tools.registry import all_tools

load_dotenv(override=True)
logger = logging.getLogger(__name__)

AGENT_MODEL = os.getenv("AGENT_MODEL") or os.getenv("MODEL") or "gemma4:31b-cloud"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
USE_GOOGLE_AI = os.getenv("USE_GOOGLE_AI", "false").lower() == "true"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def _build_langfuse_handler():
    """嘗試建立 Langfuse callback handler。三把 key 不齊就回 None。"""
    if not all(
        os.getenv(k)
        for k in ("LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_BASE_URL")
    ):
        return None
    try:
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except Exception:
        logger.exception("failed to init Langfuse handler")
        return None


_LANGFUSE_HANDLER = _build_langfuse_handler()

SYSTEM_PROMPT = """You are an NCCU course assistant. For chit-chat, answer directly. For course queries or scheduling, you MUST call a tool — never fabricate course data. If a tool returns text starting with "ERROR:", tell the user and suggest simplifying; do not retry the same tool more than twice.

Available tools:
{tools}

Use this format:
Question: the input question
Thought: what to do
Action: one of [{tool_names}]
Action Input: input for the tool
Observation: tool result
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer

Question: {input}
Thought:{agent_scratchpad}"""


def _get_llm() -> Any:
    if USE_GOOGLE_AI:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY,
        )
    return ChatOllama(
        model=AGENT_MODEL,
        base_url=OLLAMA_HOST,
        temperature=0.2,
    )


@lru_cache(maxsize=1)
def build_agent(
    max_steps: int = 5,
    timeout_sec: float = 120.0,
) -> SafeAgentExecutor:
    """組裝 ReAct agent + L2 safeguard。回傳的物件可重複使用,thread-safe 與否依 LangChain。"""
    tools = all_tools()
    llm = _get_llm()
    # Warm-up:在 WSL2 + langchain_ollama 環境觀察到,直接走 AgentExecutor 的第一個呼叫
    # 會 ConnectTimeout(134s 後失敗),但先做一次 llm.invoke 之後就穩定。原因疑似是
    # langchain_ollama 內部 httpx Client 的 lazy init 與 ReAct bind(stop=...) path 互動不良。
    try:
        llm.invoke("ping")
    except Exception:
        logger.warning("LLM warm-up failed; agent may fail on first real call", exc_info=True)
    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
    react_agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=react_agent,
        tools=tools,
        verbose=False,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    return SafeAgentExecutor(executor, max_steps=max_steps, timeout_sec=timeout_sec)


def ask(user_input: str, session_id: str | None = None) -> dict:
    """完整 pipeline:L1 → agent → L2 → L3。永不 raise。

    Args:
        user_input: 原始使用者輸入
        session_id: (可選)Langfuse trace 分組用的 session id

    Returns:
        {"output": str, "error": Optional[str], "steps": int, "elapsed_sec": float}
    """
    text, ok = sanitize_input(user_input)
    if not ok:
        return {
            "output": "您的輸入無法處理,請改寫後重試。",
            "error": "sanitization",
            "steps": 0,
            "elapsed_sec": 0.0,
        }

    agent = build_agent()

    config: dict[str, Any] | None = None
    if _LANGFUSE_HANDLER is not None:
        metadata: dict[str, Any] = {"layer": "ask", "user_input_len": len(text)}
        if session_id:
            metadata["langfuse_session_id"] = session_id
        config = {"callbacks": [_LANGFUSE_HANDLER], "metadata": metadata}

    result = agent.run(text, config=config)

    if result.get("error") is None:
        valid, msg = validate_output(result["output"])
        if not valid:
            result["output"] = "系統內部驗證未通過,已記錄。請稍後重試。"
            result["error"] = f"output_validation: {msg}"

    if _LANGFUSE_HANDLER is not None:
        try:
            from langfuse import get_client
            get_client().flush()
        except Exception:
            logger.exception("langfuse flush failed")

    return result


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "你好"
    out = ask(q)
    print(out)
