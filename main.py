import logging
import os
import uuid

import fire
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from agents.brain_agent import brain_agent
from harness import SafeAgentExecutor, sanitize_input, validate_output

load_dotenv(override=True)

# L1 擋下不安全輸入時回給使用者的訊息
_REJECT_MSG = "抱歉,您的輸入無法處理,請改用一般的課程查詢方式重新提問。"
# L3 驗證未通過時回給使用者的訊息(寧可不回課表,也不給錯誤課表)
_INVALID_OUTPUT_MSG = "抱歉,系統產生的課表未通過正確性檢查,請調整條件後再試一次。"

os.environ.setdefault(
    "LANGFUSE_BASE_URL", os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000")
)
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", os.getenv("LANGFUSE_PUBLIC_KEY") or "")
os.environ.setdefault("LANGFUSE_SECRET_KEY", os.getenv("LANGFUSE_SECRET_KEY") or "")

logger = logging.getLogger("CourseLangGraph")
logger.setLevel(logging.DEBUG)

public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")
if public_key and secret_key:
    langfuse_handler = CallbackHandler()
    logger.info(f"Langfuse tracing enabled: {os.getenv('LANGFUSE_BASE_URL')}")
else:
    langfuse_handler = None
    logger.warning("Langfuse credentials not set - tracing disabled")

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class CourseLangGraph:
    def __init__(self, cli: bool = False) -> None:
        self.brain_agent = brain_agent
        # L2:把 ReAct graph 包進安全外殼(step / timeout / exception 防護)
        self.executor = SafeAgentExecutor(brain_agent)
        logger.info("Brain Agent (ReAct) ready,三層 harness 已啟用。")

    def _base_config(self, thread_id: str | None = None) -> dict:
        """組 LangGraph config。

        thread_id 是對話記憶的鍵:同一個 thread_id 的多輪才會共用歷史。
        沒帶時發一個一次性 id——agent 掛了 checkpointer,少了 thread_id 會直接 raise,
        用一次性 id 等同「這輪無記憶」,比讓呼叫端炸掉好。
        """
        config: dict = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
        config["configurable"] = {"thread_id": thread_id or f"ephemeral-{uuid.uuid4()}"}
        return config

    def invoke(self, user_input: str, thread_id: str | None = None) -> str:
        # L1:輸入清理
        text, is_safe = sanitize_input(user_input)
        if not is_safe:
            logger.warning("input rejected by L1 sanitizer")
            return _REJECT_MSG

        # L2:安全外殼執行(永不 raise)
        result = self.executor.run(text, config=self._base_config(thread_id))
        output = result["output"]
        logger.info(
            "agent done: steps=%s elapsed=%ss error=%s",
            result["steps"], result["elapsed_sec"], result["error"],
        )
        if result["error"]:
            return output  # 已是友善訊息

        # L3:輸出驗證(衝堂等)
        ok, msg = validate_output(output)
        if not ok:
            logger.warning("output rejected by L3: %s", msg)
            return _INVALID_OUTPUT_MSG
        return output

    async def astream(self, user_input: str, thread_id: str | None = None):
        # L1:輸入清理(串流路徑同樣先擋)
        text, is_safe = sanitize_input(user_input)
        if not is_safe:
            logger.warning("input rejected by L1 sanitizer (stream)")
            yield _REJECT_MSG
            return

        # L2 的 step 上限沿用 executor 的 recursion_limit;wall-clock timeout 在串流下
        # 不套用(會切斷已輸出的 token)。L3 輸出驗證亦因逐 token 串流無法即時套用,
        # 改在非串流 invoke 路徑把關。
        config = {
            **self._base_config(thread_id),
            "recursion_limit": self.executor.recursion_limit,
        }
        async for event in self.brain_agent.astream_events(
            {"messages": [{"role": "user", "content": text}]},
            config=config,
            version="v2",
        ):
            if event.get("event") == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                content = getattr(chunk, "content", None)
                if content:
                    yield content
        logger.info("Brain Agent execution completed")


async def main():
    agent = CourseLangGraph(cli=True)
    # 整個 REPL 共用一個 thread_id,多輪追問才接得起來(輸入 new 可開新對話)
    thread_id = f"cli-{uuid.uuid4()}"
    while True:
        query = input("User: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        if query.lower() == "new":
            thread_id = f"cli-{uuid.uuid4()}"
            print("(已開始新對話,先前的上下文不再沿用)\n")
            continue
        print("Bot:")
        result = agent.invoke(query, thread_id=thread_id)
        print(result)
        print()
    if langfuse_handler:
        get_client().flush()


if __name__ == "__main__":
    fire.Fire(main)
