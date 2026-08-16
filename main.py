import logging
import os

import fire
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from agents.brain_agent import brain_agent
from harness import SafeAgentExecutor, sanitize_input, validate_output
from paths import check_contract

load_dotenv(override=True)  # paths.py already did this; kept for direct runs

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
        # 資料成品與 contract.yaml 必須對得上,否則寧可不啟動(memo,每個 process 一次)
        check_contract()
        self.brain_agent = brain_agent
        # L2:把 ReAct graph 包進安全外殼(step / timeout / exception 防護)
        self.executor = SafeAgentExecutor(brain_agent)
        logger.info("Brain Agent (ReAct) ready,三層 harness 已啟用。")

    def _base_config(self) -> dict:
        return {"callbacks": [langfuse_handler]} if langfuse_handler else {}

    def invoke(self, user_input: str) -> str:
        # L1:輸入清理
        text, is_safe = sanitize_input(user_input)
        if not is_safe:
            logger.warning("input rejected by L1 sanitizer")
            return _REJECT_MSG

        # L2:安全外殼執行(永不 raise)
        result = self.executor.run(text, config=self._base_config())
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

    async def astream(self, user_input: str):
        # L1:輸入清理(串流路徑同樣先擋)
        text, is_safe = sanitize_input(user_input)
        if not is_safe:
            logger.warning("input rejected by L1 sanitizer (stream)")
            yield _REJECT_MSG
            return

        # L2 的 step 上限沿用 executor 的 recursion_limit;wall-clock timeout 在串流下
        # 不套用(會切斷已輸出的 token)。L3 輸出驗證亦因逐 token 串流無法即時套用,
        # 改在非串流 invoke 路徑把關。
        config = {**self._base_config(), "recursion_limit": self.executor.recursion_limit}
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
    while True:
        query = input("User: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        print("Bot:")
        result = agent.invoke(query)
        print(result)
        print()
    if langfuse_handler:
        get_client().flush()


if __name__ == "__main__":
    fire.Fire(main)
