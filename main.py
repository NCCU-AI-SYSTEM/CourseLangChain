import logging
import os

import fire
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from agents.brain_agent import brain_agent

load_dotenv(override=True)

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
        logger.info("Brain Agent (ReAct) ready.")

    def invoke(self, user_input: str) -> str:
        config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
        result = self.brain_agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        messages = result.get("messages", [])
        return messages[-1].content if messages else ""

    async def astream(self, user_input: str):
        config = (
            {"callbacks": [langfuse_handler], "recursion_limit": 50}
            if langfuse_handler
            else {"recursion_limit": 50}
        )
        async for event in self.brain_agent.astream_events(
            {"messages": [{"role": "user", "content": user_input}]},
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
