import logging
import fire
import os
from dotenv import load_dotenv

from langfuse import get_client
from langfuse.langchain import CallbackHandler

from agent.graph import graph, streaming_graph
from agent.state import AgentState

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


def create_initial_state(user_input: str) -> AgentState:
    return {
        "user_input": user_input,
        "needs_sql": False,
        "sql_filter": None,
        "sql_thoughts": "",
        "sql_validation": True,
        "sql_validation_msg": "",
        "sql_retry_count": 0,
        "courses": None,
        "retrieval_thoughts": "",
        "final_response": "",
        "messages": [],
    }


class CourseLangGraph:
    def __init__(self, cli=False) -> None:
        self.agent = graph
        self.streaming_agent = streaming_graph
        logger.info("Agent ready.")

    def invoke(self, user_input: str) -> str:
        """Synchronous invoke - returns complete response."""
        initial_state = create_initial_state(user_input)
        config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
        result = self.agent.invoke(initial_state, config=config)
        return result.get("final_response", "")

    async def astream(self, user_input: str):
        """Async streaming invoke using LangGraph stream events."""
        initial_state = create_initial_state(user_input)
        config = (
            {"callbacks": [langfuse_handler], "recursion_limit": 50}
            if langfuse_handler
            else {"recursion_limit": 50}
        )

        async for event in self.streaming_agent.astream_events(
            initial_state,
            config=config,
        ):
            event_type = event.get("event", "")
            metadata = event.get("metadata", {})
            node_name = metadata.get("langgraph_node", "")

            if event_type == "on_llm_stream" and node_name == "respond":
                data = event.get("data", {})
                chunk = data.get("chunk", {})

                content = None
                if hasattr(chunk, "content"):
                    content = chunk.content
                elif hasattr(chunk, "text"):
                    content = chunk.text
                elif isinstance(chunk, dict):
                    content = chunk.get("content") or chunk.get("text")

                if content:
                    yield content


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
