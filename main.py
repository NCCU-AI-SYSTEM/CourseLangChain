import logging
import fire
import os
from dotenv import load_dotenv

from agent.graph import graph, streaming_graph
from agent.state import AgentState

load_dotenv(override=True)

logger = logging.getLogger("CourseLangGraph")
logger.setLevel(logging.DEBUG)

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
        result = self.agent.invoke(initial_state)
        return result.get("final_response", "")

    async def astream(self, user_input: str):
        """Async streaming invoke using LangGraph stream events."""
        initial_state = create_initial_state(user_input)

        async for event in self.streaming_agent.astream_events(
            initial_state,
            config={"recursion_limit": 50},
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


if __name__ == "__main__":
    fire.Fire(main)
