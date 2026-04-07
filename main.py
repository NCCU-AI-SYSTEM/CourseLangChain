import logging
import pickle
import fire
import os
from dotenv import load_dotenv

from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.ensemble import EnsembleRetriever

from utils.prompt import get_prompt

# Load env
load_dotenv(override=True)

MODEL = os.getenv("MODEL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
os.environ.setdefault(
    "LANGFUSE_BASE_URL", os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000")
)
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", os.getenv("LANGFUSE_PUBLIC_KEY") or "")
os.environ.setdefault("LANGFUSE_SECRET_KEY", os.getenv("LANGFUSE_SECRET_KEY") or "")

logger = logging.getLogger("CourseLangchain")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class CourseLangChain:
    def __init__(
        self,
        pickleFile="vectorstore.pkl",
        cli=False,
    ) -> None:

        # Model Name Defination
        prompt = get_prompt()
        # logger.info("Prompt Template:\n" + prompt)

        # Initialize Langfuse
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        if public_key and secret_key:
            self.langfuse_client = get_client()
            self.langfuse_handler = CallbackHandler()
            logger.info(f"Langfuse tracing enabled: {os.getenv('LANGFUSE_BASE_URL')}")
        else:
            self.langfuse_client = None
            self.langfuse_handler = None
            logger.warning("Langfuse credentials not set - tracing disabled")

        with open(pickleFile, "rb") as f:
            retriever: EnsembleRetriever = pickle.load(f)

        model = OllamaLLM(
            model=MODEL,
            base_url=OLLAMA_HOST,
            stop=["<|eot_id|>"],
        )

        def format_docs(docs):
            return "\n".join(f"- {doc.page_content}" for doc in docs)

        self.chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        logger.info("Chain ready.")

    def invoke(self, input) -> str:
        config = {"callbacks": [self.langfuse_handler]} if self.langfuse_handler else {}
        return self.chain.invoke(input, config=config)

    async def astream(self, input):
        config = {"callbacks": [self.langfuse_handler]} if self.langfuse_handler else {}
        async for chunk in self.chain.astream(input, config=config):
            yield chunk


async def main():
    chain = CourseLangChain(cli=True)
    while True:
        query = input("User:")
        if not query.strip():
            continue
        print("Bot:")
        async for chunk in chain.astream(query):
            print(chunk, end="", flush=True)
        print()
        if chain.langfuse_client:
            get_client().flush()


if __name__ == "__main__":
    fire.Fire(main)
