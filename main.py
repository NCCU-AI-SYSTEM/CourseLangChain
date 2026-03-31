import logging
import pickle
import fire
import os
from dotenv import load_dotenv

from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.ensemble import EnsembleRetriever

from utils.prompt import get_prompt

# Load env
load_dotenv(override=True)

MODEL = os.getenv("MODEL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

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
        return self.chain.invoke(input)


async def main():
    chain = CourseLangChain(cli=True)
    while True:
        query = input("User:")
        print("Bot:")
        async for chunk in chain.chain.astream(query):
            print(chunk, end="", flush=True)


if __name__ == "__main__":
    fire.Fire(main)