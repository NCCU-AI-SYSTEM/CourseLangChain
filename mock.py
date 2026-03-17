from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

MODEL = os.getenv("MODEL")

if MODEL is None:
    raise Exception("MODEL environment variable is not set")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

prompt = PromptTemplate(
    template=("<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. \n"
        "Use the following pieces of retrieved context to answer the question. Create a table that lists course schedule. \n"
        "The table should have three columns: name, time, teacher. Please align the entries to the left. \n"
        "Remember to respond with a table, and please add a paragraph before the table to explain. Even if there is unmeaningful request, "
        "you should still generate a mock table with moch info filled, including name, time and teacher, and some course.<|eot_id|><|start_header_id|>user<|end_header_id|>"
        "Question: {question} \n"
        "Context: \n"
        "{context} \n"
        "Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"),
    input_variables=["question", "context"],
)

model = OllamaLLM(
    model=MODEL,
    base_url=OLLAMA_HOST,
    stop=["<|eot_id|>"],
)

chain = (
    {"context": lambda x: "No context available", "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


async def generate(question: str):
    async for chunk in chain.astream(question):
        yield f"data: {json.dumps({'data': chunk})}\n\n"
    yield f"data: {json.dumps({'data': 'SPECIAL_END_TOKEN'})}\n\n"


@app.get("/api/ask")
async def main(question: str = "你好"):
    return StreamingResponse(
        generate(question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
