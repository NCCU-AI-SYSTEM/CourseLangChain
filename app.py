from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from main import CourseLangChain

app = FastAPI()


import asyncio
import json

async def generate(question: str):
    async for chunk in chain.chain.astream(question):
        yield f"data: {json.dumps({'data': chunk})}\n\n"
    yield f"data: {json.dumps({'data': 'SPECIAL_END_TOKEN'})}\n\n"


@app.get("/api/ask")
async def main(question:str = "你好"):
    return StreamingResponse(
        generate(question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

if __name__ == "__main__":
    chain = None
    try:
        import uvicorn
        chain = CourseLangChain()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        print("Deleting chain...")
        del chain
    

