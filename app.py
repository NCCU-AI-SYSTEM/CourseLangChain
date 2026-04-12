from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
import os
from dotenv import load_dotenv

from langfuse import get_client
from langfuse.langchain import CallbackHandler

load_dotenv(override=True)

os.environ.setdefault(
    "LANGFUSE_BASE_URL", os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000")
)
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", os.getenv("LANGFUSE_PUBLIC_KEY") or "")
os.environ.setdefault("LANGFUSE_SECRET_KEY", os.getenv("LANGFUSE_SECRET_KEY") or "")

app = FastAPI()

public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")
if public_key and secret_key:
    langfuse_handler = CallbackHandler()
    print(f"Langfuse tracing enabled: {os.getenv('LANGFUSE_BASE_URL')}")
else:
    langfuse_handler = None


async def generate_streaming(question: str):
    """Generator for streaming response."""
    from main import CourseLangGraph

    agent = CourseLangGraph()

    full_response = ""
    try:
        async for chunk in agent.astream(question):
            if chunk:
                full_response += str(chunk)
                yield f"data: {json.dumps({'data': str(chunk)})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

    if langfuse_handler:
        get_client().flush()
    yield f"data: {json.dumps({'data': 'SPECIAL_END_TOKEN'})}\n\n"


async def generate_non_streaming(question: str):
    """Generator for non-streaming response (single chunk)."""
    from main import CourseLangGraph

    agent = CourseLangGraph()
    try:
        result = agent.invoke(question)
        yield f"data: {json.dumps({'data': result})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

    if langfuse_handler:
        get_client().flush()
    yield f"data: {json.dumps({'data': 'SPECIAL_END_TOKEN'})}\n\n"


@app.get("/api/ask")
async def main(question: str = "你好", stream: bool = True):
    """Chat endpoint supporting both streaming and non-streaming.

    Args:
        question: The user's question
        stream: If True, returns streaming SSE response. If False, returns single response.
    """
    if stream:
        return StreamingResponse(
            generate_streaming(question),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        return StreamingResponse(
            generate_non_streaming(question),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
