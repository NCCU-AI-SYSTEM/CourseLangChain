from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()


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
