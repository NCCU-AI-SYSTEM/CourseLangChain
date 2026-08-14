from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import json
import os
from dotenv import load_dotenv

from tools import session_profile, session_schedule

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


_agent = None


def get_agent():
    """單例。CourseLangGraph 內含掛了 checkpointer 的 brain_agent,
    每個 request 重建雖然不會弄丟記憶(brain_agent 是模組級的),但沒必要重複建。"""
    global _agent
    if _agent is None:
        from main import CourseLangGraph

        _agent = CourseLangGraph()
    return _agent


async def generate_streaming(question: str, session_id: str | None = None):
    """Generator for streaming response."""
    agent = get_agent()

    full_response = ""
    try:
        async for chunk in agent.astream(question, thread_id=session_id):
            if not chunk:
                continue
            # dict = 側通道事件(目前只有候選課程清單):原樣送出,不併進文字回覆
            if isinstance(chunk, dict):
                yield f"data: {json.dumps(chunk)}\n\n"
                continue
            full_response += str(chunk)
            yield f"data: {json.dumps({'data': str(chunk)})}\n\n"
    except Exception as e:
        # 用 data 欄位回可讀訊息(前端只認 data);否則畫面會顯示 undefined
        msg = f"抱歉,系統發生錯誤,暫時無法處理您的要求。({type(e).__name__})"
        yield f"data: {json.dumps({'data': msg, 'error': str(e)})}\n\n"

    if langfuse_handler:
        get_client().flush()
    yield f"data: {json.dumps({'data': 'SPECIAL_END_TOKEN'})}\n\n"


async def generate_non_streaming(question: str, session_id: str | None = None):
    """Generator for non-streaming response (single chunk)."""
    agent = get_agent()
    try:
        result = agent.invoke(question, thread_id=session_id)
        yield f"data: {json.dumps({'data': result})}\n\n"
    except Exception as e:
        msg = f"抱歉,系統發生錯誤,暫時無法處理您的要求。({type(e).__name__})"
        yield f"data: {json.dumps({'data': msg, 'error': str(e)})}\n\n"

    if langfuse_handler:
        get_client().flush()
    yield f"data: {json.dumps({'data': 'SPECIAL_END_TOKEN'})}\n\n"


@app.get("/api/ask")
async def main(question: str = "你好", stream: bool = True, session_id: str | None = None):
    """Chat endpoint supporting both streaming and non-streaming.

    Args:
        question: The user's question
        stream: If True, returns streaming SSE response. If False, returns single response.
        session_id: 對話識別碼。同一個 session_id 的多輪會共用上下文
            (追問「那第二門課的課綱?」要靠它)。不帶則視為一次性提問、無記憶。
    """
    gen = (
        generate_streaming(question, session_id)
        if stream
        else generate_non_streaming(question, session_id)
    )
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# 課表面板 API —— 與 my_schedule_tool 共用同一份 session 狀態
# (面板手動改的、agent 改的,都是 tools/session_schedule 那份;衝堂規則也同一套)
# ---------------------------------------------------------------------------


def _schedule_payload(session_id: str) -> dict:
    courses = session_schedule.get_schedule(session_id)
    return {
        "session_id": session_id,
        "courses": session_schedule.to_dicts(courses),
        "total_credits": session_schedule.total_credits(courses),
    }


def _require_session(session_id: str | None) -> str:
    if not session_id:
        raise HTTPException(status_code=400, detail="缺少 session_id")
    return session_id


@app.get("/api/terms")
async def get_terms():
    """課表面板的學期選單:data.db 實際有資料的學期 + 目前系統鎖定的學期。

    面板讓使用者先選學期、再貼課程代碼,9 碼(全校課程查詢系統顯示的科目代號)才有辦法
    補成課表要用的 13 碼;順帶讓「這些是哪個學期的課」在畫面上看得見。
    """
    return {
        "terms": session_schedule.available_terms(),
        "current": _current_term(),
    }


@app.get("/api/schedule")
async def get_schedule(session_id: str | None = None):
    """取得這個 session 目前已排定的課表。"""
    return _schedule_payload(_require_session(session_id))


@app.post("/api/schedule")
async def add_to_schedule(payload: dict = Body(...)):
    """加一門課。衝堂、或加入同一門課的另一個班時,自動移除舊的那幾門
    (回傳 removed 與 removed_reasons 讓前端照實提示使用者)。"""
    session_id = _require_session(payload.get("session_id"))
    # 面板可能只拿到 9 碼(使用者從全校課程查詢系統複製的科目代號),
    # 用選單選的學期補成 13 碼;13 碼原樣通過。
    course_id = session_schedule.normalize_course_id(
        payload.get("course_id"), payload.get("term")
    )
    if not course_id:
        raise HTTPException(status_code=400, detail="缺少 course_id")

    result = session_schedule.add_course(session_id, course_id)
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return {
        **_schedule_payload(session_id),
        "added": session_schedule.to_dicts([result["added"]])[0],
        "removed": session_schedule.to_dicts(result["removed"]),
        # {course_id: "same_name"|"conflict"} —— 前端據此決定提示措辭
        "removed_reasons": result.get("removed_reasons", {}),
        "already": result["already"],
    }


@app.delete("/api/schedule")
async def delete_from_schedule(session_id: str | None = None, course_id: str | None = None):
    """移除一門課;不給 course_id 則清空整份課表。"""
    session_id = _require_session(session_id)
    if course_id:
        removed = session_schedule.remove_course(session_id, course_id)
    else:
        session_schedule.clear_schedule(session_id)
        removed = True
    return {**_schedule_payload(session_id), "removed": removed}


# ---------------------------------------------------------------------------
# 成績單 API —— 使用者自校務系統下載後上傳,用於個人化排課
#
# 隱私:上傳的原始 JSON **只在解析當下存在於記憶體**,解析完即丟棄——不寫檔、
# 不進資料庫、不寫暫存。只有去識別化結果(系級/已修課號/畢業缺口)留在 session
# 記憶體,程序重啟即清空,使用者也可隨時 DELETE 移除。
# ---------------------------------------------------------------------------

# 成績單 JSON 的大小上限(實際檔案約數十 KB,留寬裕餘量並擋住濫用)
MAX_RECORD_BYTES = 2 * 1024 * 1024


def _current_term() -> str:
    from paths import COURSE_SEMESTER, COURSE_YEAR

    return f"{COURSE_YEAR}{COURSE_SEMESTER}"


@app.get("/api/profile")
async def get_profile(session_id: str | None = None):
    """查詢這個 session 是否已上傳成績單(回摘要,不含任何個資)。"""
    session_id = _require_session(session_id)
    return session_profile.summary(session_profile.get_profile(session_id))


@app.post("/api/profile")
async def upload_profile(payload: dict = Body(...)):
    """上傳成績單 JSON。原始內容不落地,只保留去識別化的修課狀況。"""
    session_id = _require_session(payload.get("session_id"))
    record = payload.get("record")
    if record is None:
        raise HTTPException(status_code=400, detail="缺少 record(成績單 JSON 內容)")

    # 粗略擋住過大的輸入,避免記憶體被灌爆
    if len(json.dumps(record).encode("utf-8")) > MAX_RECORD_BYTES:
        raise HTTPException(status_code=413, detail="成績單檔案過大")

    try:
        profile = session_profile.set_profile(session_id, record, current_term=_current_term())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=400, detail="成績單解析失敗,請確認檔案格式")

    return session_profile.summary(profile)


@app.delete("/api/profile")
async def delete_profile(session_id: str | None = None):
    """移除這個 session 的修課資料。"""
    session_id = _require_session(session_id)
    existed = session_profile.clear_profile(session_id)
    return {"has_profile": False, "removed": existed}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
