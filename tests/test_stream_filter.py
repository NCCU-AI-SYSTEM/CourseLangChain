"""串流只送主迴圈的最終答案:工具內部的 LLM 輸出不可外洩到畫面。

跑:`.venv/bin/python -m tests.test_stream_filter`(在 CourseLangChain/ 下)

**不需要 LLM 與資料庫**:用假的 chat model 建 ReAct graph,塞進 CourseLangGraph.astream。

回歸對象(2026-11-15):問「星期三下午有什麼 AI 相關的課?」時,text_to_sql_tool 內部
解析時間條件的那次 LLM 呼叫回 `{"include_times": [...]}`,這則訊息沒有 tool_calls,
被 astream 當成最終答案送到前端,畫面最前面出現一段原始 JSON。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

_PASS = 0
_FAIL = 0

_LEAK = '{"include_times": [{"weekday": 3, "start_hour": 13, "end_hour": 18}]}'
_FINAL = "星期三下午的 AI 相關課程如下"


def check(cond: bool, msg: str) -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  ✓ {msg}")
    else:
        _FAIL += 1
        print(f"  ✗ FAIL: {msg}")


class _FakeToolModel(FakeMessagesListChatModel):
    """create_react_agent 會呼叫 bind_tools;假模型照原樣回傳自己即可。"""

    def bind_tools(self, tools, **kwargs):
        return self


def _build_graph():
    inner = _FakeToolModel(responses=[AIMessage(content=_LEAK)])

    @tool
    def text_to_sql_tool(question: str) -> str:
        """解析時間條件"""
        # 與 tools/text_to_sql.py 相同:工具內部直接 invoke,不另傳 config
        return inner.invoke(question).content

    outer = _FakeToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "text_to_sql_tool", "args": {"question": "星期三下午"}, "id": "c1"}],
            ),
            AIMessage(content=_FINAL),
        ]
    )
    return create_react_agent(model=outer, tools=[text_to_sql_tool], checkpointer=InMemorySaver())


async def _collect(agen) -> list:
    return [item async for item in agen]


def test_inner_llm_event_surfaces() -> None:
    """對照組:確認工具內的 LLM 輸出確實會冒成外層事件,否則下一個測試是空轉。"""
    print("[對照:工具內的 LLM 呼叫會出現在 astream_events]")
    graph = _build_graph()
    events = asyncio.run(
        _collect(
            graph.astream_events(
                {"messages": [{"role": "user", "content": "hi"}]},
                config={"configurable": {"thread_id": "probe"}},
                version="v2",
            )
        )
    )
    inner = [
        e for e in events
        if e.get("event") == "on_chat_model_end" and _LEAK in str(e["data"]["output"].content)
    ]
    check(bool(inner), "工具內的 LLM 輸出以 on_chat_model_end 冒上來")
    check(
        all((e.get("metadata") or {}).get("langgraph_node") == "tools" for e in inner),
        "它的 langgraph_node 是 'tools'(過濾依據)",
    )


def test_astream_hides_inner_llm_output() -> None:
    print("[astream 只送主迴圈的最終答案]")
    from main import CourseLangGraph

    agent = object.__new__(CourseLangGraph)  # 跳過 __init__ 的 contract 檢查(要資料庫)
    agent.brain_agent = _build_graph()
    agent.executor = SimpleNamespace(recursion_limit=25)

    items = asyncio.run(_collect(agent.astream("星期三下午有什麼 AI 相關的課?", thread_id="t")))
    texts = [i for i in items if isinstance(i, str)]
    check(not any("include_times" in t for t in texts), "時間條件 JSON 沒有送到前端")
    check(texts == [_FINAL], f"文字只有最終答案一則(得到 {texts!r})")
    check(
        any(isinstance(i, dict) and i.get("type") == "status" for i in items),
        "進度提示(側通道)照常送出",
    )


if __name__ == "__main__":
    test_inner_llm_event_surfaces()
    test_astream_hides_inner_llm_output()
    print(f"\n結果:{_PASS} passed, {_FAIL} failed")
    raise SystemExit(1 if _FAIL else 0)
