"""對話記憶(checkpointer + thread_id + 歷史裁剪)測試。

跑:`.venv/bin/python -m tests.test_memory`(在 CourseLangChain/ 下)

**不需要 LLM**:用一個假的 chat model 建 ReAct graph,驗證 checkpointer 的行為;
真正的 brain_agent 只檢查「有沒有掛上 checkpointer」與裁剪函數本身。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

_PASS = 0
_FAIL = 0


def check(cond: bool, msg: str) -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  ✓ {msg}")
    else:
        _FAIL += 1
        print(f"  ✗ FAIL: {msg}")


def test_trim_keeps_recent() -> None:
    print("[歷史裁剪:保留最近幾則]")
    from agents.brain_agent import MAX_HISTORY_MESSAGES, _trim_history

    # 造遠超過上限的 human/AI 交替訊息(從常數推導,調整上限時測試不會假失敗)
    rounds = MAX_HISTORY_MESSAGES  # 每輪 2 則 → 總數是上限的兩倍
    msgs: list = []
    for i in range(rounds):
        msgs.append(HumanMessage(content=f"問題{i}"))
        msgs.append(AIMessage(content=f"回答{i}"))

    out = _trim_history({"messages": msgs})
    check("llm_input_messages" in out, "回傳 llm_input_messages(不覆寫 state)")
    trimmed = out["llm_input_messages"]
    check(len(trimmed) <= MAX_HISTORY_MESSAGES, f"裁到不超過 {MAX_HISTORY_MESSAGES} 則")
    check(trimmed[-1].content == f"回答{rounds - 1}", "保留的是最新的訊息")
    check(all(m.content != "問題0" for m in trimmed), "最舊的訊息被裁掉")


def test_trim_short_history_untouched() -> None:
    print("[歷史裁剪:短對話原樣通過]")
    from agents.brain_agent import _trim_history

    msgs = [HumanMessage(content="你好"), AIMessage(content="哈囉")]
    trimmed = _trim_history({"messages": msgs})["llm_input_messages"]
    check(len(trimmed) == 2, "沒超過上限時不裁")


def test_trim_no_orphan_tool_message() -> None:
    print("[歷史裁剪:不切出孤兒 ToolMessage]")
    from agents.brain_agent import _trim_history

    # 造一段會誘發「剛好從 ToolMessage 切開」的歷史:
    # 每回合 = human → AI(tool_calls) → tool → AI(答案)
    msgs: list = []
    for i in range(10):
        msgs.append(HumanMessage(content=f"查課{i}"))
        msgs.append(AIMessage(
            content="",
            tool_calls=[{"name": "query_courses_tool", "args": {"keyword": "x"}, "id": f"call{i}"}],
        ))
        msgs.append(ToolMessage(content=f"結果{i}", tool_call_id=f"call{i}"))
        msgs.append(AIMessage(content=f"答案{i}"))

    trimmed = _trim_history({"messages": msgs})["llm_input_messages"]
    check(len(trimmed) > 0, "裁剪後非空")
    check(
        getattr(trimmed[0], "type", None) == "human",
        "視窗開頭是 human(不會是孤兒 ToolMessage → 避免 provider 400)",
    )
    # 每個留下的 ToolMessage,其 tool_call_id 都要能在視窗內找到對應的 tool_calls
    call_ids = {
        tc["id"]
        for m in trimmed
        for tc in (getattr(m, "tool_calls", None) or [])
    }
    orphans = [
        m for m in trimmed
        if isinstance(m, ToolMessage) and m.tool_call_id not in call_ids
    ]
    check(not orphans, "視窗內沒有對不到 tool_call 的 ToolMessage")


def test_patch_interrupted_tool_calls() -> None:
    print("[中斷留下的工具呼叫缺口:送進模型前補上]")
    from langgraph.prebuilt.chat_agent_executor import _validate_chat_history

    from agents.brain_agent import INTERRUPTED_TOOL_RESULT, _patch_interrupted_tool_calls

    broken = [
        HumanMessage(content="推薦星期三以外的課"),
        AIMessage(content="", tool_calls=[
            {"name": "retrieve_tool", "args": {"keyword": "AI"}, "id": "c1"},
            {"name": "text_to_sql_tool", "args": {}, "id": "c2"},
        ]),
        ToolMessage(content="結果", tool_call_id="c2"),  # 平行呼叫只回來一個
        HumanMessage(content="你好"),
    ]
    patched = _patch_interrupted_tool_calls(broken)
    try:
        _validate_chat_history(patched)
        valid = True
    except ValueError:
        valid = False
    check(valid, "補過的歷史通過 LangGraph 自己的檢查(INVALID_CHAT_HISTORY 不再發生)")
    added = [m for m in patched if isinstance(m, ToolMessage) and m.content == INTERRUPTED_TOOL_RESULT]
    check(len(added) == 1 and added[0].tool_call_id == "c1", "只補缺的那一個,已有結果的不重複補")
    check(patched.index(added[0]) == 2, "補在那則 AIMessage 的正後方")

    normal = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "c1"}]),
        ToolMessage(content="r", tool_call_id="c1"),
        AIMessage(content="答"),
    ]
    check(_patch_interrupted_tool_calls(normal) is normal, "完整的歷史原樣通過,不多補")


def test_cancel_mid_tool_then_ask_again() -> None:
    """重現 2026-11-02 docker 上的事故:工具跑到一半被取消,同一個 session 之後每題都失敗。"""
    print("[工具執行中被取消後,同一個 session 還能繼續問]")
    import asyncio

    from langchain_core.language_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.prebuilt import create_react_agent

    from agents.brain_agent import _trim_history

    class Scripted(BaseChatModel):
        """第一次呼叫工具,之後都直接回答。"""

        calls: int = 0

        @property
        def _llm_type(self) -> str:
            return "scripted"

        def bind_tools(self, *_a, **_k):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                msg = AIMessage(content="", tool_calls=[{"name": "slow_tool", "args": {"q": "x"}, "id": "c1"}])
            else:
                msg = AIMessage(content="你好!")
            return ChatResult(generations=[ChatGeneration(message=msg)])

    @tool
    async def slow_tool(q: str) -> str:
        """很慢的工具(模擬第一次查詢在載入 embedding 模型)。"""
        await asyncio.sleep(30)
        return "結果"

    def run(hook) -> str:
        graph = create_react_agent(
            model=Scripted(), tools=[slow_tool], pre_model_hook=hook, checkpointer=InMemorySaver()
        )
        cfg = {"configurable": {"thread_id": "s1"}}

        async def scenario() -> str:
            async def first_turn():
                async for ev in graph.astream_events(
                    {"messages": [HumanMessage(content="查課")]}, config=cfg, version="v2"
                ):
                    if ev["event"] == "on_tool_start":
                        tool_started.set()

            tool_started = asyncio.Event()
            task = asyncio.create_task(first_turn())
            await tool_started.wait()
            task.cancel()  # 使用者重新整理頁面:app.py 的 pump.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            try:
                out = await graph.ainvoke({"messages": [HumanMessage(content="你好")]}, config=cfg)
                return out["messages"][-1].content
            except ValueError as e:
                return f"ValueError: {str(e)[:40]}"

        return asyncio.run(scenario())

    # 對照組:沒有修補時真的會壞(證明這個測試重現得出事故)
    control = run(lambda state: {"llm_input_messages": state["messages"]})
    check(control.startswith("ValueError"), f"對照組(不修補)重現 INVALID_CHAT_HISTORY(實際:{control!r})")
    out = run(_trim_history)
    check(out == "你好!", f"有修補時,下一題正常回答(實際:{out!r})")


def test_brain_agent_has_checkpointer() -> None:
    print("[brain_agent 已掛 checkpointer]")
    from agents.brain_agent import brain_agent

    check(brain_agent.checkpointer is not None, "brain_agent.checkpointer 存在")


def test_memory_across_turns() -> None:
    """用假 model 驗證:同 thread_id 看得到歷史,不同 thread_id 看不到。"""
    print("[記憶:同 thread 接得上 / 跨 thread 隔離]")
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.prebuilt import create_react_agent

    seen: list[list] = []

    class RecordingModel(FakeMessagesListChatModel):
        """記下每次被呼叫時收到的完整訊息串,用來斷言歷史有沒有帶進來。"""

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen.append(list(messages))
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    model = RecordingModel(responses=[AIMessage(content="好的")] * 10)
    graph = create_react_agent(
        model=model,
        tools=[],
        prompt=SystemMessage(content="sys"),
        checkpointer=InMemorySaver(),
    )

    cfg_a = {"configurable": {"thread_id": "A"}}
    graph.invoke({"messages": [HumanMessage(content="我想修資料庫系統")]}, config=cfg_a)
    graph.invoke({"messages": [HumanMessage(content="它的課綱呢")]}, config=cfg_a)

    second_turn = seen[-1]
    contents = " ".join(str(getattr(m, "content", "")) for m in second_turn)
    check("我想修資料庫系統" in contents, "同 thread_id 的第二輪看得到第一輪的提問")
    check("它的課綱呢" in contents, "同 thread_id 的第二輪含本輪提問")

    # 換一個 thread_id,不該看到 A 的歷史
    graph.invoke({"messages": [HumanMessage(content="隨便問問")]},
                 config={"configurable": {"thread_id": "B"}})
    contents_b = " ".join(str(getattr(m, "content", "")) for m in seen[-1])
    check("我想修資料庫系統" not in contents_b, "不同 thread_id 之間互相隔離")


def test_ephemeral_thread_id_when_missing() -> None:
    print("[未帶 thread_id → 一次性 id,不 raise]")
    from main import CourseLangGraph

    cfg1 = CourseLangGraph._base_config(object.__new__(CourseLangGraph), None)
    cfg2 = CourseLangGraph._base_config(object.__new__(CourseLangGraph), None)
    tid1 = cfg1["configurable"]["thread_id"]
    tid2 = cfg2["configurable"]["thread_id"]
    check(bool(tid1), "沒帶 thread_id 時仍產生一個 id")
    check(tid1 != tid2, "每次產生的一次性 id 不同(等同無記憶)")

    cfg3 = CourseLangGraph._base_config(object.__new__(CourseLangGraph), "sess-123")
    check(cfg3["configurable"]["thread_id"] == "sess-123", "有帶 thread_id 時照用")


def test_step_count_excludes_history() -> None:
    print("[steps 統計不把歷史算進來]")
    from harness.agent_wrapper import SafeAgentExecutor

    # 兩輪對話:第一輪 4 則,第二輪 human + AI(tool) + tool + AI = 再 4 則
    messages = [
        HumanMessage(content="第一輪"),
        AIMessage(content=""),
        ToolMessage(content="r", tool_call_id="c1"),
        AIMessage(content="答1"),
        HumanMessage(content="第二輪"),
        AIMessage(content=""),
        ToolMessage(content="r", tool_call_id="c2"),
        AIMessage(content="答2"),
    ]
    steps = SafeAgentExecutor._count_steps({"messages": messages})
    check(steps == 3, f"只算最後一則 human 之後的訊息(得到 {steps},期望 3)")

    single = SafeAgentExecutor._count_steps(
        {"messages": [HumanMessage(content="只有一輪"), AIMessage(content="答")]}
    )
    check(single == 1, "單輪對話仍正確")


if __name__ == "__main__":
    test_trim_keeps_recent()
    test_trim_short_history_untouched()
    test_trim_no_orphan_tool_message()
    test_patch_interrupted_tool_calls()
    test_cancel_mid_tool_then_ask_again()
    test_brain_agent_has_checkpointer()
    test_memory_across_turns()
    test_ephemeral_thread_id_when_missing()
    test_step_count_excludes_history()
    print(f"\n結果:{_PASS} passed, {_FAIL} failed")
    raise SystemExit(1 if _FAIL else 0)
