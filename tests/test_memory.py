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
    test_brain_agent_has_checkpointer()
    test_memory_across_turns()
    test_ephemeral_thread_id_when_missing()
    test_step_count_excludes_history()
    print(f"\n結果:{_PASS} passed, {_FAIL} failed")
    raise SystemExit(1 if _FAIL else 0)
