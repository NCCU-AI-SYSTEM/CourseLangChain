import logging
import os
import uuid

import fire
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from agents.brain_agent import brain_agent
from harness import SafeAgentExecutor, sanitize_input, validate_output
from paths import check_contract
from tools.retrieve import parse_formatted_docs, retrieve_tool

load_dotenv(override=True)  # paths.py already did this; kept for direct runs

# L1 擋下不安全輸入時回給使用者的訊息
_REJECT_MSG = "抱歉,您的輸入無法處理,請改用一般的課程查詢方式重新提問。"
# L3 驗證未通過時回給使用者的訊息(寧可不回課表,也不給錯誤課表)
_INVALID_OUTPUT_MSG = "抱歉,系統產生的課表未通過正確性檢查,請調整條件後再試一次。"

os.environ.setdefault(
    "LANGFUSE_BASE_URL", os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000")
)
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", os.getenv("LANGFUSE_PUBLIC_KEY") or "")
os.environ.setdefault("LANGFUSE_SECRET_KEY", os.getenv("LANGFUSE_SECRET_KEY") or "")

logger = logging.getLogger("CourseLangGraph")
logger.setLevel(logging.DEBUG)

public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")
if public_key and secret_key:
    langfuse_handler = CallbackHandler()
    logger.info(f"Langfuse tracing enabled: {os.getenv('LANGFUSE_BASE_URL')}")
else:
    langfuse_handler = None
    logger.warning("Langfuse credentials not set - tracing disabled")

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


# on_tool_start 時送給前端的進度文字。刻意做成「工具名 → 人話」的對照表:
# 工具名不外流,前端只拿到可以直接顯示的句子,也就不必跟著後端的工具清單一起改。
# 對照不到的工具就不送——寧可不顯示進度,也不要把內部名稱漏出去。
_TOOL_STATUS = {
    "text_to_sql_tool": "正在解析時間條件…",
    retrieve_tool.name: "正在查詢課程…",   # 不寫死:工具改名時 import 會先炸,不會靜默失效
    "course_detail_tool": "正在讀課程大綱…",
    "schedule_tool": "正在排課…",
    "my_schedule_tool": "正在更新你的課表…",
    "user_profile_tool": "正在讀取修課紀錄…",
}


def _check_tool_status_keys() -> None:
    """確保 _TOOL_STATUS 的每個 key 都是實際註冊的工具名。

    工具一旦改名,進度提示只會安靜地不再出現 —— 畫面上沒有任何錯誤可循,
    而側通道的候選課程也會跟著消失。寧可在開機時大聲失敗。
    """
    from tools.registry import all_tools

    known = {getattr(t, "name", "") for t in all_tools()}
    unknown = sorted(set(_TOOL_STATUS) - known)
    if unknown:
        raise RuntimeError(
            f"_TOOL_STATUS 含有未註冊的工具名:{unknown}。"
            "工具可能被改名或移除,請同步更新 main.py 的對照表。"
        )


def _tool_output_text(event: dict) -> str:
    """取 on_tool_end 事件裡的工具回傳文字。

    langgraph 依版本可能給 ToolMessage 物件或原始 str,兩種都要能取到內容。
    """
    output = (event.get("data") or {}).get("output")
    return str(getattr(output, "content", output) or "")


class CourseLangGraph:
    def __init__(self, cli: bool = False) -> None:
        # 資料成品與 contract.yaml 必須對得上,否則寧可不啟動(memo,每個 process 一次)
        check_contract()
        _check_tool_status_keys()
        self.brain_agent = brain_agent
        # L2:把 ReAct graph 包進安全外殼(step / timeout / exception 防護)
        self.executor = SafeAgentExecutor(brain_agent)
        logger.info("Brain Agent (ReAct) ready,三層 harness 已啟用。")

    def _base_config(self, thread_id: str | None = None) -> dict:
        """組 LangGraph config。

        thread_id 是對話記憶的鍵:同一個 thread_id 的多輪才會共用歷史。
        沒帶時發一個一次性 id——agent 掛了 checkpointer,少了 thread_id 會直接 raise,
        用一次性 id 等同「這輪無記憶」,比讓呼叫端炸掉好。
        """
        config: dict = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
        config["configurable"] = {"thread_id": thread_id or f"ephemeral-{uuid.uuid4()}"}
        return config

    def invoke(self, user_input: str, thread_id: str | None = None) -> str:
        # L1:輸入清理
        text, is_safe = sanitize_input(user_input)
        if not is_safe:
            logger.warning("input rejected by L1 sanitizer")
            return _REJECT_MSG

        # L2:安全外殼執行(永不 raise)
        result = self.executor.run(text, config=self._base_config(thread_id))
        output = result["output"]
        logger.info(
            "agent done: steps=%s elapsed=%ss error=%s",
            result["steps"], result["elapsed_sec"], result["error"],
        )
        if result["error"]:
            return output  # 已是友善訊息

        # L3:輸出驗證(衝堂等)
        ok, msg = validate_output(output)
        if not ok:
            logger.warning("output rejected by L3: %s", msg)
            return _INVALID_OUTPUT_MSG
        return output

    async def astream(self, user_input: str, thread_id: str | None = None):
        """逐段產出回覆。

        yield 兩種型別,呼叫端(app.py)要分開處理:
        - `str`:LLM 吐出的文字 token
        - `dict`:側通道事件,目前有兩種
          - `{"type": "status", "text": ...}`:工具開始執行的進度提示(不含工具名)
          - `{"type": "courses", "courses": [...]}`:候選課程,course_id 直接取自
            工具輸出、不經 LLM 轉述
        """
        # L1:輸入清理(串流路徑同樣先擋)
        text, is_safe = sanitize_input(user_input)
        if not is_safe:
            logger.warning("input rejected by L1 sanitizer (stream)")
            yield _REJECT_MSG
            return

        # L2 的 step 上限沿用 executor 的 recursion_limit;wall-clock timeout 在串流下
        # 不套用(會切斷已輸出的 token)。L3 輸出驗證亦因逐 token 串流無法即時套用,
        # 改在非串流 invoke 路徑把關。
        config = {
            **self._base_config(thread_id),
            "recursion_limit": self.executor.recursion_limit,
        }
        seen_ids: set[str] = set()
        async for event in self.brain_agent.astream_events(
            {"messages": [{"role": "user", "content": text}]},
            config=config,
            version="v2",
        ):
            kind = event.get("event")
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                content = getattr(chunk, "content", None)
                if content:
                    yield content
            elif kind == "on_tool_start":
                # 進度提示。ReAct 在 CPU-only 下一題要數分鐘,中間必須讓使用者
                # 看得出還在跑;但顯示的是人話,不是工具名。
                status = _TOOL_STATUS.get(event.get("name") or "")
                if status:
                    yield {"type": "status", "text": status}
            elif kind == "on_tool_end" and event.get("name") == retrieve_tool.name:
                # 側通道:候選課程直接取自工具輸出,不經 LLM 轉述,前端據此畫
                # 「加入課表」按鈕——徹底避開模型抄錯 13 碼 course_id 的風險。
                # 排課時同一輪可能查兩次,用 seen_ids 去重。
                fresh = [
                    c
                    for c in parse_formatted_docs(_tool_output_text(event))
                    if c["course_id"] not in seen_ids
                ]
                if fresh:
                    seen_ids.update(c["course_id"] for c in fresh)
                    yield {"type": "courses", "courses": fresh}
        logger.info("Brain Agent execution completed")


async def main():
    agent = CourseLangGraph(cli=True)
    # 整個 REPL 共用一個 thread_id,多輪追問才接得起來(輸入 new 可開新對話)
    thread_id = f"cli-{uuid.uuid4()}"
    while True:
        query = input("User: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        if query.lower() == "new":
            thread_id = f"cli-{uuid.uuid4()}"
            print("(已開始新對話,先前的上下文不再沿用)\n")
            continue
        print("Bot:")
        result = agent.invoke(query, thread_id=thread_id)
        print(result)
        print()
    if langfuse_handler:
        get_client().flush()


if __name__ == "__main__":
    fire.Fire(main)
