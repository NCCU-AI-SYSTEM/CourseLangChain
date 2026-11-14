"""L2:agent 主迴圈的 safeguard。

包一層 LangGraph **CompiledGraph**(`create_react_agent` 的產物),確保:
- ReAct 步數不超過 `max_steps`(透過 graph 的 `recursion_limit` 換算)
- 整體 wall-clock 不超過 `timeout_sec`(背景執行緒 + future timeout)
- 任何 LLM / tool 拋出的 exception 都轉成人類可讀字串,不外漏 stack trace
- 回傳值固定為 `dict`(`{"output": str, "steps": int, "elapsed_sec": float, "error": Optional[str]}`)

串流路徑(`main.py::CourseLangGraph.astream`)不經過 `run()`,但用同一組訊息與
`describe_failure()`,保證和這裡一致:逾時、步數超限、例外都轉成友善訊息。

協作者通常不會直接動這個檔案。如果你需要新增監控 / log,在 `_log_*` 函數裡加。

注意:LangGraph 的一個 ReAct 回合 ≈ 2 個 graph superstep(agent 節點 + tool 節點),
所以 `recursion_limit = 2 * max_steps + 1`。
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "8"))
# 本機模型的速度差距很大 —— 實測 9B 的 thinking 模型在 Apple Silicon 上,多工具的問句
# 要 ~400s;首次查詢還要再加上載入 embedding 模型的時間。預設取 600s 是因為超時訊息
# 看起來像系統壞了、不像設定太小,寧可等。雲端模型快很多,可以用環境變數調小。
DEFAULT_TIMEOUT_SEC = float(os.getenv("AGENT_TIMEOUT_SEC", "600"))

# 給使用者看的失敗訊息。invoke 與 astream 兩條路徑共用,使用者不論走哪條看到的都一樣。
TIMEOUT_MESSAGE = "處理時間過長,請簡化您的要求後重試。"
RECURSION_MESSAGE = "查詢步驟過多,請把問題拆得更具體一點再試。"
ERROR_MESSAGE = "系統暫時無法處理您的要求,請稍後再試。"
EMPTY_MESSAGE = "這次沒能組出回覆,請換個問法再試一次。"


def describe_failure(exc: BaseException) -> tuple[str, str]:
    """把 agent 執行中的例外轉成 `(給使用者的訊息, 錯誤代碼)`。

    錯誤代碼給 log / telemetry 用;訊息不含例外內容 —— 例外字串可能帶著
    內部網址、SQL、檔案路徑,不該出現在使用者畫面上。
    """
    # GraphRecursionError(步數超限)給「把問題拆小」的建議,而不是一般錯誤
    if "Recursion" in type(exc).__name__:
        return RECURSION_MESSAGE, "recursion_limit"
    if isinstance(exc, TimeoutError):
        return TIMEOUT_MESSAGE, "timeout"
    return ERROR_MESSAGE, type(exc).__name__


class SafeAgentExecutor:
    """ReAct agent(LangGraph CompiledGraph)的安全外殼。

    Args:
        graph: `create_react_agent(...)` 回傳的 CompiledGraph
        max_steps: ReAct 最多迭代幾回合,防止無限呼叫工具
        timeout_sec: 總 wall-clock 上限
    """

    def __init__(
        self,
        graph: Any,
        max_steps: int = DEFAULT_MAX_STEPS,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self.graph = graph
        self.max_steps = max_steps
        self.timeout_sec = timeout_sec
        self.recursion_limit = 2 * max_steps + 1

    def _build_config(self, config: dict | None) -> dict:
        merged = dict(config or {})
        # 呼叫端沒指定才補,別蓋掉外部傳進來的 recursion_limit
        merged.setdefault("recursion_limit", self.recursion_limit)
        return merged

    def run(self, user_input: str, config: dict | None = None) -> dict:
        """同步呼叫 agent。回傳 dict,**永不 raise**。

        Args:
            user_input: 使用者問題
            config: 透傳給 LangGraph 的 config(callbacks / metadata / recursion_limit 等)
        """
        start = time.time()
        cfg = self._build_config(config)
        payload = {"messages": [{"role": "user", "content": user_input}]}

        box: dict[str, Any] = {}

        def _invoke() -> None:
            try:
                box["result"] = self.graph.invoke(payload, config=cfg)
            except BaseException as exc:  # noqa: BLE001 — 帶回主執行緒處理
                box["exc"] = exc

        # daemon thread:逾時可提早返回,且不阻擋程序結束(背景殘留任務自行結束)
        worker = threading.Thread(target=_invoke, daemon=True)
        worker.start()
        worker.join(self.timeout_sec)

        try:
            if worker.is_alive():
                self._log_timeout(user_input)
                return {
                    "output": TIMEOUT_MESSAGE,
                    "steps": 0,
                    "elapsed_sec": round(time.time() - start, 2),
                    "error": "timeout",
                }
            if "exc" in box:
                raise box["exc"]
            result = box["result"]
            return {
                "output": self._extract_output(result),
                "steps": self._count_steps(result),
                "elapsed_sec": round(time.time() - start, 2),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            # GraphRecursionError(步數超限)也走這裡,給友善訊息而非 stack trace
            self._log_error(user_input, e)
            message, code = describe_failure(e)
            return {
                "output": message,
                "steps": self.max_steps if code == "recursion_limit" else 0,
                "elapsed_sec": round(time.time() - start, 2),
                "error": code,
            }

    @staticmethod
    def _extract_output(result: Any) -> str:
        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                last = messages[-1]
                content = getattr(last, "content", None)
                if content is None:          # 不是訊息物件,只能整個印出來
                    return str(last)
                # content 是空字串代表「模型這一輪什麼都沒說」,是合法狀態而不是
                # 缺欄位。這裡若退回 str(last),使用者會拿到整個 AIMessage repr
                # ——連 token 數與 run id 都在裡面。小模型偶爾會這樣收尾。
                text = content if isinstance(content, str) else str(content)
                return text.strip() or EMPTY_MESSAGE
        return str(result)

    @staticmethod
    def _count_steps(result: Any) -> int:
        """本輪的步數:最後一則 human 之後的訊息數(telemetry 用)。

        不能用 `len(messages) - 1` —— 開了對話記憶之後 graph 回傳的 messages 帶著
        整段歷史,那個數字會隨輪數一路長大,第五輪會報成二十幾步。這裡只數最後一則
        human 之後的訊息,才是這一輪真正走了幾步。
        """
        if not isinstance(result, dict):
            return 0
        messages = result.get("messages", [])
        last_human = -1
        for i, m in enumerate(messages):
            if getattr(m, "type", None) == "human":
                last_human = i
        return max(0, len(messages) - last_human - 1)

    def _log_timeout(self, user_input: str) -> None:
        logger.warning("agent timeout(%.0fs): input=%r", self.timeout_sec, user_input[:80])

    def _log_error(self, user_input: str, exc: BaseException) -> None:
        logger.exception("agent execution failed: input=%r", user_input[:80])


def run_agent(executor: SafeAgentExecutor, user_input: str) -> dict:
    """便利函數,等同 `executor.run(user_input)`。"""
    return executor.run(user_input)
