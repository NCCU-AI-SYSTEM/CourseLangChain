"""L2:agent 主迴圈的 safeguard。

包一層 LangGraph **CompiledGraph**(`create_react_agent` 的產物),確保:
- ReAct 步數不超過 `max_steps`(透過 graph 的 `recursion_limit` 換算)
- 整體 wall-clock 不超過 `timeout_sec`(背景執行緒 + future timeout)
- 任何 LLM / tool 拋出的 exception 都轉成人類可讀字串,不外漏 stack trace
- 回傳值固定為 `dict`(`{"output": str, "steps": int, "elapsed_sec": float, "error": Optional[str]}`)

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
                    "output": "處理時間過長,請簡化您的要求後重試。",
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
            name = type(e).__name__
            if "Recursion" in name:
                self._log_error(user_input, e)
                return {
                    "output": "查詢步驟過多,請把問題拆得更具體一點再試。",
                    "steps": self.max_steps,
                    "elapsed_sec": round(time.time() - start, 2),
                    "error": "recursion_limit",
                }
            self._log_error(user_input, e)
            return {
                "output": "系統暫時無法處理您的要求,請稍後再試。",
                "steps": 0,
                "elapsed_sec": round(time.time() - start, 2),
                "error": name,
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
                return text.strip() or "這次沒能組出回覆,請換個問法再試一次。"
        return str(result)

    @staticmethod
    def _count_steps(result: Any) -> int:
        """以 messages 數扣掉首則 human 當作步數的粗估(telemetry 用)。"""
        if isinstance(result, dict):
            return max(0, len(result.get("messages", [])) - 1)
        return 0

    def _log_timeout(self, user_input: str) -> None:
        logger.warning("agent timeout(%.0fs): input=%r", self.timeout_sec, user_input[:80])

    def _log_error(self, user_input: str, exc: BaseException) -> None:
        logger.exception("agent execution failed: input=%r", user_input[:80])


def run_agent(executor: SafeAgentExecutor, user_input: str) -> dict:
    """便利函數,等同 `executor.run(user_input)`。"""
    return executor.run(user_input)
