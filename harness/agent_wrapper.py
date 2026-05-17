"""L2:agent 主迴圈的 safeguard。

包一層 LangChain `AgentExecutor`,確保:
- ReAct 步數不超過 `max_steps`
- 整體 wall-clock 不超過 `timeout_sec`
- 任何 LLM / tool 拋出的 exception 都轉成人類可讀字串,不外漏 stack trace
- 回傳值固定為 `dict`(`{"output": str, "steps": int, "error": Optional[str]}`),
  方便 FastAPI / CLI 統一處理

協作者通常不會直接動這個檔案。如果你需要新增監控 / log,在 `_log_*` 函數裡加。
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 5
DEFAULT_TIMEOUT_SEC = 30.0


class SafeAgentExecutor:
    """ReAct agent 的安全外殼。

    Args:
        executor: LangChain `AgentExecutor` 實例(由 `agents.brain_agent.build_brain_agent` 建立)
        max_steps: ReAct 最多迭代幾步,防止無限呼叫工具
        timeout_sec: 總 wall-clock 上限
    """

    def __init__(
        self,
        executor: Any,
        max_steps: int = DEFAULT_MAX_STEPS,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self.executor = executor
        self.max_steps = max_steps
        self.timeout_sec = timeout_sec
        executor.max_iterations = max_steps
        executor.max_execution_time = timeout_sec

    def run(self, user_input: str, config: dict | None = None, **input_extras: Any) -> dict:
        """同步呼叫 agent。回傳 dict,**永不 raise**。

        Args:
            user_input: 使用者問題,塞進 `input` 欄位
            config: 透傳給 LangChain Runnable 的 config(callbacks / metadata 等)
            input_extras: 其他要塞進 input dict 的欄位(例如 chat_history)
        """
        start = time.time()
        try:
            invoke_args: list[Any] = [{"input": user_input, **input_extras}]
            if config is not None:
                invoke_args.append(config)
            result = self.executor.invoke(*invoke_args)
            output = result.get("output") if isinstance(result, dict) else str(result)
            steps = len(result.get("intermediate_steps", [])) if isinstance(result, dict) else 0
            return {
                "output": output or "",
                "steps": steps,
                "elapsed_sec": round(time.time() - start, 2),
                "error": None,
            }
        except TimeoutError as e:
            self._log_timeout(user_input, e)
            return {
                "output": "處理時間過長,請簡化您的要求後重試。",
                "steps": 0,
                "elapsed_sec": round(time.time() - start, 2),
                "error": "timeout",
            }
        except Exception as e:
            self._log_error(user_input, e)
            return {
                "output": "系統暫時無法處理您的要求,請稍後再試。",
                "steps": 0,
                "elapsed_sec": round(time.time() - start, 2),
                "error": type(e).__name__,
            }

    def _log_timeout(self, user_input: str, exc: BaseException) -> None:
        logger.warning("agent timeout: input=%r err=%s", user_input[:80], exc)

    def _log_error(self, user_input: str, exc: BaseException) -> None:
        logger.exception("agent execution failed: input=%r", user_input[:80])


def run_agent(executor: SafeAgentExecutor, user_input: str) -> dict:
    """便利函數,等同 `executor.run(user_input)`。"""
    return executor.run(user_input)
