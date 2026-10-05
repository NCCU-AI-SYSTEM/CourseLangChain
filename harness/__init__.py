"""Harness layer:把正確性 / 安全性 / 穩定性的保證從 ReAct agent 主流程剝離出來。

三層職責(輸入 → 模型 → 輸出):
- L1 input_sanitizer  : 過濾 prompt injection、長度上限
- L2 agent_wrapper    : SafeAgentExecutor — 包 LangChain AgentExecutor,加 timeout / step 上限 / exception 捕捉
- L3 output_validator : 出口檢查(衝堂、學分、必修數),由各 tool 註冊自己的檢查器

新增 tool 時通常**不需要動 harness**;但若該 tool 產出需要驗證(例如課表),可在
`tools/<your_tool>.py` 內 import `register_validator` 來掛 L3 檢查。
"""

from .agent_wrapper import SafeAgentExecutor, run_agent
from .errors import HarnessError, OutputValidationError, SanitizationError
from .input_sanitizer import sanitize_input
from .output_validator import register_validator, validate_output

__all__ = [
    "SafeAgentExecutor",
    "run_agent",
    "HarnessError",
    "SanitizationError",
    "OutputValidationError",
    "sanitize_input",
    "register_validator",
    "validate_output",
]
