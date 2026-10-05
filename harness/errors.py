"""Harness 自有的 exception 階層。Tool 內部的錯誤不要 raise 這些 — tool 應該
回字串(`"ERROR: ..."`),讓 agent 自行決策。這裡的 exception 只發生在 harness
本身偵測到威脅 / 違規時。"""


class HarnessError(Exception):
    """Harness 層基底 exception。"""


class SanitizationError(HarnessError):
    """L1:輸入被判定為不安全(prompt injection / 違規長度)。"""


class OutputValidationError(HarnessError):
    """L3:輸出未通過註冊的驗證器。"""

    def __init__(self, validator_name: str, message: str):
        self.validator_name = validator_name
        self.message = message
        super().__init__(f"[{validator_name}] {message}")
