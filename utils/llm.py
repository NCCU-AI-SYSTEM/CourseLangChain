"""Chat model factory —— 所有 LLM 都走 OpenAI 相容 API。

換模型 / 換供應者只要改 .env 的 OPENAI_BASE_URL / OPENAI_MODEL / OPENAI_API_KEY,
程式不用動。各家端點的寫法見 .env.example。
"""

from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from paths import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

# AIMessage.additional_kwargs 裡存 {tool_call_id: extra_content} 的 key。
_EXTRA_KEY = "tool_call_extra_content"


class _ChatOpenAIKeepExtra(ChatOpenAI):
    """把 tool call 的 `extra_content` 原樣帶回下一輪。

    Gemini 3 的 OpenAI 相容端點在每個 tool call 附上
    `extra_content.google.thought_signature`,下一輪送回對話時必須原封不動帶著,
    否則回 400「Function call is missing a thought_signature」—— agent 在第一次
    呼叫工具之後就壞掉。langchain-openai(1.6.7)解析回應時會丟掉這個欄位,
    所以這裡收起來放進 additional_kwargs,組下一次請求時再塞回去。
    其他端點不會回 extra_content,這層對它們沒有作用。
    """

    def _create_chat_result(self, response: Any, generation_info: dict | None = None):
        result = super()._create_chat_result(response, generation_info)
        data = response if isinstance(response, dict) else response.model_dump()
        for choice, gen in zip(data.get("choices") or [], result.generations):
            _stash(gen.message, (choice.get("message") or {}).get("tool_calls"))
        return result

    def _convert_chunk_to_generation_chunk(self, chunk: dict, *args: Any, **kwargs: Any):
        gen = super()._convert_chunk_to_generation_chunk(chunk, *args, **kwargs)
        if gen is not None:
            choices = chunk.get("choices") or [{}]
            _stash(gen.message, (choices[0].get("delta") or {}).get("tool_calls"))
        return gen

    def _get_request_payload(self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        extras: dict = {}
        for msg in self._convert_input(input_).to_messages():
            if isinstance(msg, AIMessage):
                extras.update(msg.additional_kwargs.get(_EXTRA_KEY) or {})
        if extras:
            for m in payload.get("messages") or []:
                for tc in m.get("tool_calls") or []:
                    if tc.get("id") in extras:
                        tc["extra_content"] = extras[tc["id"]]
        return payload


def _stash(message: Any, raw_tool_calls: list | None) -> None:
    found = {
        tc["id"]: tc["extra_content"]
        for tc in raw_tool_calls or []
        if tc.get("id") and tc.get("extra_content")
    }
    if found:
        message.additional_kwargs.setdefault(_EXTRA_KEY, {}).update(found)


def get_chat_llm(temperature: float = 0.3, *, extract: bool = False, **kwargs) -> ChatOpenAI:
    """回傳設定好的 Chat model。

    extract=True 給抽取型任務(text_to_sql):關掉 thinking、限制輸出長度。
    thinking 模型不關的話，思考會吃掉輸出上限，最後回空字串 —— 細節見
    tools/text_to_sql.py 開頭的註解。

    其餘 kwargs 原樣傳給 ChatOpenAI,會蓋過這裡的預設值。例如端點不吃
    reasoning_effort 時傳 reasoning_effort=None 拿掉它，或用 extra_body 加
    端點自己的參數。
    """
    defaults = {}
    if extract:
        # OpenAI 標準參數。實測 Gemini 與 LiteLLM → llama.cpp 都吃,回應的
        # reasoning 是空的。(llama.cpp 專用的 chat_template_kwargs 會被 Gemini
        # 以 400 拒絕。)LiteLLM 要允許這個參數才會轉送。
        defaults["reasoning_effort"] = "none"
        defaults["max_tokens"] = 512
    return _ChatOpenAIKeepExtra(
        model=OPENAI_MODEL,
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        temperature=temperature,
        **{**defaults, **kwargs},
    )
