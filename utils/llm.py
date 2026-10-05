"""Chat model factory —— 選哪個供應者只在這裡判斷一次。"""

from langchain_core.language_models.chat_models import BaseChatModel

from paths import (
    GOOGLE_API_KEY,
    GOOGLE_MODEL,
    LLM_PROVIDER,
    MODEL,
    OLLAMA_HOST,
    OLLAMA_NUM_CTX,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)


def get_chat_llm(temperature: float = 0.3, *, extract: bool = False) -> BaseChatModel:
    """回傳設定好的 Chat model。

    extract=True 給抽取型任務(text_to_sql):關掉 thinking、限制輸出長度。
    thinking 模型不關就停不下來 —— 細節見 tools/text_to_sql.py 開頭的註解。
    """
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {}
        if extract:
            # llama.cpp / vLLM 的 chat template 參數,經 LiteLLM 原樣轉送。
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            kwargs["max_tokens"] = 512
        return ChatOpenAI(
            model=OPENAI_MODEL,
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            temperature=temperature,
            **kwargs,
        )

    if LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GOOGLE_MODEL,
            temperature=temperature,
            google_api_key=GOOGLE_API_KEY,
        )

    if LLM_PROVIDER != "ollama":
        raise ValueError(f"未知的 LLM_PROVIDER={LLM_PROVIDER!r}(可用:ollama / google / openai)")

    from langchain_ollama import ChatOllama

    if extract:
        return ChatOllama(
            model=MODEL,
            base_url=OLLAMA_HOST,
            reasoning=False,
            num_predict=512,
            temperature=temperature,
        )
    return ChatOllama(
        model=MODEL,
        base_url=OLLAMA_HOST,
        temperature=temperature,
        num_ctx=OLLAMA_NUM_CTX,
    )
