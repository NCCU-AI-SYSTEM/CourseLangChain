"""手動端到端驗證(需連 Ollama + 載入 vectorstore.pkl + 送 Langfuse)。

跑:`.venv/bin/python -m tests._e2e_manual`
非自動化測試,故以底線開頭;驗的是「真模型是否照 system prompt 走對工具流程」。
"""
import time

import main
from langfuse import get_client
from tools.query_courses import _get_retriever

g = main.CourseLangGraph()

t = time.time()
_get_retriever()  # 暖快取,避免第一個查詢被 120s timeout 切掉
print(f"[warm retriever] {time.time() - t:.1f}s", flush=True)

QUERIES = [
    ("A 閒聊", "你好,你可以幫我做什麼?"),
    ("B 查課", "給我關於人工智慧的課"),
    ("D 排課", "幫我用管理相關的課排出 6 到 9 學分、避開星期五的課表"),
]

for label, q in QUERIES:
    print("\n" + "=" * 70, flush=True)
    print(f"### {label}:{q}", flush=True)
    t = time.time()
    out = g.invoke(q)
    print(f"--- 回應({time.time() - t:.1f}s)---", flush=True)
    print(out, flush=True)

get_client().flush()
print("\n[langfuse flushed]", flush=True)
