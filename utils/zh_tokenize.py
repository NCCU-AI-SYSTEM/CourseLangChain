"""中文斷詞 —— 給 BM25Retriever 當 preprocess_func 用。

問題背景:langchain 的 BM25Retriever 預設 `preprocess_func = str.split()`。中文沒有
空格,整句課名會變成一個巨大 token,查詢關鍵字(如「管理」)永遠對不上 → 每篇分數
都≈0 → BM25 退化成回傳原始列順序(雜訊)。這裡提供一個會真正斷詞的 preprocess_func。

優先用 jieba(品質最好);若環境沒裝 jieba,自動退回「字元 bigram + 單字」備援,
仍比整句一個 token 好得多,且零依賴。
"""
from __future__ import annotations

import re
from typing import List

# 只保留中英數,其餘(標點、空白)當分隔
_TOKEN_CHARS = re.compile(r"[一-鿿]+|[a-zA-Z0-9]+")

try:
    import jieba

    # 關掉 jieba 啟動時的 logging 雜訊
    jieba.setLogLevel(60)
    _HAS_JIEBA = True
except Exception:  # pragma: no cover - 視環境而定
    _HAS_JIEBA = False


def _bigram_fallback(text: str) -> List[str]:
    """無 jieba 時:中文切成單字 + 相鄰 bigram,英數整段保留。"""
    tokens: List[str] = []
    for chunk in _TOKEN_CHARS.findall(text):
        if chunk[0].isascii():
            tokens.append(chunk.lower())
            continue
        # 中文:單字 + bigram,讓「資料庫」既能配「資料」也能配「資料庫」
        tokens.extend(chunk)
        tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return tokens


def tokenize(text: str) -> List[str]:
    """BM25 用的 preprocess_func:回傳 token 串列。"""
    if not text:
        return []
    if _HAS_JIEBA:
        return [t for t in jieba.lcut(text) if t.strip()]
    return _bigram_fallback(text)
