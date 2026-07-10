"""query_courses —— 向後相容包裝,委派給 retrieve.py。

原有 import `from tools.query_courses import query_courses_tool` 繼續可用。
功能已移至 `tools.retrieve.retrieve_tool`。
"""
from .retrieve import retrieve_tool as query_courses_tool  # noqa: F401
